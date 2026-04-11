"""Main experiment: uncertainty quantification on the taboo secret word task.

Runs all UQ methods on the taboo task and compares their calibration.
This script handles the full pipeline:
1. Load model with LoRA support
2. For each (target_word, context_prompt, verbalizer_prompt):
   a. Load target LoRA, collect activations
   b. Set up steering hook on oracle
   c. Run each UQ method
   d. Record predictions and confidence scores
3. Compute calibration metrics and save results
"""

import hashlib
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from pao.answer_extraction import extract_predicted_word
from pao.calibration.metrics import (
    auroc,
    confidence_separation,
    expected_calibration_error,
    negative_log_likelihood,
    print_calibration_summary,
)
from pao.calibration.secret_word_calibration import (
    WordPrediction,
)
from pao.config import (
    AO_ROOT,
    TABOO_WORDS,
    VERBALIZER_PROMPTS_TABOO,
    ExperimentConfig,
    ModelConfig,
    SamplingConfig,
)
from pao.hf_utils import (
    SPECIAL_TOKEN,
    collect_activations_multiple_layers,
    encode_messages,
    find_pattern_in_tokens,
    get_hf_submodule,
    get_introspection_prefix,
    load_lora_adapter,
    load_model,
    load_tokenizer,
    set_seed,
)
from pao.methods.logprob_baseline import logprob_confidence
from pao.methods.mcmc_oracle import mcmc_agreement, mcmc_oracle_sample
from pao.methods.steering_sensitivity import steering_sensitivity_confidence
from pao.methods.temperature_bootstrap import temperature_bootstrap
from pao.oracle_sampler import SteeredAutoregressiveSampler

# Bump when any on-disk-incompatible change is made to the experiment logic
# (prompt format, answer extraction, confidence definitions, ...). The value
# is mixed into ``config_hash`` so stale checkpoints fail loudly instead of
# silently appending incompatible predictions.
CODE_VERSION = "2"


def setup_model(
    cfg: ModelConfig,
) -> tuple[AutoModelForCausalLM, AutoTokenizer, torch.device, Optional[str]]:
    """Load model, tokenizer, and eagerly load the verbalizer LoRA.

    The verbalizer adapter is loaded once here and left as the active adapter.
    Target LoRAs are loaded lazily (and cached) in the outer experiment loop.
    """
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(cfg.model_name)
    model = load_model(cfg.model_name, cfg.dtype)
    model.eval()

    # Add dummy adapter so PeftModel API is available
    dummy_config = LoraConfig()
    model.add_adapter(dummy_config, adapter_name="default")

    verbalizer_adapter: Optional[str] = None
    if cfg.verbalizer_lora_path:
        verbalizer_adapter = load_lora_adapter(model, cfg.verbalizer_lora_path)
        model.set_adapter(verbalizer_adapter)

    return model, tokenizer, device, verbalizer_adapter


def prepare_activation_and_sampler(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    cfg: ModelConfig,
    target_adapter: str,
    verbalizer_adapter: Optional[str],
    context_prompt: str,
    verbalizer_prompt: str,
) -> tuple[SteeredAutoregressiveSampler, list[int]]:
    """Collect activations from the already-loaded target LoRA and build a sampler.

    The target adapter must already be loaded (by the caller, once per target_word).
    This function only switches between target and verbalizer adapters — it never
    loads or deletes.

    Returns:
        (sampler, oracle_context_ids). The sampler's hook is NOT attached yet;
        the caller must use ``with sampler:`` or call ``sampler.attach_hook()``.
    """
    # Encode context prompt for activation collection
    formatted_prompt = [{"role": "user", "content": context_prompt}]
    inputs_BL = encode_messages(
        tokenizer=tokenizer,
        message_dicts=[formatted_prompt],
        add_generation_prompt=True,
        enable_thinking=False,
        device=device,
    )

    # Collect activations from the target LoRA model
    act_layer = int(model.config.num_hidden_layers * (cfg.selected_layer_percent / 100))

    model.set_adapter(target_adapter)
    submodules = {act_layer: get_hf_submodule(model, act_layer)}
    acts_by_layer = collect_activations_multiple_layers(
        model=model,
        submodules=submodules,
        inputs_BL=inputs_BL,
        min_offset=None,
        max_offset=None,
    )

    # Switch back to the verbalizer adapter for oracle generation.
    if verbalizer_adapter is not None:
        model.set_adapter(verbalizer_adapter)

    # Extract activation vectors for steering
    acts_BLD = acts_by_layer[act_layer]  # (1, L, D)
    seq_len = acts_BLD.shape[1]

    # Use segment positions (last 10 tokens by default)
    start = max(0, seq_len + cfg.segment_start_idx)
    end = (
        seq_len + cfg.segment_end_idx
        if cfg.segment_end_idx <= 0
        else cfg.segment_end_idx
    )
    positions_rel = list(range(start, end))

    steering_vectors = [acts_BLD[0, positions_rel, :]]  # list of (K, D)

    # Build the oracle prompt using the exact chat-template format the oracle
    # LoRA was trained with:
    #   user message = get_introspection_prefix(layer, K) + verbalizer_prompt
    #   apply_chat_template(add_generation_prompt=True, enable_thinking=False)
    injection_layer = cfg.injection_layer
    num_positions = len(positions_rel)
    oracle_user_content = (
        get_introspection_prefix(act_layer, num_positions) + verbalizer_prompt
    )
    oracle_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": oracle_user_content}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(oracle_ids, list):
        raise TypeError("Expected list of token ids from apply_chat_template")

    # Steering positions land on the SPECIAL_TOKEN (" ?") occurrences inside
    # the rendered user message. find_pattern_in_tokens asserts they are a
    # consecutive run followed by a newline, matching the training layout.
    steering_positions = [
        find_pattern_in_tokens(oracle_ids, SPECIAL_TOKEN, num_positions, tokenizer)
    ]

    # Get injection submodule
    injection_submodule = get_hf_submodule(model, injection_layer)

    sampler = SteeredAutoregressiveSampler(
        model=model,
        tokenizer=tokenizer,
        device=device,
        submodule=injection_submodule,
        steering_vectors=steering_vectors,
        positions=steering_positions,
        steering_coefficient=1.0,
        dtype=cfg.dtype,
    )

    return sampler, oracle_ids


def config_hash(exp_cfg: ExperimentConfig) -> str:
    """Return a deterministic hash of config fields that affect iteration identity.

    Excludes output_dir, device, and dtype since they don't change which results
    are produced, only where/how they're stored/computed.
    """
    sampling = exp_cfg.sampling
    key = {
        "code_version": CODE_VERSION,
        "model_name": exp_cfg.model.model_name,
        "verbalizer_lora_path": exp_cfg.model.verbalizer_lora_path,
        "target_lora_template": exp_cfg.model.target_lora_template,
        "injection_layer": exp_cfg.model.injection_layer,
        "selected_layer_percent": exp_cfg.model.selected_layer_percent,
        "segment_start_idx": exp_cfg.model.segment_start_idx,
        "segment_end_idx": exp_cfg.model.segment_end_idx,
        "context_prompt_file": exp_cfg.context_prompt_file,
        "max_context_prompts": exp_cfg.max_context_prompts,
        "seed": exp_cfg.seed,
        "bootstrap_k": sampling.bootstrap_k,
        "bootstrap_temperatures": sampling.bootstrap_temperatures,
        "mcmc_temperatures": sampling.mcmc_temperatures,
        "mcmc_steps": sampling.mcmc_steps,
        "mcmc_block_num": sampling.mcmc_block_num,
        "mcmc_max_new_tokens": sampling.mcmc_max_new_tokens,
        "power_agreement_k": sampling.power_agreement_k,
        "sensitivity_coefficients": sampling.sensitivity_coefficients,
        "max_new_tokens": sampling.max_new_tokens,
    }
    serialized = json.dumps(key, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def save_checkpoint(
    path: Path,
    predictions: dict[str, list[WordPrediction]],
    exp_cfg: ExperimentConfig,
    total_completed: int,
    total_expected: int,
) -> None:
    """Atomically write checkpoint to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_completed": total_completed,
            "total_expected": total_expected,
            "config_hash": config_hash(exp_cfg),
        },
        "predictions": {
            method: [asdict(p) for p in preds] for method, preds in predictions.items()
        },
    }
    tmp_path = path.with_suffix(".tmp.json")
    with open(tmp_path, "w") as f:
        json.dump(checkpoint, f)
    os.replace(tmp_path, path)


def load_checkpoint(
    path: Path,
    exp_cfg: ExperimentConfig,
) -> tuple[dict[str, list[WordPrediction]], set[tuple[str, str, str]]]:
    """Load checkpoint, validate config compatibility, return predictions and completed keys."""
    try:
        with open(path) as f:
            checkpoint = json.load(f)
    except (json.JSONDecodeError, OSError):
        tmp_path = path.with_suffix(".tmp.json")
        print(f"Warning: checkpoint.json unreadable, trying {tmp_path.name}")
        with open(tmp_path) as f:
            checkpoint = json.load(f)

    saved_hash = checkpoint["metadata"]["config_hash"]
    current_hash = config_hash(exp_cfg)
    if saved_hash != current_hash:
        raise ValueError(
            f"Checkpoint config mismatch (saved: {saved_hash}, current: {current_hash}). "
            "Use --no-resume to start fresh."
        )

    method_names = ["logprob", "bootstrap", "sensitivity", "mcmc", "mcmc_agreement"]
    raw = checkpoint["predictions"]
    lengths = [len(raw.get(m, [])) for m in method_names]
    min_len = min(lengths)
    if len(set(lengths)) > 1:
        print(
            f"Warning: method list length mismatch {dict(zip(method_names, lengths))}; truncating to {min_len}"
        )

    predictions: dict[str, list[WordPrediction]] = {
        m: [WordPrediction(**d) for d in raw.get(m, [])[:min_len]] for m in method_names
    }

    # Build completed keys from any one method (all are written atomically together)
    completed_keys: set[tuple[str, str, str]] = {
        (p.target_word, p.verbalizer_prompt, p.context_prompt)
        for p in predictions["logprob"]
    }
    return predictions, completed_keys


ExperimentState = tuple[
    AutoModelForCausalLM, AutoTokenizer, torch.device, Optional[str], list[str]
]


def setup_experiment_state(exp_cfg: ExperimentConfig) -> ExperimentState:
    """Load model, tokenizer, and context prompts. Everything except the loop."""
    set_seed(exp_cfg.seed)
    model, tokenizer, device, verbalizer_adapter = setup_model(exp_cfg.model)

    ctx_file = AO_ROOT / exp_cfg.context_prompt_file
    with open(ctx_file) as f:
        context_prompts = [line.strip() for line in f if line.strip()]

    if exp_cfg.max_context_prompts is not None:
        context_prompts = context_prompts[: exp_cfg.max_context_prompts]

    return model, tokenizer, device, verbalizer_adapter, context_prompts


def run_all_methods(
    exp_cfg: ExperimentConfig,
    checkpoint_every: int = 50,
    no_resume: bool = False,
    max_iterations: Optional[int] = None,
    preloaded_state: Optional[ExperimentState] = None,
) -> dict[str, list[WordPrediction]]:
    """Run all UQ methods on the taboo task.

    If ``preloaded_state`` is given, skips model loading and reuses it.
    If ``max_iterations`` is set, stops after processing that many *new*
    iterations (i.e. iterations not already in the checkpoint) and saves.
    Returns a dict mapping method_name -> list of WordPrediction.
    """
    if preloaded_state is None:
        preloaded_state = setup_experiment_state(exp_cfg)
    model, tokenizer, device, verbalizer_adapter, context_prompts = preloaded_state

    verbalizer_prompts = VERBALIZER_PROMPTS_TABOO
    target_words = TABOO_WORDS
    sampling = exp_cfg.sampling

    all_predictions: dict[str, list[WordPrediction]] = {
        "logprob": [],
        "bootstrap": [],
        "sensitivity": [],
        "mcmc": [],
        "mcmc_agreement": [],
    }

    total = len(target_words) * len(verbalizer_prompts) * len(context_prompts)
    checkpoint_path = Path(exp_cfg.output_dir) / "checkpoint.json"
    completed_keys: set[tuple[str, str, str]] = set()

    if not no_resume and checkpoint_path.exists():
        all_predictions, completed_keys = load_checkpoint(checkpoint_path, exp_cfg)
        print(
            f"Resumed from checkpoint: {len(completed_keys)}/{total} iterations already completed"
        )

    pbar = tqdm(total=total, initial=len(completed_keys), desc="Running UQ methods")
    iterations_since_save = 0
    processed_this_call = 0
    hit_cap = False

    for target_word in target_words:
        if hit_cap:
            break
        # Load (and cache) the target LoRA exactly once per target word.
        target_lora_path = exp_cfg.model.target_lora_template.format(word=target_word)
        target_adapter = load_lora_adapter(model, target_lora_path)

        for verbalizer_prompt in verbalizer_prompts:
            if hit_cap:
                break
            for ctx_prompt in context_prompts:
                if (target_word, verbalizer_prompt, ctx_prompt) in completed_keys:
                    pbar.update(1)
                    continue

                sampler, oracle_ids = prepare_activation_and_sampler(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    cfg=exp_cfg.model,
                    target_adapter=target_adapter,
                    verbalizer_adapter=verbalizer_adapter,
                    context_prompt=ctx_prompt,
                    verbalizer_prompt=verbalizer_prompt,
                )

                with sampler:
                    # --- Method 1: Log-prob baseline ---
                    # Confidence = max first-token probability. The chat template
                    # puts the assistant's first content token right after the
                    # generation prompt, so this directly measures how peaked
                    # the oracle is on one candidate word.
                    lp_result = logprob_confidence(
                        sampler=sampler,
                        context=oracle_ids,
                        max_new_tokens=sampling.max_new_tokens,
                    )
                    lp_word = extract_predicted_word(
                        lp_result.generated_text, TABOO_WORDS
                    )
                    all_predictions["logprob"].append(
                        WordPrediction(
                            target_word=target_word,
                            context_prompt=ctx_prompt,
                            verbalizer_prompt=verbalizer_prompt,
                            predicted_answer=lp_result.generated_text,
                            confidence=lp_result.first_token_max_prob,
                            is_correct=lp_word == target_word,
                            method="logprob",
                            method_metadata={
                                "extracted_word": lp_word,
                                "first_token_entropy": lp_result.first_token_entropy,
                                "first_token_max_prob": lp_result.first_token_max_prob,
                                "mean_log_prob": lp_result.mean_log_prob,
                                "min_log_prob": lp_result.min_log_prob,
                                "geometric_mean_prob": lp_result.geometric_mean_prob,
                            },
                        )
                    )

                    # --- Method 2: Temperature bootstrap ---
                    boot_result = temperature_bootstrap(
                        sampler=sampler,
                        context=oracle_ids,
                        answer_vocab=TABOO_WORDS,
                        k=sampling.bootstrap_k,
                        temperature=sampling.bootstrap_temperatures[0],
                        max_new_tokens=sampling.max_new_tokens,
                    )
                    all_predictions["bootstrap"].append(
                        WordPrediction(
                            target_word=target_word,
                            context_prompt=ctx_prompt,
                            verbalizer_prompt=verbalizer_prompt,
                            predicted_answer=boot_result.mode_answer,
                            confidence=boot_result.mode_frequency,
                            is_correct=boot_result.mode_answer == target_word,
                            method="bootstrap",
                            method_metadata={
                                "raw_samples": boot_result.samples,
                                "normalized_samples": boot_result.normalized_samples,
                                "entropy": boot_result.entropy,
                                "num_unique": boot_result.num_unique,
                                "temperature": boot_result.temperature,
                                "k": boot_result.k,
                            },
                        )
                    )

                    # --- Method 6: Steering-coefficient sensitivity ---
                    sens_result = steering_sensitivity_confidence(
                        sampler=sampler,
                        context=oracle_ids,
                        coefficients=sampling.sensitivity_coefficients,
                        answer_vocab=TABOO_WORDS,
                        max_new_tokens=sampling.max_new_tokens,
                    )
                    all_predictions["sensitivity"].append(
                        WordPrediction(
                            target_word=target_word,
                            context_prompt=ctx_prompt,
                            verbalizer_prompt=verbalizer_prompt,
                            predicted_answer=sens_result.mode_answer,
                            confidence=sens_result.mode_frequency,
                            is_correct=sens_result.mode_answer == target_word,
                            method="sensitivity",
                            method_metadata={
                                "coefficients": sens_result.coefficients,
                                "answers": sens_result.answers,
                                "normalized_answers": sens_result.normalized_answers,
                                "per_coef_mean_logprob": sens_result.per_coef_mean_logprob,
                                "entropy": sens_result.entropy,
                                "num_unique": sens_result.num_unique,
                            },
                        )
                    )

                    # --- Method 3: Single MCMC sample (with acceptance ratio as signal) ---
                    mcmc_result = mcmc_oracle_sample(
                        sampler=sampler,
                        context=oracle_ids,
                        temperature=sampling.mcmc_temperatures[0],
                        mcmc_steps=sampling.mcmc_steps,
                        max_new_tokens=sampling.mcmc_max_new_tokens,
                        block_num=sampling.mcmc_block_num,
                    )
                    mcmc_word = extract_predicted_word(
                        mcmc_result.generated_text, TABOO_WORDS
                    )
                    all_predictions["mcmc"].append(
                        WordPrediction(
                            target_word=target_word,
                            context_prompt=ctx_prompt,
                            verbalizer_prompt=verbalizer_prompt,
                            predicted_answer=mcmc_result.generated_text,
                            confidence=1.0
                            - mcmc_result.acceptance_ratio,  # low acceptance = high confidence
                            is_correct=mcmc_word == target_word,
                            method="mcmc",
                            method_metadata={
                                "extracted_word": mcmc_word,
                                "acceptance_ratio": mcmc_result.acceptance_ratio,
                                "alpha": mcmc_result.alpha,
                                "temperature": mcmc_result.temperature,
                            },
                        )
                    )

                    # --- Method 4: MCMC agreement ---
                    agree_result = mcmc_agreement(
                        sampler=sampler,
                        context=oracle_ids,
                        answer_vocab=TABOO_WORDS,
                        k=sampling.power_agreement_k,
                        temperature=sampling.mcmc_temperatures[0],
                        mcmc_steps=sampling.mcmc_steps,
                        max_new_tokens=sampling.mcmc_max_new_tokens,
                        block_num=sampling.mcmc_block_num,
                    )
                    all_predictions["mcmc_agreement"].append(
                        WordPrediction(
                            target_word=target_word,
                            context_prompt=ctx_prompt,
                            verbalizer_prompt=verbalizer_prompt,
                            predicted_answer=agree_result.mode_answer,
                            confidence=agree_result.mode_frequency,
                            is_correct=agree_result.mode_answer == target_word,
                            method="mcmc_agreement",
                            method_metadata={
                                "normalized_samples": agree_result.normalized_samples,
                                "entropy": agree_result.entropy,
                                "num_unique": agree_result.num_unique,
                                "mean_acceptance_ratio": agree_result.mean_acceptance_ratio,
                                "k": agree_result.k,
                                "alpha": agree_result.alpha,
                            },
                        )
                    )

                pbar.update(1)
                iterations_since_save += 1
                processed_this_call += 1
                if iterations_since_save >= checkpoint_every:
                    save_checkpoint(
                        checkpoint_path, all_predictions, exp_cfg, pbar.n, total
                    )
                    iterations_since_save = 0

                if max_iterations is not None and processed_this_call >= max_iterations:
                    hit_cap = True
                    break

            if hit_cap:
                break
            # Save at the natural boundary (all context prompts for one word+verbalizer done)
            save_checkpoint(checkpoint_path, all_predictions, exp_cfg, pbar.n, total)
            iterations_since_save = 0

    # Final save to capture any last iterations not yet flushed
    save_checkpoint(checkpoint_path, all_predictions, exp_cfg, pbar.n, total)
    pbar.close()
    return all_predictions


def print_smoke_report(
    predictions: dict[str, list[WordPrediction]],
    cfg: ExperimentConfig,
) -> None:
    """Pretty-print the smoke-test results so the user can sanity-check them."""
    method_names = ["logprob", "bootstrap", "sensitivity", "mcmc", "mcmc_agreement"]
    n = min(len(predictions[m]) for m in method_names)
    if n == 0:
        print("  (no iterations processed — already fully resumed?)")
        return

    total = (
        len(TABOO_WORDS)
        * len(VERBALIZER_PROMPTS_TABOO)
        * (cfg.max_context_prompts or 99999)
    )
    print(f"\n{'=' * 78}")
    print(f"  Smoke test — {n} iteration(s) completed (of {total} total)")
    print(f"{'=' * 78}")

    for i in range(n):
        target = predictions["logprob"][i].target_word
        ctx = predictions["logprob"][i].context_prompt
        verb = predictions["logprob"][i].verbalizer_prompt
        print(f"\n  [{i + 1}] target={target!r}")
        print(f"      context:    {ctx[:70]}{'...' if len(ctx) > 70 else ''}")
        print(f"      verbalizer: {verb[:70]}{'...' if len(verb) > 70 else ''}")
        for m in method_names:
            p = predictions[m][i]
            ok = "✓" if p.is_correct else "✗"
            extracted = p.method_metadata.get(
                "extracted_word",
                p.method_metadata.get("normalized_samples", [p.predicted_answer])[0]
                if isinstance(p.method_metadata.get("normalized_samples"), list)
                else p.predicted_answer,
            )
            raw = p.predicted_answer.replace("\n", " ")[:40]
            print(
                f"      {m:<16} {ok}  extracted={extracted!r:<12} "
                f"conf={p.confidence:.3f}  raw={raw!r}"
            )
    print(f"\n{'=' * 78}")


def evaluate_and_save(
    all_predictions: dict[str, list[WordPrediction]],
    output_dir: str,
):
    """Compute calibration metrics for each method and save results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary = {}

    for method_name, predictions in all_predictions.items():
        confidences = [p.confidence for p in predictions]
        correctness = [p.is_correct for p in predictions]

        cal = expected_calibration_error(confidences, correctness)
        nll = negative_log_likelihood(confidences, correctness)
        accuracy = sum(correctness) / max(len(correctness), 1)
        auroc_score = auroc(confidences, correctness)
        conf_correct, conf_wrong = confidence_separation(confidences, correctness)

        print_calibration_summary(cal, method_name)
        print(f"  Accuracy:           {accuracy:.3f}")
        print(f"  NLL:                {nll:.4f}")
        print(f"  AUROC:              {auroc_score:.3f}")
        print(f"  Conf|correct:       {conf_correct:.3f}")
        print(f"  Conf|wrong:         {conf_wrong:.3f}")

        summary[method_name] = {
            "accuracy": accuracy,
            "ece": cal.ece,
            "brier_score": cal.brier_score,
            "nll": nll,
            "auroc": auroc_score,
            "conf_given_correct": conf_correct,
            "conf_given_wrong": conf_wrong,
            "n_samples": len(predictions),
        }

        # Save per-method details
        method_output = {
            "method": method_name,
            "accuracy": accuracy,
            "calibration": {
                "ece": cal.ece,
                "brier_score": cal.brier_score,
                "bin_confidences": cal.bin_confidences,
                "bin_accuracies": cal.bin_accuracies,
                "bin_counts": cal.bin_counts,
            },
            "nll": nll,
            "auroc": auroc_score,
            "conf_given_correct": conf_correct,
            "conf_given_wrong": conf_wrong,
            "predictions": [asdict(p) for p in predictions],
        }

        with open(output_path / f"{method_name}_results.json", "w") as f:
            json.dump(method_output, f, indent=2)

    # Save summary comparison
    with open(output_path / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 78}")
    print("  Method Comparison Summary")
    print(f"{'=' * 78}")
    print(
        f"  {'Method':<18} {'Acc':>8} {'ECE':>8} {'Brier':>8} "
        f"{'NLL':>8} {'AUROC':>8} {'C|ok':>7} {'C|err':>7}"
    )
    print(f"{'─' * 78}")
    for method, stats in summary.items():
        print(
            f"  {method:<18} {stats['accuracy']:>8.3f} {stats['ece']:>8.4f} "
            f"{stats['brier_score']:>8.4f} {stats['nll']:>8.4f} "
            f"{stats['auroc']:>8.3f} "
            f"{stats['conf_given_correct']:>7.3f} {stats['conf_given_wrong']:>7.3f}"
        )
    print(f"{'=' * 78}")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run taboo UQ experiments")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="Base model name")
    parser.add_argument(
        "--max-prompts", type=int, default=None, help="Limit context prompts"
    )
    parser.add_argument(
        "--output-dir", default="results/taboo_uq", help="Output directory"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-k", type=int, default=20)
    parser.add_argument("--mcmc-steps", type=int, default=5)
    parser.add_argument("--mcmc-temp", type=float, default=0.25)
    parser.add_argument("--agreement-k", type=int, default=10)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Save checkpoint every N completed iterations (default: 50)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore any existing checkpoint and start fresh",
    )
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Skip the smoke-test phase and go straight to the full run",
    )
    parser.add_argument(
        "--smoke-iterations",
        type=int,
        default=1,
        help="Number of iterations to run in the smoke-test phase (default: 1)",
    )
    args = parser.parse_args()

    os.chdir(AO_ROOT)  # So dataset paths resolve correctly

    cfg = ExperimentConfig(
        model=ModelConfig(model_name=args.model),
        sampling=SamplingConfig(
            bootstrap_k=args.bootstrap_k,
            mcmc_steps=args.mcmc_steps,
            mcmc_temperatures=[args.mcmc_temp],
            power_agreement_k=args.agreement_k,
        ),
        output_dir=args.output_dir,
        max_context_prompts=args.max_prompts,
        seed=args.seed,
    )

    import sys

    # Load model + prompts once; reused for smoke and full run.
    state = setup_experiment_state(cfg)

    checkpoint_path = Path(cfg.output_dir) / "checkpoint.json"
    resuming = not args.no_resume and checkpoint_path.exists()

    if not args.skip_smoke and not resuming:
        print(
            f"\n>>> Smoke test: running {args.smoke_iterations} iteration(s) "
            "to validate the pipeline before the full sweep"
        )
        smoke_predictions = run_all_methods(
            cfg,
            checkpoint_every=args.checkpoint_every,
            no_resume=args.no_resume,
            max_iterations=args.smoke_iterations,
            preloaded_state=state,
        )
        print_smoke_report(smoke_predictions, cfg)
        try:
            resp = input("\nProceed with the full run? [y/N]: ").strip().lower()
        except EOFError:
            resp = ""
        if resp not in ("y", "yes"):
            print("Aborted. The smoke-test iterations are preserved in the checkpoint")
            print(f"at {checkpoint_path} — rerun to resume from there.")
            sys.exit(0)

    predictions = run_all_methods(
        cfg,
        checkpoint_every=args.checkpoint_every,
        no_resume=args.no_resume,
        preloaded_state=state,
    )
    evaluate_and_save(predictions, cfg.output_dir)
