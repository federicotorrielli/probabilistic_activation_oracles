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

import os
import sys
import json
import random
from pathlib import Path
from typing import Optional
from dataclasses import asdict

import torch
from tqdm import tqdm
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure submodule imports work
from pao.config import (
    PROJECT_ROOT, AO_ROOT,
    TABOO_WORDS, VERBALIZER_PROMPTS_TABOO,
    ModelConfig, SamplingConfig, ExperimentConfig,
)

from nl_probes.utils.common import load_model, load_tokenizer, set_seed
from nl_probes.utils.activation_utils import (
    collect_activations_multiple_layers,
    get_hf_submodule,
)
from nl_probes.utils.dataset_utils import create_training_datapoint, SPECIAL_TOKEN
from nl_probes.base_experiment import (
    VerbalizerEvalConfig,
    VerbalizerInputInfo,
    encode_messages,
    collect_target_activations,
    load_lora_adapter,
    sanitize_lora_name,
)

from pao.oracle_sampler import SteeredAutoregressiveSampler
from pao.methods.logprob_baseline import logprob_confidence, LogProbResult
from pao.methods.temperature_bootstrap import temperature_bootstrap, BootstrapResult
from pao.methods.mcmc_oracle import mcmc_oracle_sample, mcmc_agreement, MCMCResult, MCMCAgreementResult
from pao.methods.steering_sensitivity import steering_sensitivity_confidence, SensitivityResult
from pao.calibration.metrics import (
    expected_calibration_error,
    negative_log_likelihood,
    print_calibration_summary,
)
from pao.calibration.secret_word_calibration import (
    WordPrediction,
    CalibrationTestResult,
    save_calibration_results,
)


def setup_model(cfg: ModelConfig) -> tuple[AutoModelForCausalLM, AutoTokenizer, torch.device]:
    """Load model, tokenizer, and set up LoRA support."""
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(cfg.model_name)
    model = load_model(cfg.model_name, cfg.dtype)
    model.eval()

    # Add dummy adapter so PeftModel API is available
    dummy_config = LoraConfig()
    model.add_adapter(dummy_config, adapter_name="default")

    return model, tokenizer, device


def prepare_activation_and_sampler(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    cfg: ModelConfig,
    target_word: str,
    context_prompt: str,
    verbalizer_prompt: str,
) -> tuple[SteeredAutoregressiveSampler, list[int]]:
    """Collect activations from target LoRA and set up a steered sampler.

    Returns:
        (sampler, oracle_context_ids) where sampler has hook NOT yet attached.
        Caller must use `with sampler:` or call sampler.attach_hook().
    """
    target_lora_path = cfg.target_lora_template.format(word=target_word)
    sanitized_target = load_lora_adapter(model, target_lora_path)

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

    model.enable_adapters()
    model.set_adapter(sanitized_target)
    submodules = {act_layer: get_hf_submodule(model, act_layer)}
    acts_by_layer = collect_activations_multiple_layers(
        model=model,
        submodules=submodules,
        inputs_BL=inputs_BL,
        min_offset=None,
        max_offset=None,
    )
    model.disable_adapters()

    # Extract activation vectors for steering
    acts_BLD = acts_by_layer[act_layer]  # (1, L, D)
    seq_len = acts_BLD.shape[1]

    # Use segment positions (last 10 tokens by default)
    start = max(0, seq_len + cfg.segment_start_idx)
    end = seq_len + cfg.segment_end_idx if cfg.segment_end_idx <= 0 else cfg.segment_end_idx
    positions_rel = list(range(start, end))

    steering_vectors = [acts_BLD[0, positions_rel, :]]  # list of (K, D)
    steering_positions = [positions_rel]  # list of list[int]

    # Build the oracle prompt with steering tokens
    injection_layer = cfg.injection_layer
    num_positions = len(positions_rel)
    introspection_prefix = f"Layer: {act_layer}\n" + SPECIAL_TOKEN * num_positions + " \n"
    oracle_prompt = introspection_prefix + verbalizer_prompt
    oracle_ids = tokenizer.encode(oracle_prompt, add_special_tokens=True)

    # Adjust steering positions to match where SPECIAL_TOKEN appears in oracle_ids
    # The steering positions in the oracle prompt correspond to the " ?" tokens
    special_token_id = tokenizer.encode(SPECIAL_TOKEN, add_special_tokens=False)
    if len(special_token_id) == 1:
        special_id = special_token_id[0]
        special_positions = [i for i, tid in enumerate(oracle_ids) if tid == special_id]
        if len(special_positions) >= num_positions:
            steering_positions = [special_positions[:num_positions]]

    # Get injection submodule
    injection_submodule = get_hf_submodule(model, injection_layer)

    # Load verbalizer LoRA
    if cfg.verbalizer_lora_path:
        sanitized_verbalizer = load_lora_adapter(model, cfg.verbalizer_lora_path)
        model.set_adapter(sanitized_verbalizer)

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

    # Clean up target adapter
    if sanitized_target in model.peft_config:
        model.delete_adapter(sanitized_target)

    return sampler, oracle_ids


def run_all_methods(
    exp_cfg: ExperimentConfig,
) -> dict[str, list[WordPrediction]]:
    """Run all UQ methods on the taboo task.

    Returns:
        Dict mapping method_name -> list of WordPrediction.
    """
    set_seed(exp_cfg.seed)
    model, tokenizer, device = setup_model(exp_cfg.model)

    # Load context prompts
    ctx_file = AO_ROOT / exp_cfg.context_prompt_file
    with open(ctx_file) as f:
        context_prompts = [line.strip() for line in f if line.strip()]

    if exp_cfg.max_context_prompts is not None:
        context_prompts = context_prompts[:exp_cfg.max_context_prompts]

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
    pbar = tqdm(total=total, desc="Running UQ methods")

    for target_word in target_words:
        for verbalizer_prompt in verbalizer_prompts:
            for ctx_prompt in context_prompts:
                sampler, oracle_ids = prepare_activation_and_sampler(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    cfg=exp_cfg.model,
                    target_word=target_word,
                    context_prompt=ctx_prompt,
                    verbalizer_prompt=verbalizer_prompt,
                )

                with sampler:
                    # --- Method 1: Log-prob baseline ---
                    lp_result = logprob_confidence(
                        sampler=sampler,
                        context=oracle_ids,
                        max_new_tokens=sampling.max_new_tokens,
                    )
                    all_predictions["logprob"].append(WordPrediction(
                        target_word=target_word,
                        context_prompt=ctx_prompt,
                        verbalizer_prompt=verbalizer_prompt,
                        predicted_answer=lp_result.generated_text,
                        confidence=lp_result.normalized_prob,
                        is_correct=lp_result.generated_text.strip().lower().rstrip(".!?,;:") == target_word,
                        method="logprob",
                        method_metadata={
                            "mean_log_prob": lp_result.mean_log_prob,
                            "min_log_prob": lp_result.min_log_prob,
                            "first_token_entropy": lp_result.first_token_entropy,
                        },
                    ))

                    # --- Method 2: Temperature bootstrap ---
                    boot_result = temperature_bootstrap(
                        sampler=sampler,
                        context=oracle_ids,
                        k=sampling.bootstrap_k,
                        temperature=sampling.bootstrap_temperatures[0],
                        max_new_tokens=sampling.max_new_tokens,
                    )
                    all_predictions["bootstrap"].append(WordPrediction(
                        target_word=target_word,
                        context_prompt=ctx_prompt,
                        verbalizer_prompt=verbalizer_prompt,
                        predicted_answer=boot_result.mode_answer,
                        confidence=boot_result.mode_frequency,
                        is_correct=boot_result.mode_answer == target_word,
                        method="bootstrap",
                        method_metadata={
                            "entropy": boot_result.entropy,
                            "num_unique": boot_result.num_unique,
                            "temperature": boot_result.temperature,
                            "k": boot_result.k,
                        },
                    ))

                    # --- Method 6: Steering-coefficient sensitivity ---
                    sens_result = steering_sensitivity_confidence(
                        sampler=sampler,
                        context=oracle_ids,
                        coefficients=sampling.sensitivity_coefficients,
                        max_new_tokens=sampling.max_new_tokens,
                    )
                    all_predictions["sensitivity"].append(WordPrediction(
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
                            "per_coef_mean_logprob": sens_result.per_coef_mean_logprob,
                            "entropy": sens_result.entropy,
                            "num_unique": sens_result.num_unique,
                        },
                    ))

                    # --- Method 3: Single MCMC sample (with acceptance ratio as signal) ---
                    mcmc_result = mcmc_oracle_sample(
                        sampler=sampler,
                        context=oracle_ids,
                        temperature=sampling.mcmc_temperatures[0],
                        mcmc_steps=sampling.mcmc_steps,
                        max_new_tokens=sampling.mcmc_max_new_tokens,
                        block_num=sampling.mcmc_block_num,
                    )
                    mcmc_answer = mcmc_result.generated_text.strip().lower().rstrip(".!?,;:")
                    all_predictions["mcmc"].append(WordPrediction(
                        target_word=target_word,
                        context_prompt=ctx_prompt,
                        verbalizer_prompt=verbalizer_prompt,
                        predicted_answer=mcmc_result.generated_text,
                        confidence=1.0 - mcmc_result.acceptance_ratio,  # low acceptance = high confidence
                        is_correct=mcmc_answer == target_word,
                        method="mcmc",
                        method_metadata={
                            "acceptance_ratio": mcmc_result.acceptance_ratio,
                            "alpha": mcmc_result.alpha,
                            "temperature": mcmc_result.temperature,
                        },
                    ))

                    # --- Method 4: MCMC agreement ---
                    agree_result = mcmc_agreement(
                        sampler=sampler,
                        context=oracle_ids,
                        k=sampling.power_agreement_k,
                        temperature=sampling.mcmc_temperatures[0],
                        mcmc_steps=sampling.mcmc_steps,
                        max_new_tokens=sampling.mcmc_max_new_tokens,
                        block_num=sampling.mcmc_block_num,
                    )
                    all_predictions["mcmc_agreement"].append(WordPrediction(
                        target_word=target_word,
                        context_prompt=ctx_prompt,
                        verbalizer_prompt=verbalizer_prompt,
                        predicted_answer=agree_result.mode_answer,
                        confidence=agree_result.mode_frequency,
                        is_correct=agree_result.mode_answer == target_word,
                        method="mcmc_agreement",
                        method_metadata={
                            "entropy": agree_result.entropy,
                            "num_unique": agree_result.num_unique,
                            "mean_acceptance_ratio": agree_result.mean_acceptance_ratio,
                            "k": agree_result.k,
                            "alpha": agree_result.alpha,
                        },
                    ))

                pbar.update(1)

    pbar.close()
    return all_predictions


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

        print_calibration_summary(cal, method_name)
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  NLL: {nll:.4f}")

        summary[method_name] = {
            "accuracy": accuracy,
            "ece": cal.ece,
            "brier_score": cal.brier_score,
            "nll": nll,
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
            "predictions": [asdict(p) for p in predictions],
        }

        with open(output_path / f"{method_name}_results.json", "w") as f:
            json.dump(method_output, f, indent=2)

    # Save summary comparison
    with open(output_path / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print("  Method Comparison Summary")
    print(f"{'=' * 60}")
    print(f"  {'Method':<20} {'Accuracy':>10} {'ECE':>10} {'Brier':>10} {'NLL':>10}")
    print(f"{'─' * 60}")
    for method, stats in summary.items():
        print(
            f"  {method:<20} {stats['accuracy']:>10.3f} {stats['ece']:>10.4f} "
            f"{stats['brier_score']:>10.4f} {stats['nll']:>10.4f}"
        )
    print(f"{'=' * 60}")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run taboo UQ experiments")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="Base model name")
    parser.add_argument("--max-prompts", type=int, default=None, help="Limit context prompts")
    parser.add_argument("--output-dir", default="results/taboo_uq", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-k", type=int, default=20)
    parser.add_argument("--mcmc-steps", type=int, default=5)
    parser.add_argument("--mcmc-temp", type=float, default=0.25)
    parser.add_argument("--agreement-k", type=int, default=10)
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

    predictions = run_all_methods(cfg)
    evaluate_and_save(predictions, cfg.output_dir)
