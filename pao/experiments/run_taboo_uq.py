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
import random
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import matplotlib
import torch
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from pao.answer_extraction import extract_predicted_word
from pao.calibration.metrics import (
    CalibrationResult,
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
    MODEL_PRESETS,
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
    get_text_config,
    load_lora_adapter,
    load_model,
    load_tokenizer,
    set_seed,
)
from pao.methods.direct_elicitation import (
    direct_elicitation,
    score_linguistic_confidence,
)
from pao.methods.logprob_baseline import logprob_confidence
from pao.methods.mcmc_oracle import mcmc_agreement, mcmc_oracle_sample
from pao.methods.steering_sensitivity import steering_sensitivity_confidence
from pao.methods.temperature_bootstrap import temperature_bootstrap
from pao.oracle_sampler import SteeredAutoregressiveSampler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Bump when any on-disk-incompatible change is made to the experiment logic
# (prompt format, answer extraction, confidence definitions, ...). The value
# is mixed into ``config_hash`` so stale checkpoints fail loudly instead of
# silently appending incompatible predictions.
CODE_VERSION = "12"


def temperature_tag(temp: float) -> str:
    """Format a temperature for stable method names / filenames."""
    return str(temp).replace(".", "p")


def get_method_names(sampling: SamplingConfig) -> list[str]:
    """Return all per-run method names in a stable order."""
    method_names = [
        "logprob_offset",
        "logprob_no_offset",
        "direct",
    ]
    if sampling.direct_linguistic_enabled:
        method_names.extend(
            [
                "direct_linguistic_expected",
                "direct_linguistic_p_very_high",
                "direct_linguistic_p_high_plus",
            ]
        )
    method_names.extend(
        f"bootstrap_t{temperature_tag(temp)}"
        for temp in sampling.bootstrap_temperatures
    )
    method_names.append("sensitivity")
    method_names.extend(
        f"mcmc_t{temperature_tag(temp)}" for temp in sampling.mcmc_temperatures
    )
    method_names.extend(
        f"mcmc_agreement_t{temperature_tag(temp)}"
        for temp in sampling.mcmc_temperatures
    )
    return method_names


def setup_model(
    cfg: ModelConfig,
) -> tuple[AutoModelForCausalLM, AutoTokenizer, torch.device, Optional[str]]:
    """Load model, tokenizer, and eagerly load the verbalizer LoRA.

    The verbalizer adapter is loaded once here and left as the active adapter.
    Target LoRAs are loaded lazily (and cached) in the outer experiment loop.
    """
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(cfg.model_name)
    model = load_model(
        cfg.model_name,
        cfg.dtype,
        attn_implementation=cfg.attn_implementation,
    )
    model.eval()

    # Add dummy adapter so PeftModel API is available. target_modules is
    # required for architectures PEFT can't auto-map (e.g. Qwen3.6).
    dummy_config = LoraConfig(target_modules=["q_proj"])
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
) -> tuple[SteeredAutoregressiveSampler, list[int], list[dict[str, str]]]:
    """Collect activations from the already-loaded target LoRA and build a sampler.

    The target adapter must already be loaded (by the caller, once per target_word).
    This function only switches between target and verbalizer adapters — it never
    loads or deletes.

    Returns:
        (sampler, oracle_context_ids, oracle_messages). The sampler's hook is
        NOT attached yet; the caller must use ``with sampler:`` or call
        ``sampler.attach_hook()``.
    """
    # Collect activations from the prompt-only sequence under the target LoRA.
    # This matches the upstream taboo eval contract: base_experiment.py
    # defaults add_response_to_context_prompt=False, and
    # experiments/taboo_open_ended_eval.py uses that default. The oracle was
    # trained to read activations at the last |segment_start_idx| tokens of
    # the user turn (with apply_chat_template(add_generation_prompt=True)),
    # so we must feed the same shape at inference.
    formatted_prompt = [{"role": "user", "content": context_prompt}]
    inputs_BL = encode_messages(
        tokenizer=tokenizer,
        message_dicts=[formatted_prompt],
        add_generation_prompt=True,
        enable_thinking=False,
        device=device,
    )

    num_layers = get_text_config(model).num_hidden_layers
    act_layer = int(num_layers * (cfg.selected_layer_percent / 100))

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
    # transformers >=5 defaults apply_chat_template to return a BatchEncoding;
    # pass return_dict=False so we get a flat list of token ids here.
    oracle_messages = [{"role": "user", "content": oracle_user_content}]
    oracle_ids = tokenizer.apply_chat_template(
        oracle_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors=None,
        return_dict=False,
        padding=False,
        enable_thinking=False,
    )
    if not isinstance(oracle_ids, list) or (
        oracle_ids and not isinstance(oracle_ids[0], int)
    ):
        raise TypeError(
            f"Expected list of token ids from apply_chat_template, got {type(oracle_ids).__name__}"
        )

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

    return sampler, oracle_ids, oracle_messages


def config_hash(exp_cfg: ExperimentConfig) -> str:
    """Return a deterministic hash of config fields that affect iteration identity.

    Excludes output_dir, device, and dtype since they don't change which results
    are produced, only where/how they're stored/computed.
    """
    sampling = exp_cfg.sampling
    key = {
        "code_version": CODE_VERSION,
        "model_name": exp_cfg.model.model_name,
        "attn_implementation": exp_cfg.model.attn_implementation,
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
        "direct_answer_temperature": sampling.direct_answer_temperature,
        "direct_confidence_temperature": sampling.direct_confidence_temperature,
        "direct_retry_on_parse_failure": sampling.direct_retry_on_parse_failure,
        "direct_structured_fallback": sampling.direct_structured_fallback,
        "direct_linguistic_enabled": sampling.direct_linguistic_enabled,
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

    method_names = get_method_names(exp_cfg.sampling)
    raw = checkpoint["predictions"]
    missing = [m for m in method_names if m not in raw]
    if missing:
        raise ValueError(
            f"Checkpoint missing methods {missing}. Use --no-resume to start fresh."
        )

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
        for p in predictions[method_names[0]]
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
    method_names = get_method_names(sampling)

    all_predictions: dict[str, list[WordPrediction]] = {m: [] for m in method_names}

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

                sampler, oracle_ids, oracle_messages = prepare_activation_and_sampler(
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
                    # --- Method 1a/1b: Log-prob baselines ---
                    # We track both variants side-by-side:
                    #  - with char-to-token offset mapping
                    #  - offset-free prefix approximation
                    lp_result = logprob_confidence(
                        sampler=sampler,
                        context=oracle_ids,
                        answer_vocab=TABOO_WORDS,
                        max_new_tokens=sampling.max_new_tokens,
                    )

                    lp_shared_metadata = {
                        "extracted_word": lp_result.extracted_word,
                        "first_token_entropy": lp_result.first_token_entropy,
                        "first_token_max_prob": lp_result.first_token_max_prob,
                        "per_token_entropies": lp_result.per_token_entropies,
                        "mean_token_entropy": lp_result.mean_token_entropy,
                        "negative_mean_token_entropy": (
                            lp_result.negative_mean_token_entropy
                        ),
                        "mean_token_max_prob": lp_result.mean_token_max_prob,
                        "mean_log_prob": lp_result.mean_log_prob,
                        "min_log_prob": lp_result.min_log_prob,
                        "geometric_mean_prob": lp_result.geometric_mean_prob,
                        "word_prob_with_offset": lp_result.word_prob_with_offset,
                        "word_n_tokens_with_offset": lp_result.word_n_tokens_with_offset,
                        "word_prob_with_offset_fallback_used": (
                            lp_result.word_prob_with_offset_fallback_used
                        ),
                        "word_prob_with_offset_fallback_reason": (
                            lp_result.word_prob_with_offset_fallback_reason
                        ),
                        "word_prob_no_offset": lp_result.word_prob_no_offset,
                        "word_n_tokens_no_offset": lp_result.word_n_tokens_no_offset,
                        "word_prob_no_offset_fallback_used": (
                            lp_result.word_prob_no_offset_fallback_used
                        ),
                        "word_prob_no_offset_fallback_reason": (
                            lp_result.word_prob_no_offset_fallback_reason
                        ),
                        "word_prob_no_offset_truncated": (
                            lp_result.word_prob_no_offset_truncated
                        ),
                        "extracted_word_source": lp_result.extracted_word_source,
                        "answer_vocab_size": lp_result.answer_vocab_size,
                    }

                    all_predictions["logprob_offset"].append(
                        WordPrediction(
                            target_word=target_word,
                            context_prompt=ctx_prompt,
                            verbalizer_prompt=verbalizer_prompt,
                            predicted_answer=lp_result.generated_text,
                            confidence=lp_result.word_prob_with_offset,
                            is_correct=lp_result.extracted_word == target_word,
                            method="logprob_offset",
                            method_metadata={
                                **lp_shared_metadata,
                                "confidence_variant": "with_offset",
                            },
                        )
                    )

                    all_predictions["logprob_no_offset"].append(
                        WordPrediction(
                            target_word=target_word,
                            context_prompt=ctx_prompt,
                            verbalizer_prompt=verbalizer_prompt,
                            predicted_answer=lp_result.generated_text,
                            confidence=lp_result.word_prob_no_offset,
                            is_correct=lp_result.extracted_word == target_word,
                            method="logprob_no_offset",
                            method_metadata={
                                **lp_shared_metadata,
                                "confidence_variant": "no_offset",
                            },
                        )
                    )

                    # --- Method 3: Direct confidence elicitation ---
                    elicitation_result = direct_elicitation(
                        sampler=sampler,
                        context=oracle_ids,
                        oracle_messages=oracle_messages,
                        max_new_tokens=sampling.max_new_tokens,
                        answer_temperature=sampling.direct_answer_temperature,
                        confidence_temperature=(sampling.direct_confidence_temperature),
                        retry_on_parse_failure=(sampling.direct_retry_on_parse_failure),
                        structured_fallback=sampling.direct_structured_fallback,
                    )
                    elicited_word = extract_predicted_word(
                        elicitation_result.answer_text, TABOO_WORDS
                    )
                    all_predictions["direct"].append(
                        WordPrediction(
                            target_word=target_word,
                            context_prompt=ctx_prompt,
                            verbalizer_prompt=verbalizer_prompt,
                            predicted_answer=elicitation_result.answer_text,
                            confidence=elicitation_result.parsed_confidence,
                            is_correct=elicited_word == target_word,
                            method="direct",
                            method_metadata={
                                "extracted_word": elicited_word,
                                "confidence_text": elicitation_result.confidence_text,
                                "raw_confidence_value": elicitation_result.raw_confidence_value,
                                "confidence_decode_method": (
                                    elicitation_result.confidence_decode_method
                                ),
                                "freeform_confidence_text": (
                                    elicitation_result.freeform_confidence_text
                                ),
                                "answer_temperature": (
                                    elicitation_result.answer_temperature
                                ),
                                "confidence_temperature": (
                                    elicitation_result.confidence_temperature
                                ),
                                "answer_do_sample": (
                                    elicitation_result.answer_do_sample
                                ),
                                "confidence_do_sample": (
                                    elicitation_result.confidence_do_sample
                                ),
                                "confidence_parse_failed": (
                                    elicitation_result.confidence_parse_failed
                                ),
                                "confidence_retry_used": (
                                    elicitation_result.confidence_retry_used
                                ),
                                "confidence_retry_text": (
                                    elicitation_result.confidence_retry_text
                                ),
                                "confidence_structured_fallback_used": (
                                    elicitation_result.confidence_structured_fallback_used
                                ),
                                "confidence_default_fallback_used": (
                                    elicitation_result.confidence_default_fallback_used
                                ),
                                "structured_top_confidence_value": (
                                    elicitation_result.structured_top_confidence_value
                                ),
                                "structured_top_probability": (
                                    elicitation_result.structured_top_probability
                                ),
                                "structured_entropy": elicitation_result.structured_entropy,
                                "structured_top_candidates": (
                                    elicitation_result.structured_top_candidates
                                ),
                            },
                        )
                    )

                    # --- Method 3b: Verbalized-linguistic confidence ---
                    # Reuses the answer from `direct_elicitation` so the
                    # answer turn isn't repeated. Emits three readouts from
                    # the same 5-label distribution. See
                    # findings/direct_elicitation_variants_2026-05-11.md.
                    if sampling.direct_linguistic_enabled:
                        ling_result = score_linguistic_confidence(
                            sampler=sampler,
                            oracle_messages=oracle_messages,
                            answer_text=elicitation_result.answer_text,
                        )
                        ling_metadata = {
                            "extracted_word": elicited_word,
                            "labels": ling_result.labels,
                            "label_log_scores": ling_result.label_log_scores,
                            "label_probs": ling_result.label_probs,
                            "top_label": ling_result.top_label,
                            "top_label_idx": ling_result.top_label_idx,
                            "expected_value": ling_result.expected_value,
                            "p_very_high": ling_result.p_very_high,
                            "p_high_plus": ling_result.p_high_plus,
                            "prompt": ling_result.prompt,
                        }
                        for ling_method, ling_conf in (
                            (
                                "direct_linguistic_expected",
                                ling_result.expected_value,
                            ),
                            (
                                "direct_linguistic_p_very_high",
                                ling_result.p_very_high,
                            ),
                            (
                                "direct_linguistic_p_high_plus",
                                ling_result.p_high_plus,
                            ),
                        ):
                            all_predictions[ling_method].append(
                                WordPrediction(
                                    target_word=target_word,
                                    context_prompt=ctx_prompt,
                                    verbalizer_prompt=verbalizer_prompt,
                                    predicted_answer=elicitation_result.answer_text,
                                    confidence=ling_conf,
                                    is_correct=elicited_word == target_word,
                                    method=ling_method,
                                    method_metadata={
                                        **ling_metadata,
                                        "confidence_variant": ling_method.replace(
                                            "direct_linguistic_", ""
                                        ),
                                    },
                                )
                            )

                    # --- Method 2: Temperature bootstrap ---
                    for boot_temp in sampling.bootstrap_temperatures:
                        method_name = f"bootstrap_t{temperature_tag(boot_temp)}"
                        boot_result = temperature_bootstrap(
                            sampler=sampler,
                            context=oracle_ids,
                            answer_vocab=TABOO_WORDS,
                            k=sampling.bootstrap_k,
                            temperature=boot_temp,
                            max_new_tokens=sampling.max_new_tokens,
                        )
                        all_predictions[method_name].append(
                            WordPrediction(
                                target_word=target_word,
                                context_prompt=ctx_prompt,
                                verbalizer_prompt=verbalizer_prompt,
                                predicted_answer=boot_result.mode_answer,
                                confidence=boot_result.mode_frequency,
                                is_correct=boot_result.mode_answer == target_word,
                                method=method_name,
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

                    # --- Method 4: Single MCMC sample (with acceptance ratio as signal) ---
                    # --- Method 5: MCMC agreement ---
                    for mcmc_temp in sampling.mcmc_temperatures:
                        single_method = f"mcmc_t{temperature_tag(mcmc_temp)}"
                        agree_method = f"mcmc_agreement_t{temperature_tag(mcmc_temp)}"

                        mcmc_result = mcmc_oracle_sample(
                            sampler=sampler,
                            context=oracle_ids,
                            temperature=mcmc_temp,
                            mcmc_steps=sampling.mcmc_steps,
                            max_new_tokens=sampling.mcmc_max_new_tokens,
                            block_num=sampling.mcmc_block_num,
                        )
                        mcmc_word = extract_predicted_word(
                            mcmc_result.generated_text, TABOO_WORDS
                        )
                        all_predictions[single_method].append(
                            WordPrediction(
                                target_word=target_word,
                                context_prompt=ctx_prompt,
                                verbalizer_prompt=verbalizer_prompt,
                                predicted_answer=mcmc_result.generated_text,
                                confidence=mcmc_result.acceptance_ratio,
                                is_correct=mcmc_word == target_word,
                                method=single_method,
                                method_metadata={
                                    "extracted_word": mcmc_word,
                                    "acceptance_ratio": mcmc_result.acceptance_ratio,
                                    "confidence_variant": "acceptance_ratio",
                                    "alpha": mcmc_result.alpha,
                                    "temperature": mcmc_result.temperature,
                                },
                            )
                        )

                        agree_result = mcmc_agreement(
                            sampler=sampler,
                            context=oracle_ids,
                            answer_vocab=TABOO_WORDS,
                            k=sampling.power_agreement_k,
                            temperature=mcmc_temp,
                            mcmc_steps=sampling.mcmc_steps,
                            max_new_tokens=sampling.mcmc_max_new_tokens,
                            block_num=sampling.mcmc_block_num,
                        )
                        all_predictions[agree_method].append(
                            WordPrediction(
                                target_word=target_word,
                                context_prompt=ctx_prompt,
                                verbalizer_prompt=verbalizer_prompt,
                                predicted_answer=agree_result.mode_answer,
                                confidence=agree_result.mode_frequency,
                                is_correct=agree_result.mode_answer == target_word,
                                method=agree_method,
                                method_metadata={
                                    "raw_outputs": [
                                        r.generated_text for r in agree_result.samples
                                    ],
                                    "normalized_samples": agree_result.normalized_samples,
                                    "entropy": agree_result.entropy,
                                    "num_unique": agree_result.num_unique,
                                    "mean_acceptance_ratio": agree_result.mean_acceptance_ratio,
                                    "k": agree_result.k,
                                    "alpha": agree_result.alpha,
                                    "temperature": mcmc_temp,
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
    method_names = get_method_names(cfg.sampling)
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
        target = predictions[method_names[0]][i].target_word
        ctx = predictions[method_names[0]][i].context_prompt
        verb = predictions[method_names[0]][i].verbalizer_prompt
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


def summarize_predictions(predictions: list[WordPrediction]) -> dict[str, object]:
    """Compute the shared evaluation statistics for one method."""
    confidences = [p.confidence for p in predictions]
    correctness = [p.is_correct for p in predictions]
    cal = expected_calibration_error(confidences, correctness)
    nll = negative_log_likelihood(confidences, correctness)
    accuracy = sum(correctness) / max(len(correctness), 1)
    auroc_score = auroc(confidences, correctness)
    conf_correct, conf_wrong = confidence_separation(confidences, correctness)
    metadata_summary = summarize_metadata(predictions)
    return {
        "accuracy": accuracy,
        "calibration": cal,
        "nll": nll,
        "auroc": auroc_score,
        "conf_given_correct": conf_correct,
        "conf_given_wrong": conf_wrong,
        "n_samples": len(predictions),
        "metadata": metadata_summary,
    }


def summarize_metadata(predictions: list[WordPrediction]) -> dict[str, object]:
    """Aggregate diagnostic metadata such as fallback rates."""
    n = len(predictions)
    flag_suffixes = (
        "_fallback_used",
        "_parse_failed",
        "_retry_used",
        "_truncated",
    )
    flag_keys = sorted(
        {
            key
            for p in predictions
            for key, value in p.method_metadata.items()
            if isinstance(value, bool) and key.endswith(flag_suffixes)
        }
    )
    flag_counts = {}
    for key in flag_keys:
        count = sum(bool(p.method_metadata.get(key, False)) for p in predictions)
        flag_counts[key] = {
            "count": count,
            "rate": count / n if n else 0.0,
        }

    categorical_counts = {}
    for key in (
        "confidence_decode_method",
        "confidence_variant",
        "extracted_word_source",
    ):
        values = [
            str(p.method_metadata[key])
            for p in predictions
            if key in p.method_metadata and p.method_metadata[key] is not None
        ]
        if values:
            categorical_counts[key] = dict(Counter(values))

    return {
        "flag_counts": flag_counts,
        "categorical_counts": categorical_counts,
    }


def save_reliability_diagram(
    calibration: CalibrationResult,
    method_name: str,
    output_path: Path,
) -> None:
    """Render a reliability diagram PNG for one method."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.6", linewidth=1)

    points = [
        (conf, acc, count)
        for conf, acc, count in zip(
            calibration.bin_confidences,
            calibration.bin_accuracies,
            calibration.bin_counts,
        )
        if count > 0
    ]
    if points:
        xs, ys, counts = zip(*points, strict=False)
        sizes = [30 + 6 * count for count in counts]
        ax.scatter(xs, ys, s=sizes, color="#1f77b4", alpha=0.85)
        ax.plot(xs, ys, color="#1f77b4", alpha=0.5, linewidth=1)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability: {method_name}")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def sample_controlled_n_words(seed: int, n_values: list[int]) -> dict[int, list[str]]:
    """Match the fixed-seed controlled-N sampling protocol."""
    rng = random.Random(seed)
    sampled: dict[int, list[str]] = {}
    for n in n_values:
        if n <= len(TABOO_WORDS):
            sampled[n] = rng.sample(TABOO_WORDS, n)
    return sampled


def evaluate_and_save(
    all_predictions: dict[str, list[WordPrediction]],
    output_dir: str,
    seed: int,
):
    """Compute calibration metrics for each method and save results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary = {}

    for method_name, predictions in all_predictions.items():
        stats = summarize_predictions(predictions)
        cal = stats["calibration"]
        nll = stats["nll"]
        accuracy = stats["accuracy"]
        auroc_score = stats["auroc"]
        conf_correct = stats["conf_given_correct"]
        conf_wrong = stats["conf_given_wrong"]
        metadata = stats["metadata"]

        print_calibration_summary(cal, method_name)
        print(f"  Accuracy:           {accuracy:.3f}")
        print(f"  NLL:                {nll:.4f}")
        print(f"  AUROC:              {auroc_score:.3f}")
        print(f"  Conf|correct:       {conf_correct:.3f}")
        print(f"  Conf|wrong:         {conf_wrong:.3f}")
        flag_counts = metadata["flag_counts"]
        if flag_counts:
            print("  Metadata flags:")
            for key, item in flag_counts.items():
                print(f"    {key}: {item['count']} ({item['rate']:.2%})")

        summary[method_name] = {
            "accuracy": accuracy,
            "ece": cal.ece,
            "brier_score": cal.brier_score,
            "nll": nll,
            "auroc": auroc_score,
            "conf_given_correct": conf_correct,
            "conf_given_wrong": conf_wrong,
            "n_samples": len(predictions),
            "metadata": metadata,
        }

        save_reliability_diagram(
            cal,
            method_name,
            output_path / "reliability_diagrams" / f"{method_name}.png",
        )

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
            "metadata": metadata,
            "predictions": [asdict(p) for p in predictions],
        }

        with open(output_path / f"{method_name}_results.json", "w") as f:
            json.dump(method_output, f, indent=2)

    controlled_n_summary: dict[str, dict[str, float | int | list[str]]] = {}
    controlled_n_words = sample_controlled_n_words(seed, [2, 5, 10, 20])
    for n, words in controlled_n_words.items():
        word_set = set(words)
        subset_summary = {}
        subset_dir = output_path / "controlled_n" / f"N{n}"
        subset_dir.mkdir(parents=True, exist_ok=True)
        for method_name, predictions in all_predictions.items():
            subset_predictions = [p for p in predictions if p.target_word in word_set]
            stats = summarize_predictions(subset_predictions)
            cal = stats["calibration"]
            subset_summary[method_name] = {
                "accuracy": stats["accuracy"],
                "ece": cal.ece,
                "brier_score": cal.brier_score,
                "nll": stats["nll"],
                "auroc": stats["auroc"],
                "conf_given_correct": stats["conf_given_correct"],
                "conf_given_wrong": stats["conf_given_wrong"],
                "n_samples": stats["n_samples"],
                "metadata": stats["metadata"],
            }
            save_reliability_diagram(
                cal,
                f"{method_name} (N={n})",
                subset_dir / f"{method_name}_reliability.png",
            )
            with open(subset_dir / f"{method_name}_results.json", "w") as f:
                json.dump(
                    {
                        "method": method_name,
                        "controlled_n": n,
                        "target_words": words,
                        "accuracy": stats["accuracy"],
                        "calibration": {
                            "ece": cal.ece,
                            "brier_score": cal.brier_score,
                            "bin_confidences": cal.bin_confidences,
                            "bin_accuracies": cal.bin_accuracies,
                            "bin_counts": cal.bin_counts,
                        },
                        "nll": stats["nll"],
                        "auroc": stats["auroc"],
                        "conf_given_correct": stats["conf_given_correct"],
                        "conf_given_wrong": stats["conf_given_wrong"],
                        "metadata": stats["metadata"],
                        "predictions": [asdict(p) for p in subset_predictions],
                    },
                    f,
                    indent=2,
                )
        controlled_n_summary[f"N{n}"] = {
            "target_words": words,
            "methods": subset_summary,
        }

    # Save summary comparison
    with open(output_path / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(output_path / "controlled_n_summary.json", "w") as f:
        json.dump(controlled_n_summary, f, indent=2)

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
    parser.add_argument(
        "--preset",
        default="qwen3-8b",
        choices=sorted(MODEL_PRESETS),
        help="Model preset (selects base model + verbalizer LoRA + target-LoRA template)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the preset's base model id (advanced; leaves verbalizer/target from preset)",
    )
    parser.add_argument(
        "--attn-implementation",
        default="auto",
        help=(
            "Transformers attention backend. auto uses the fastest known-working "
            "backend per model family; examples: sdpa, flash_attention_2, "
            "flash_attention_3, kernels-community/flash-attn2."
        ),
    )
    parser.add_argument(
        "--max-prompts", type=int, default=None, help="Limit context prompts"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: results/<preset>/taboo_uq)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-k", type=int, default=20)
    parser.add_argument(
        "--bootstrap-temps",
        type=float,
        nargs="+",
        default=None,
        help="Override the bootstrap temperature sweep",
    )
    parser.add_argument(
        "--direct-answer-temp",
        type=float,
        default=0.0,
        help="Temperature for direct-elicitation answer generation (0 = greedy)",
    )
    parser.add_argument(
        "--direct-confidence-temp",
        type=float,
        default=0.0,
        help="Temperature for direct-elicitation confidence generation (0 = greedy)",
    )
    parser.add_argument(
        "--no-direct-retry",
        action="store_true",
        help="Disable the stricter confidence retry after an unparseable answer",
    )
    parser.add_argument(
        "--no-direct-structured-fallback",
        action="store_true",
        help="Disable structured 0..100 scoring when confidence parsing fails",
    )
    parser.add_argument("--mcmc-steps", type=int, default=5)
    parser.add_argument(
        "--mcmc-temp",
        type=float,
        default=None,
        help="Deprecated single-temperature override for MCMC methods",
    )
    parser.add_argument(
        "--mcmc-temps",
        type=float,
        nargs="+",
        default=None,
        help="Override the MCMC temperature sweep",
    )
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

    sampling_cfg = SamplingConfig(
        bootstrap_k=args.bootstrap_k,
        direct_answer_temperature=args.direct_answer_temp,
        direct_confidence_temperature=args.direct_confidence_temp,
        direct_retry_on_parse_failure=not args.no_direct_retry,
        direct_structured_fallback=not args.no_direct_structured_fallback,
        mcmc_steps=args.mcmc_steps,
        power_agreement_k=args.agreement_k,
    )
    if args.bootstrap_temps is not None:
        sampling_cfg.bootstrap_temperatures = args.bootstrap_temps
    if args.mcmc_temps is not None:
        sampling_cfg.mcmc_temperatures = args.mcmc_temps
    elif args.mcmc_temp is not None:
        sampling_cfg.mcmc_temperatures = [args.mcmc_temp]

    model_cfg = ModelConfig.from_preset(args.preset)
    if args.model is not None:
        model_cfg.model_name = args.model
    model_cfg.attn_implementation = args.attn_implementation

    output_dir = args.output_dir or f"results/{args.preset}/taboo_uq"

    cfg = ExperimentConfig(
        model=model_cfg,
        sampling=sampling_cfg,
        output_dir=output_dir,
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
    evaluate_and_save(predictions, cfg.output_dir, seed=cfg.seed)
