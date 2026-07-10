"""Backfill ONLY the direct-linguistic readouts (Method 3b) for a preset.

The main taboo experiment runs six method families in one monolithic loop, so
there is no flag to compute just the verbalized-linguistic confidence. Models
that were run before Method 3b existed (e.g. qwen3-8b, qwen3.6-27b) therefore
lack the three ``direct_linguistic_*_results.json`` files that gemma-3-27b has.

This script replays only the cheap part of that slice:
    1. ``direct_elicitation`` to regenerate the same greedy answer (so
       ``predicted_answer`` / ``is_correct`` match a full run byte-for-byte), then
    2. ``score_linguistic_confidence`` to score the five labels.

It emits the three production readouts (``expected``, ``p_very_high``,
``p_high_plus``) via the same ``evaluate_and_save`` writer, so the output dir
mirrors a real run's ``direct_linguistic_*_results.json`` + ``controlled_n/`` +
reliability diagrams. Writes to a sidecar dir by default so nothing in the
model's existing ``taboo_uq/`` is touched. Resumable via its own checkpoint.

    uv run python -m pao.experiments.backfill_direct_linguistic --preset qwen3.6-27b
    uv run python -m pao.experiments.backfill_direct_linguistic --preset qwen3-8b
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from pao.answer_extraction import extract_predicted_word
from pao.calibration.secret_word_calibration import WordPrediction
from pao.config import (
    AO_ROOT,
    ExperimentConfig,
    ModelConfig,
    SamplingConfig,
    TABOO_WORDS,
    VERBALIZER_PROMPTS_TABOO,
)
from pao.experiments.run_taboo_uq import (
    evaluate_and_save,
    prepare_activation_and_sampler,
    setup_experiment_state,
)
from pao.hf_utils import load_lora_adapter
from pao.methods.direct_elicitation import (
    direct_elicitation,
    score_linguistic_confidence,
)

LING_METHODS = [
    "direct_linguistic_expected",
    "direct_linguistic_p_very_high",
    "direct_linguistic_p_high_plus",
]


def _load_checkpoint(
    path: Path,
) -> tuple[dict[str, list[WordPrediction]], set[tuple[str, str, str]]]:
    all_predictions: dict[str, list[WordPrediction]] = {m: [] for m in LING_METHODS}
    completed_keys: set[tuple[str, str, str]] = set()
    if not path.exists():
        return all_predictions, completed_keys
    raw = json.loads(path.read_text())
    for method, preds in raw["predictions"].items():
        all_predictions[method] = [WordPrediction(**p) for p in preds]
    completed_keys = {tuple(k) for k in raw["completed_keys"]}
    return all_predictions, completed_keys


def _save_checkpoint(
    path: Path,
    all_predictions: dict[str, list[WordPrediction]],
    completed_keys: set[tuple[str, str, str]],
) -> None:
    payload = {
        "predictions": {
            m: [asdict(p) for p in preds] for m, preds in all_predictions.items()
        },
        "completed_keys": sorted(completed_keys),
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill only the direct-linguistic readouts for a preset"
    )
    parser.add_argument("--preset", default="qwen3-8b", choices=sorted(_presets()))
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default: results/<preset>/direct_linguistic_backfill",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Limit context prompts (full runs use all 100 -> 6000 slices)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--attn-implementation",
        default="auto",
        help="Transformers attention backend (qwen3.6-27b must stay on fa2)",
    )
    args = parser.parse_args()

    os.chdir(AO_ROOT)  # So dataset paths resolve, matching run_taboo_uq.

    model_cfg = ModelConfig.from_preset(args.preset)
    model_cfg.attn_implementation = args.attn_implementation
    cfg = ExperimentConfig(
        model=model_cfg,
        sampling=SamplingConfig(),
        output_dir=args.output_dir
        or f"results/{args.preset}/direct_linguistic_backfill",
        max_context_prompts=args.max_prompts,
        seed=args.seed,
    )
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sampling = cfg.sampling

    model, tokenizer, device, verbalizer_adapter, context_prompts = (
        setup_experiment_state(cfg)
    )

    ckpt = out_dir / "checkpoint.json"
    if args.no_resume:
        all_predictions = {m: [] for m in LING_METHODS}
        completed_keys: set[tuple[str, str, str]] = set()
    else:
        all_predictions, completed_keys = _load_checkpoint(ckpt)

    total = len(TABOO_WORDS) * len(VERBALIZER_PROMPTS_TABOO) * len(context_prompts)
    pbar = tqdm(total=total, initial=len(completed_keys), desc="direct-linguistic")
    since_save = 0

    for target_word in TABOO_WORDS:
        target_lora_path = cfg.model.target_lora_template.format(word=target_word)
        target_adapter = load_lora_adapter(model, target_lora_path)

        for verbalizer_prompt in VERBALIZER_PROMPTS_TABOO:
            for ctx_prompt in context_prompts:
                key = (target_word, verbalizer_prompt, ctx_prompt)
                if key in completed_keys:
                    continue

                sampler, oracle_ids, oracle_messages = prepare_activation_and_sampler(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    cfg=cfg.model,
                    target_adapter=target_adapter,
                    verbalizer_adapter=verbalizer_adapter,
                    context_prompt=ctx_prompt,
                    verbalizer_prompt=verbalizer_prompt,
                )

                with sampler:
                    # Regenerate the answer exactly as Method 3 would, so the
                    # answer turn the linguistic scoring sees is identical.
                    elicitation_result = direct_elicitation(
                        sampler=sampler,
                        context=oracle_ids,
                        oracle_messages=oracle_messages,
                        max_new_tokens=sampling.max_new_tokens,
                        answer_temperature=sampling.direct_answer_temperature,
                        confidence_temperature=sampling.direct_confidence_temperature,
                        retry_on_parse_failure=sampling.direct_retry_on_parse_failure,
                        structured_fallback=sampling.direct_structured_fallback,
                    )
                    elicited_word = extract_predicted_word(
                        elicitation_result.answer_text, TABOO_WORDS
                    )
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
                    ("direct_linguistic_expected", ling_result.expected_value),
                    ("direct_linguistic_p_very_high", ling_result.p_very_high),
                    ("direct_linguistic_p_high_plus", ling_result.p_high_plus),
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

                completed_keys.add(key)
                pbar.update(1)
                since_save += 1
                if since_save >= args.checkpoint_every:
                    _save_checkpoint(ckpt, all_predictions, completed_keys)
                    since_save = 0

    pbar.close()
    _save_checkpoint(ckpt, all_predictions, completed_keys)

    # Same writer the full experiment uses: emits the three
    # direct_linguistic_*_results.json, controlled_n/, reliability diagrams,
    # and a comparison_summary.json scoped to just these three methods.
    evaluate_and_save(all_predictions, cfg.output_dir, cfg.seed)
    print(f"\nDone. Backfill written to {cfg.output_dir}")


def _presets() -> list[str]:
    from pao.config import MODEL_PRESETS

    return list(MODEL_PRESETS)


if __name__ == "__main__":
    main()
