"""Backfill the two ARR-rebuttal methods without a full 16-method re-run.

Two tasks, one shared grid loop (20 words x 3 verbalizers x 100 contexts):

  --task forced_choice   M7: closed-vocabulary forced choice over the 20 taboo
                         words, scored under the steering hook (reviewer 3's
                         requested baseline). Emits ``forced_choice``.

  --task extra_temps     Extends the bootstrap temperature grid past 1.5 on the
                         low-accuracy oracles, to locate the optimum reviewer 3
                         noted we had not reached. Emits ``bootstrap_t1p75`` and
                         ``bootstrap_t2p0`` (override with --temps).

Mirrors backfill_direct_linguistic.py: reuses setup_experiment_state,
prepare_activation_and_sampler, and evaluate_and_save, writes to a sidecar dir,
and is resumable via its own checkpoint.

    uv run python -m pao.experiments.backfill_extra_methods \
        --preset qwen3-8b --task forced_choice
    uv run python -m pao.experiments.backfill_extra_methods \
        --preset qwen3.6-27b --task extra_temps --temps 1.75 2.0
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from pao.calibration.secret_word_calibration import WordPrediction
from pao.config import (
    AO_ROOT,
    MODEL_PRESETS,
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
    temperature_tag,
)
from pao.hf_utils import load_lora_adapter
from pao.methods.forced_choice import forced_choice_confidence
from pao.methods.temperature_bootstrap import temperature_bootstrap


def _method_names(task: str, temps: list[float]) -> list[str]:
    if task == "forced_choice":
        return ["forced_choice"]
    return [f"bootstrap_t{temperature_tag(t)}" for t in temps]


def _load_checkpoint(
    path: Path, methods: list[str]
) -> tuple[dict[str, list[WordPrediction]], set[tuple[str, str, str]]]:
    preds: dict[str, list[WordPrediction]] = {m: [] for m in methods}
    done: set[tuple[str, str, str]] = set()
    if not path.exists():
        return preds, done
    raw = json.loads(path.read_text())
    for method, plist in raw["predictions"].items():
        preds[method] = [WordPrediction(**p) for p in plist]
    done = {tuple(k) for k in raw["completed_keys"]}
    return preds, done


def _save_checkpoint(
    path: Path,
    preds: dict[str, list[WordPrediction]],
    done: set[tuple[str, str, str]],
) -> None:
    payload = {
        "predictions": {m: [asdict(p) for p in pl] for m, pl in preds.items()},
        "completed_keys": sorted(done),
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(path)


def _score_sample(
    task: str,
    temps: list[float],
    sampler,
    oracle_ids: list[int],
    target_word: str,
    sampling: SamplingConfig,
) -> list[tuple[str, str, float, bool, dict]]:
    """Return (method, predicted_answer, confidence, is_correct, metadata) rows."""
    rows: list[tuple[str, str, float, bool, dict]] = []
    with sampler:
        if task == "forced_choice":
            res = forced_choice_confidence(sampler, oracle_ids, TABOO_WORDS)
            rows.append(
                (
                    "forced_choice",
                    res.predicted_word,
                    res.confidence,
                    res.predicted_word == target_word,
                    {
                        "word_probs": res.word_probs,
                        "word_log_probs": res.word_log_probs,
                        "word_log_probs_norm": res.word_log_probs_norm,
                        "predicted_word_norm": res.predicted_word_norm,
                        "n_tokens_per_word": res.n_tokens_per_word,
                    },
                )
            )
        else:
            for t in temps:
                res = temperature_bootstrap(
                    sampler=sampler,
                    context=oracle_ids,
                    answer_vocab=TABOO_WORDS,
                    k=sampling.bootstrap_k,
                    temperature=t,
                    max_new_tokens=sampling.max_new_tokens,
                )
                rows.append(
                    (
                        f"bootstrap_t{temperature_tag(t)}",
                        res.mode_answer,
                        res.mode_frequency,
                        res.mode_answer == target_word,
                        {
                            "raw_samples": res.samples,
                            "normalized_samples": res.normalized_samples,
                            "entropy": res.entropy,
                            "num_unique": res.num_unique,
                            "temperature": res.temperature,
                            "k": res.k,
                        },
                    )
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="qwen3-8b", choices=sorted(MODEL_PRESETS))
    parser.add_argument(
        "--task", required=True, choices=["forced_choice", "extra_temps"]
    )
    parser.add_argument(
        "--temps",
        type=float,
        nargs="+",
        default=[1.75, 2.0],
        help="extra_temps only: bootstrap temperatures to add",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--attn-implementation",
        default="auto",
        help="qwen3.6-27b must stay on flash_attention_2",
    )
    args = parser.parse_args()

    os.chdir(AO_ROOT)  # dataset paths resolve relative to AO_ROOT, as run_taboo_uq.

    methods = _method_names(args.task, args.temps)

    model_cfg = ModelConfig.from_preset(args.preset)
    model_cfg.attn_implementation = args.attn_implementation
    cfg = ExperimentConfig(
        model=model_cfg,
        sampling=SamplingConfig(),
        output_dir=args.output_dir or f"results/{args.preset}/{args.task}_backfill",
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
        preds, done = {m: [] for m in methods}, set()
    else:
        preds, done = _load_checkpoint(ckpt, methods)

    total = len(TABOO_WORDS) * len(VERBALIZER_PROMPTS_TABOO) * len(context_prompts)
    pbar = tqdm(total=total, initial=len(done), desc=args.task)
    since_save = 0

    for target_word in TABOO_WORDS:
        target_lora_path = cfg.model.target_lora_template.format(word=target_word)
        target_adapter = load_lora_adapter(model, target_lora_path)

        for verbalizer_prompt in VERBALIZER_PROMPTS_TABOO:
            for ctx_prompt in context_prompts:
                key = (target_word, verbalizer_prompt, ctx_prompt)
                if key in done:
                    continue

                sampler, oracle_ids, _ = prepare_activation_and_sampler(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    cfg=cfg.model,
                    target_adapter=target_adapter,
                    verbalizer_adapter=verbalizer_adapter,
                    context_prompt=ctx_prompt,
                    verbalizer_prompt=verbalizer_prompt,
                )

                for method, answer, conf, correct, meta in _score_sample(
                    args.task, args.temps, sampler, oracle_ids, target_word, sampling
                ):
                    preds[method].append(
                        WordPrediction(
                            target_word=target_word,
                            context_prompt=ctx_prompt,
                            verbalizer_prompt=verbalizer_prompt,
                            predicted_answer=answer,
                            confidence=conf,
                            is_correct=correct,
                            method=method,
                            method_metadata=meta,
                        )
                    )

                done.add(key)
                pbar.update(1)
                since_save += 1
                if since_save >= args.checkpoint_every:
                    _save_checkpoint(ckpt, preds, done)
                    since_save = 0

    pbar.close()
    _save_checkpoint(ckpt, preds, done)
    evaluate_and_save(preds, cfg.output_dir, cfg.seed)
    print(f"\nDone. {args.task} backfill written to {cfg.output_dir}")


if __name__ == "__main__":
    main()
