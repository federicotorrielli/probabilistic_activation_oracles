"""Mini-study: direct-elicitation prompt variants on Qwen3-8B + steered activations.

Question: the zero-shot "rate 0-100" prompt saturates at ~100 regardless of
correctness (ECE 58%, AUROC 0.516 in results/qwen3-8b/run_3). Can a different
prompting strategy break that saturation?

Five variants are scored against the SAME generated answer so the comparison is
purely about the elicitation prompt, not the steering or the answer turn.

Variants:
    V0 zero_shot              current control: "On a scale of 0 to 100..."
    V1 few_shot_numeric       4 in-context demos at conf 95/70/40/15
    V2 verbalized_linguistic  5 word labels scored via constrained logits
    V3 p_true                 "Is this answer correct? Yes/No"  -> P(Yes)
    V4 hedged                 same numeric prompt with overconfidence warning

Slice: 6 words x 1 verbalizer x 5 context prompts = 30 samples per variant.
Cheap enough to run on a B200 in a few minutes.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from pao.answer_extraction import extract_predicted_word
from pao.config import (
    AO_ROOT,
    ExperimentConfig,
    ModelConfig,
    SamplingConfig,
    VERBALIZER_PROMPTS_TABOO,
)
from pao.experiments.run_taboo_uq import (
    prepare_activation_and_sampler,
    setup_experiment_state,
)
from pao.hf_utils import load_lora_adapter
from pao.methods.direct_elicitation import (
    _apply_chat_template,
    _generate_text,
    _score_continuations,
    parse_confidence_number,
)
from pao.oracle_sampler import SteeredAutoregressiveSampler

WORDS = ["moon", "snow", "smile", "wave", "song", "clock"]
VERBALIZER_PROMPT = VERBALIZER_PROMPTS_TABOO[0]
NUM_CONTEXTS = 5

LINGUISTIC_LABELS = ["very low", "low", "medium", "high", "very high"]
LINGUISTIC_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9]

FEW_SHOT_DEMOS = [
    {
        "question": "What is the capital of France?",
        "answer": "Paris",
        "confidence": 95,
    },
    {
        "question": "What is 17 times 24?",
        "answer": "408",
        "confidence": 70,
    },
    {
        "question": "In what year was the painter Caravaggio born?",
        "answer": "1571",
        "confidence": 40,
    },
    {
        "question": "What was the population of Reykjavik in 1850?",
        "answer": "1200",
        "confidence": 15,
    },
]


@dataclass
class VariantPrediction:
    word: str
    context_prompt: str
    answer_text: str
    extracted_word: str
    is_correct: bool
    variant: str
    confidence: float  # in [0, 1]
    raw_text: str
    extra: dict


def _format_few_shot_block() -> str:
    lines = []
    for demo in FEW_SHOT_DEMOS:
        lines.append(
            f"Q: {demo['question']}\n"
            f"A: {demo['answer']}\n"
            f"Confidence (0-100): {demo['confidence']}\n"
        )
    return "\n".join(lines)


def _confidence_from_freeform(
    sampler: SteeredAutoregressiveSampler,
    oracle_messages: list[dict[str, str]],
    answer_text: str,
    user_prompt: str,
    max_new_tokens: int = 10,
) -> tuple[float, str]:
    """Run a turn-2 chat with `user_prompt` and parse a 0-100 number.

    Returns (confidence_in_[0,1], raw_generated_text).
    Falls back to 1.0 if parsing fails (matches the saturation default we
    are trying to escape, so failures don't artificially help).
    """
    turn2_messages = oracle_messages + [
        {"role": "assistant", "content": answer_text},
        {"role": "user", "content": user_prompt},
    ]
    turn2_context = _apply_chat_template(sampler.tokenizer, turn2_messages)
    text, _ = _generate_text(
        sampler=sampler,
        context=turn2_context,
        temperature=0.0,
        max_new_tokens=max_new_tokens,
    )
    raw_value = parse_confidence_number(text)
    if raw_value is None:
        return 1.0, text
    return max(0.0, min(100.0, raw_value)) / 100.0, text


def _zero_shot_confidence(sampler, oracle_messages, answer_text):
    return _confidence_from_freeform(
        sampler,
        oracle_messages,
        answer_text,
        user_prompt=(
            "On a scale of 0 to 100, how confident are you in your answer? "
            "Reply with just the number."
        ),
    )


def _hedged_confidence(sampler, oracle_messages, answer_text):
    return _confidence_from_freeform(
        sampler,
        oracle_messages,
        answer_text,
        user_prompt=(
            "Be honest. You may be wrong, and language models often overestimate "
            "their own confidence. On a scale of 0 to 100, how confident are you "
            "in your answer? Reply with just the number."
        ),
    )


def _few_shot_confidence(sampler, oracle_messages, answer_text):
    demo_block = _format_few_shot_block()
    user_prompt = (
        "I will ask how confident you are in your answer above. First, here are "
        "examples of well-calibrated confidence ratings on different questions:\n\n"
        f"{demo_block}\n"
        "Now rate your confidence in YOUR answer above on the same 0-100 scale. "
        "Reply with just the number."
    )
    return _confidence_from_freeform(
        sampler, oracle_messages, answer_text, user_prompt, max_new_tokens=10
    )


def _verbalized_linguistic_confidence(
    sampler: SteeredAutoregressiveSampler,
    oracle_messages: list[dict[str, str]],
    answer_text: str,
) -> tuple[float, str, dict]:
    """Score the 5 linguistic labels via constrained logits."""
    user_prompt = (
        "How confident are you in your answer? Reply with exactly one of these "
        "options and nothing else: very low, low, medium, high, very high."
    )
    turn2_messages = oracle_messages + [
        {"role": "assistant", "content": answer_text},
        {"role": "user", "content": user_prompt},
    ]
    turn2_context = _apply_chat_template(sampler.tokenizer, turn2_messages)

    # Encode each label as the bytes that would appear at the start of an
    # assistant reply. Use add_special_tokens=False so we don't insert a BOS.
    label_ids = [
        sampler.tokenizer.encode(label, add_special_tokens=False)
        for label in LINGUISTIC_LABELS
    ]
    # Some tokenizers (Qwen) need a leading space for the first token to be
    # consistent. If any label tokenizes to zero tokens (shouldn't happen)
    # fall back to the with-space variant.
    label_ids = [
        sampler.tokenizer.encode(" " + LINGUISTIC_LABELS[i], add_special_tokens=False)
        if len(ids) == 0
        else ids
        for i, ids in enumerate(label_ids)
    ]
    scores = _score_continuations(
        sampler=sampler,
        context=turn2_context,
        continuations=label_ids,
        batch_size=8,
    )
    log_t = torch.tensor(scores, dtype=torch.float32)
    probs = torch.softmax(log_t, dim=0)
    expected = float(sum(p.item() * v for p, v in zip(probs, LINGUISTIC_VALUES)))
    top_idx = int(torch.argmax(probs).item())
    top_label = LINGUISTIC_LABELS[top_idx]
    top_value = LINGUISTIC_VALUES[top_idx]
    extra = {
        "labels": LINGUISTIC_LABELS,
        "label_log_scores": scores,
        "label_probs": [float(p.item()) for p in probs],
        "top_label": top_label,
        "top_value": top_value,
        "expected_value": expected,
    }
    return expected, top_label, extra


def _p_true_confidence(
    sampler: SteeredAutoregressiveSampler,
    oracle_messages: list[dict[str, str]],
    answer_text: str,
    verbalizer_prompt: str,
) -> tuple[float, str, dict]:
    """Score Yes vs No for "Is this answer correct?"."""
    user_prompt = (
        f"Question: {verbalizer_prompt}\n"
        f"Proposed answer: {answer_text.strip()}\n\n"
        "Is the proposed answer correct? Reply with just Yes or No."
    )
    # We re-pose this as a fresh single-turn chat. The activation hook stays
    # attached so the oracle still "sees" the steered context.
    turn2_messages = [{"role": "user", "content": user_prompt}]
    turn2_context = _apply_chat_template(sampler.tokenizer, turn2_messages)

    yes_variants = ["Yes", " Yes", "yes", " yes"]
    no_variants = ["No", " No", "no", " no"]

    def _encode(text):
        return sampler.tokenizer.encode(text, add_special_tokens=False)

    yes_ids = [_encode(v) for v in yes_variants if len(_encode(v)) > 0]
    no_ids = [_encode(v) for v in no_variants if len(_encode(v)) > 0]
    all_ids = yes_ids + no_ids
    scores = _score_continuations(
        sampler=sampler,
        context=turn2_context,
        continuations=all_ids,
        batch_size=8,
    )
    yes_scores = scores[: len(yes_ids)]
    no_scores = scores[len(yes_ids) :]
    # Aggregate via log-sum-exp so multiple casings of "yes" combine sanely.
    yes_lse = torch.logsumexp(torch.tensor(yes_scores), dim=0).item()
    no_lse = torch.logsumexp(torch.tensor(no_scores), dim=0).item()
    p_yes = math.exp(yes_lse) / (math.exp(yes_lse) + math.exp(no_lse))
    extra = {
        "yes_variants": yes_variants,
        "no_variants": no_variants,
        "yes_log_scores": yes_scores,
        "no_log_scores": no_scores,
        "yes_logsumexp": yes_lse,
        "no_logsumexp": no_lse,
    }
    raw = f"P(Yes)={p_yes:.3f}"
    return p_yes, raw, extra


def _run_one_pair(
    sampler: SteeredAutoregressiveSampler,
    oracle_ids: list[int],
    oracle_messages: list[dict[str, str]],
    target_word: str,
    context_prompt: str,
    verbalizer_prompt: str,
    max_new_tokens: int,
) -> list[VariantPrediction]:
    """Generate the answer once, then score every variant against it."""
    answer_text, _ = _generate_text(
        sampler=sampler,
        context=oracle_ids,
        temperature=0.0,
        max_new_tokens=max_new_tokens,
    )
    extracted = extract_predicted_word(answer_text, WORDS)
    is_correct = extracted == target_word

    out: list[VariantPrediction] = []

    conf, raw = _zero_shot_confidence(sampler, oracle_messages, answer_text)
    out.append(
        VariantPrediction(
            word=target_word,
            context_prompt=context_prompt,
            answer_text=answer_text,
            extracted_word=extracted,
            is_correct=is_correct,
            variant="zero_shot",
            confidence=conf,
            raw_text=raw,
            extra={},
        )
    )

    conf, raw = _few_shot_confidence(sampler, oracle_messages, answer_text)
    out.append(
        VariantPrediction(
            word=target_word,
            context_prompt=context_prompt,
            answer_text=answer_text,
            extracted_word=extracted,
            is_correct=is_correct,
            variant="few_shot_numeric",
            confidence=conf,
            raw_text=raw,
            extra={},
        )
    )

    conf, raw_label, extra = _verbalized_linguistic_confidence(
        sampler, oracle_messages, answer_text
    )
    out.append(
        VariantPrediction(
            word=target_word,
            context_prompt=context_prompt,
            answer_text=answer_text,
            extracted_word=extracted,
            is_correct=is_correct,
            variant="verbalized_linguistic",
            confidence=conf,
            raw_text=raw_label,
            extra=extra,
        )
    )

    conf, raw, extra = _p_true_confidence(
        sampler, oracle_messages, answer_text, verbalizer_prompt
    )
    out.append(
        VariantPrediction(
            word=target_word,
            context_prompt=context_prompt,
            answer_text=answer_text,
            extracted_word=extracted,
            is_correct=is_correct,
            variant="p_true",
            confidence=conf,
            raw_text=raw,
            extra=extra,
        )
    )

    conf, raw = _hedged_confidence(sampler, oracle_messages, answer_text)
    out.append(
        VariantPrediction(
            word=target_word,
            context_prompt=context_prompt,
            answer_text=answer_text,
            extracted_word=extracted,
            is_correct=is_correct,
            variant="hedged",
            confidence=conf,
            raw_text=raw,
            extra={},
        )
    )

    return out


def _ece(confs: list[float], correct: list[bool], n_bins: int = 10) -> float:
    if not confs:
        return float("nan")
    bins = [[] for _ in range(n_bins)]
    for c, y in zip(confs, correct):
        idx = min(int(c * n_bins), n_bins - 1)
        bins[idx].append((c, y))
    total = len(confs)
    ece = 0.0
    for b in bins:
        if not b:
            continue
        avg_c = sum(c for c, _ in b) / len(b)
        avg_acc = sum(1.0 for _, y in b if y) / len(b)
        ece += abs(avg_c - avg_acc) * len(b) / total
    return ece


def _brier(confs: list[float], correct: list[bool]) -> float:
    if not confs:
        return float("nan")
    return sum((c - (1.0 if y else 0.0)) ** 2 for c, y in zip(confs, correct)) / len(confs)


def _auroc(confs: list[float], correct: list[bool]) -> float:
    pos = [c for c, y in zip(confs, correct) if y]
    neg = [c for c, y in zip(confs, correct) if not y]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def _summarize(predictions: list[VariantPrediction]) -> dict[str, dict]:
    by_variant: dict[str, list[VariantPrediction]] = {}
    for p in predictions:
        by_variant.setdefault(p.variant, []).append(p)
    summary = {}
    for variant, preds in by_variant.items():
        confs = [p.confidence for p in preds]
        correct = [p.is_correct for p in preds]
        mean_correct = (
            sum(c for c, y in zip(confs, correct) if y) / max(1, sum(correct))
            if any(correct)
            else float("nan")
        )
        mean_wrong = (
            sum(c for c, y in zip(confs, correct) if not y)
            / max(1, sum(1 for y in correct if not y))
            if any(not y for y in correct)
            else float("nan")
        )
        summary[variant] = {
            "n": len(preds),
            "accuracy": sum(correct) / len(preds),
            "mean_conf_correct": mean_correct,
            "mean_conf_wrong": mean_wrong,
            "conf_gap": (mean_correct - mean_wrong)
            if (not math.isnan(mean_correct) and not math.isnan(mean_wrong))
            else float("nan"),
            "ece_10": _ece(confs, correct, 10),
            "brier": _brier(confs, correct),
            "auroc": _auroc(confs, correct),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="qwen3-8b")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to results/<preset>/mini_direct_variants_YYYY-MM-DD",
    )
    parser.add_argument("--num-contexts", type=int, default=NUM_CONTEXTS)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    args = parser.parse_args()

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if args.output_dir is None:
        args.output_dir = f"results/{args.preset}/mini_direct_variants_{today}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[mini] preset={args.preset} words={WORDS} contexts={args.num_contexts}")
    print(f"[mini] output_dir={out_dir}")

    model_cfg = ModelConfig.from_preset(args.preset)
    sampling_cfg = SamplingConfig(max_new_tokens=args.max_new_tokens)
    exp_cfg = ExperimentConfig(
        model=model_cfg,
        sampling=sampling_cfg,
        max_context_prompts=args.num_contexts,
        output_dir=str(out_dir),
    )

    model, tokenizer, device, verbalizer_adapter, context_prompts = (
        setup_experiment_state(exp_cfg)
    )

    all_predictions: list[VariantPrediction] = []
    total = len(WORDS) * len(context_prompts)
    done = 0
    for target_word in WORDS:
        target_lora_path = exp_cfg.model.target_lora_template.format(word=target_word)
        target_adapter = load_lora_adapter(model, target_lora_path)
        for ctx in context_prompts:
            sampler, oracle_ids, oracle_messages = prepare_activation_and_sampler(
                model=model,
                tokenizer=tokenizer,
                device=device,
                cfg=exp_cfg.model,
                target_adapter=target_adapter,
                verbalizer_adapter=verbalizer_adapter,
                context_prompt=ctx,
                verbalizer_prompt=VERBALIZER_PROMPT,
            )
            with sampler:
                preds = _run_one_pair(
                    sampler=sampler,
                    oracle_ids=oracle_ids,
                    oracle_messages=oracle_messages,
                    target_word=target_word,
                    context_prompt=ctx,
                    verbalizer_prompt=VERBALIZER_PROMPT,
                    max_new_tokens=args.max_new_tokens,
                )
            all_predictions.extend(preds)
            done += 1
            print(
                f"[mini] {done}/{total} word={target_word} "
                f"answer={preds[0].extracted_word!r} correct={preds[0].is_correct} "
                f"V0={preds[0].confidence:.2f} V1={preds[1].confidence:.2f} "
                f"V2={preds[2].confidence:.2f} V3={preds[3].confidence:.2f} "
                f"V4={preds[4].confidence:.2f}"
            )

    # Write per-variant JSONL.
    by_variant: dict[str, list[VariantPrediction]] = {}
    for p in all_predictions:
        by_variant.setdefault(p.variant, []).append(p)
    for variant, preds in by_variant.items():
        path = out_dir / f"{variant}.jsonl"
        with open(path, "w") as f:
            for p in preds:
                f.write(json.dumps(asdict(p)) + "\n")

    summary = _summarize(all_predictions)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(
            {
                "metadata": {
                    "preset": args.preset,
                    "model": exp_cfg.model.model_name,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "words": WORDS,
                    "num_contexts": len(context_prompts),
                    "verbalizer_prompt": VERBALIZER_PROMPT,
                    "context_prompts": context_prompts,
                },
                "summary": summary,
            },
            f,
            indent=2,
        )

    print("\n[mini] === SUMMARY ===")
    header = (
        f"{'variant':<24} {'n':>3} {'acc':>5} {'c_correct':>10} {'c_wrong':>8} "
        f"{'gap':>6} {'ECE':>6} {'Brier':>6} {'AUROC':>6}"
    )
    print(header)
    for variant, row in summary.items():
        print(
            f"{variant:<24} {row['n']:>3} {row['accuracy']:>5.2f} "
            f"{row['mean_conf_correct']:>10.3f} {row['mean_conf_wrong']:>8.3f} "
            f"{row['conf_gap']:>6.3f} {row['ece_10']:>6.3f} "
            f"{row['brier']:>6.3f} {row['auroc']:>6.3f}"
        )


if __name__ == "__main__":
    main()
