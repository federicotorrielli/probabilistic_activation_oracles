"""Direct confidence elicitation: ask the oracle how confident it is.

Two-turn approach:
1. Ask the oracle "What is the secret word?"
2. Ask "On a scale of 0-100, how confident are you?"

Both turns use the same steering hook. Expected to be poorly calibrated
(LLMs are notoriously overconfident) but useful as a comparison point.
"""

import re
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from pao.hf_utils import get_hf_activation_steering_hook
from pao.oracle_sampler import SteeredAutoregressiveSampler

# Verbalized-linguistic readout. Scoring these five labels under the steered
# hook ranks correct vs. wrong answers far more cleanly than free-form numeric
# elicitation (which saturates at ~100% regardless of correctness). See
# findings/direct_elicitation_variants_2026-05-11.md for the mini-study.
LINGUISTIC_LABELS: list[str] = ["very low", "low", "medium", "high", "very high"]
LINGUISTIC_VALUES: list[float] = [0.1, 0.3, 0.5, 0.7, 0.9]
LINGUISTIC_PROMPT: str = (
    "How confident are you in your answer? Reply with exactly one of these "
    "options and nothing else: very low, low, medium, high, very high."
)


@dataclass
class LinguisticConfidenceResult:
    """Verbalized linguistic confidence via constrained log-prob scoring.

    The five labels are scored as one-token continuations after a turn-2
    user prompt. Multiple readouts are returned so downstream analysis
    can pick the most useful one without re-running the scoring pass:

    - ``expected_value``: weighted by ``LINGUISTIC_VALUES``. Preserves
      rank but compresses everything to a narrow band because the top
      label is almost always "very high".
    - ``p_very_high``: P(top label). The strongest discriminator on
      qwen3-8b in the mini-study (AUROC 0.957, gap 0.077).
    - ``p_high_plus``: P(high) + P(very_high). Mirrors how a human would
      collapse the upper half of the scale.
    """

    labels: list[str]
    label_log_scores: list[float]
    label_probs: list[float]
    top_label: str
    top_label_idx: int
    expected_value: float  # in [0, 1]
    p_very_high: float  # in [0, 1]
    p_high_plus: float  # in [0, 1]
    prompt: str


@dataclass
class ElicitationResult:
    """Result from direct confidence elicitation."""

    answer_text: str
    confidence_text: str
    parsed_confidence: float  # 0-1 scale
    raw_confidence_value: Optional[float]  # raw number from model (0-100)
    confidence_decode_method: str
    freeform_confidence_text: str
    answer_temperature: float
    confidence_temperature: float
    answer_do_sample: bool
    confidence_do_sample: bool
    confidence_parse_failed: bool
    confidence_retry_used: bool = False
    confidence_retry_text: str | None = None
    confidence_structured_fallback_used: bool = False
    confidence_default_fallback_used: bool = False
    structured_top_confidence_value: Optional[float] = None
    structured_top_probability: Optional[float] = None
    structured_entropy: Optional[float] = None
    structured_top_candidates: list[dict[str, float]] | None = None


def parse_confidence_number(text: str) -> Optional[float]:
    """Extract a numeric confidence value from free-text response."""
    # Look for patterns like "85", "85%", "0.85", "85/100"
    text = text.strip()

    # Try percentage pattern: "85%" or "85 percent"
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if match:
        return float(match.group(1))

    # Try fraction pattern: "85/100"
    match = re.search(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", text)
    if match:
        num, denom = float(match.group(1)), float(match.group(2))
        if denom > 0:
            return (num / denom) * 100

    # Try bare number
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if match:
        raw = match.group(1)
        value = float(raw)
        if "." in raw and 0.0 <= value <= 1.0:
            return value * 100.0
        return value

    return None


@torch.no_grad()
def _score_continuations(
    sampler: SteeredAutoregressiveSampler,
    context: list[int],
    continuations: list[list[int]],
    batch_size: int = 32,
) -> list[float]:
    """Return sequence log-probs for continuations after ``context``.

    The caller may already be inside ``with sampler:`` with a single-example
    hook attached. Candidate scoring needs a batched hook, so this temporarily
    swaps the hook out and restores the original attachment state afterward.
    """
    if not continuations:
        return []
    if any(len(ids) == 0 for ids in continuations):
        raise ValueError("Cannot score an empty continuation")

    pad_id = sampler._get_pad_token_id()
    if pad_id is None:
        pad_id = sampler.tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("pad_token_id or eos_token_id is required for scoring")

    had_hook = sampler._handle is not None
    if had_hook:
        sampler.detach_hook()

    scores: list[float] = []
    try:
        for start in range(0, len(continuations), batch_size):
            batch = continuations[start : start + batch_size]
            batch_len = len(batch)
            sequences = [context + ids for ids in batch]
            max_len = max(len(seq) for seq in sequences)

            input_rows = []
            mask_rows = []
            for seq in sequences:
                pad_len = max_len - len(seq)
                input_rows.append(seq + [pad_id] * pad_len)
                mask_rows.append([1] * len(seq) + [0] * pad_len)

            input_ids = torch.tensor(
                input_rows, dtype=torch.long, device=sampler.device
            )
            attention_mask = torch.tensor(
                mask_rows, dtype=torch.long, device=sampler.device
            )

            hook = get_hf_activation_steering_hook(
                vectors=list(sampler._steering_vectors) * batch_len,
                positions=list(sampler._positions) * batch_len,
                steering_coefficient=sampler._steering_coefficient,
                device=sampler.device,
                dtype=sampler._dtype,
            )
            handle = sampler.submodule.register_forward_hook(hook)
            try:
                logits = sampler.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits
            finally:
                handle.remove()

            context_len = len(context)
            for row_idx, token_ids in enumerate(batch):
                token_count = len(token_ids)
                step_logits = logits[
                    row_idx,
                    context_len - 1 : context_len + token_count - 1,
                    :,
                ]
                log_probs = F.log_softmax(step_logits.float(), dim=-1)
                targets = torch.tensor(
                    token_ids, dtype=torch.long, device=logits.device
                )
                score = log_probs.gather(-1, targets.view(-1, 1)).sum().item()
                scores.append(float(score))
    finally:
        if had_hook:
            sampler.attach_hook()

    return scores


def _structured_numeric_confidence(
    sampler: SteeredAutoregressiveSampler,
    context: list[int],
    batch_size: int = 32,
) -> tuple[float, float, float, float, list[dict[str, float]]]:
    """Score integer answers 0..100 and return expected confidence metadata."""
    values = list(range(101))
    stop_ids = sampler._get_generation_stop_ids()
    if isinstance(stop_ids, list):
        stop_id = stop_ids[0] if stop_ids else None
    else:
        stop_id = stop_ids

    tokenized = [
        sampler.tokenizer.encode(str(value), add_special_tokens=False)
        + ([stop_id] if stop_id is not None else [])
        for value in values
    ]
    scores = _score_continuations(
        sampler=sampler,
        context=context,
        continuations=tokenized,
        batch_size=batch_size,
    )

    score_t = torch.tensor(scores, dtype=torch.float32)
    probs = torch.softmax(score_t, dim=0)
    value_t = torch.tensor(values, dtype=torch.float32)
    expected_value = float((probs * value_t).sum().item())
    top_idx = int(torch.argmax(probs).item())
    top_value = float(values[top_idx])
    top_probability = float(probs[top_idx].item())
    entropy = float(-(probs * torch.log(probs.clamp_min(1e-30))).sum().item())

    top_probs, top_indices = torch.topk(probs, k=min(5, len(values)))
    top_candidates = [
        {
            "value": float(values[int(idx.item())]),
            "probability": float(prob.item()),
            "log_score": float(scores[int(idx.item())]),
        }
        for prob, idx in zip(top_probs, top_indices, strict=True)
    ]

    return expected_value, top_value, top_probability, entropy, top_candidates


def _do_sample_for_temperature(temperature: float) -> bool:
    """Use greedy decoding at temperature 0, sampling otherwise."""
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    return temperature > 0.0


def _apply_chat_template(tokenizer, messages: list[dict[str, str]]) -> list[int]:
    context = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors=None,
        return_dict=False,
        padding=False,
        enable_thinking=False,
    )
    if not isinstance(context, list) or (context and not isinstance(context[0], int)):
        raise TypeError("Expected list of token ids from tokenizer.apply_chat_template")
    return context


def _generate_text(
    sampler: SteeredAutoregressiveSampler,
    context: list[int],
    temperature: float,
    max_new_tokens: int,
) -> tuple[str, bool]:
    do_sample = _do_sample_for_temperature(temperature)
    full_seq, _, _ = sampler.generate_with_logprobs(
        context=context,
        temperature=temperature if do_sample else 1.0,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
    )
    generated_ids = full_seq[len(context) :]
    text = sampler.tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text, do_sample


def direct_elicitation(
    sampler: SteeredAutoregressiveSampler,
    context: list[int],
    oracle_messages: list[dict[str, str]],
    confidence_prompt: str = "On a scale of 0 to 100, how confident are you in your answer? Reply with just the number.",
    retry_confidence_prompt: str = "Reply with a single integer from 0 to 100 only.",
    max_new_tokens: int = 20,
    max_confidence_tokens: int = 10,
    answer_temperature: float = 0.0,
    confidence_temperature: float = 0.0,
    retry_on_parse_failure: bool = True,
    structured_fallback: bool = True,
    structured_batch_size: int = 32,
) -> ElicitationResult:
    """Two-turn elicitation: generate answer, then ask for confidence.

    The same steering hook stays active for both turns, so the oracle
    "sees" the same activation context when self-assessing confidence.

    Args:
        sampler: SteeredAutoregressiveSampler with hook attached.
        context: Tokenized oracle prompt for the main question.
        confidence_prompt: Follow-up prompt for confidence.
        retry_confidence_prompt: Stricter retry prompt if free-form parsing fails.
        max_new_tokens: Max tokens for the answer.
        max_confidence_tokens: Max tokens for the confidence response.
        answer_temperature: Turn-1 answer temperature. ``0`` means greedy.
        confidence_temperature: Turn-2 confidence temperature. ``0`` means greedy.
        retry_on_parse_failure: If the first free-form confidence has no
            parseable number, ask one stricter follow-up before structured
            candidate scoring.
        structured_fallback: If the free-form response has no parseable number,
            score the integer candidates 0..100 and use their expected value.
        structured_batch_size: Batch size for structured numeric candidate scoring.

    Returns:
        ElicitationResult with parsed confidence.
    """
    # Turn 1: Generate the answer
    answer_text, answer_do_sample = _generate_text(
        sampler=sampler,
        context=context,
        temperature=answer_temperature,
        max_new_tokens=max_new_tokens,
    )

    # Turn 2: Re-render the conversation as a real follow-up chat turn.
    turn2_messages = oracle_messages + [
        {"role": "assistant", "content": answer_text},
        {"role": "user", "content": confidence_prompt},
    ]
    turn2_context = _apply_chat_template(sampler.tokenizer, turn2_messages)
    confidence_text, confidence_do_sample = _generate_text(
        sampler=sampler,
        context=turn2_context,
        temperature=confidence_temperature,
        max_new_tokens=max_confidence_tokens,
    )

    freeform_confidence_text = confidence_text
    raw_value = parse_confidence_number(confidence_text)
    confidence_parse_failed = raw_value is None
    decode_method = "freeform"
    retry_used = False
    retry_text = None
    structured_fallback_used = False
    default_fallback_used = False
    structured_top_value = None
    structured_top_probability = None
    structured_entropy = None
    structured_top_candidates = None

    if raw_value is None and retry_on_parse_failure:
        retry_used = True
        retry_messages = turn2_messages + [
            {"role": "assistant", "content": confidence_text},
            {"role": "user", "content": retry_confidence_prompt},
        ]
        retry_context = _apply_chat_template(sampler.tokenizer, retry_messages)
        retry_text, _ = _generate_text(
            sampler=sampler,
            context=retry_context,
            temperature=confidence_temperature,
            max_new_tokens=max_confidence_tokens,
        )
        retry_raw_value = parse_confidence_number(retry_text)
        if retry_raw_value is not None:
            raw_value = retry_raw_value
            confidence_text = retry_text
            decode_method = "freeform_retry"

    if raw_value is None and structured_fallback:
        structured_fallback_used = True
        (
            raw_value,
            structured_top_value,
            structured_top_probability,
            structured_entropy,
            structured_top_candidates,
        ) = _structured_numeric_confidence(
            sampler=sampler,
            context=turn2_context,
            batch_size=structured_batch_size,
        )
        confidence_text = f"{raw_value:.2f}"
        decode_method = "structured_numeric_expected_value"

    if raw_value is not None:
        # Clamp to [0, 100] and normalize to [0, 1]
        parsed_confidence = max(0.0, min(100.0, raw_value)) / 100.0
    else:
        # Last-resort sentinel. This should be rare with retry+structured scoring.
        default_fallback_used = True
        parsed_confidence = 0.5

    return ElicitationResult(
        answer_text=answer_text,
        confidence_text=confidence_text,
        parsed_confidence=parsed_confidence,
        raw_confidence_value=raw_value,
        confidence_decode_method=decode_method,
        freeform_confidence_text=freeform_confidence_text,
        answer_temperature=answer_temperature,
        confidence_temperature=confidence_temperature,
        answer_do_sample=answer_do_sample,
        confidence_do_sample=confidence_do_sample,
        confidence_parse_failed=confidence_parse_failed,
        confidence_retry_used=retry_used,
        confidence_retry_text=retry_text,
        confidence_structured_fallback_used=structured_fallback_used,
        confidence_default_fallback_used=default_fallback_used,
        structured_top_confidence_value=structured_top_value,
        structured_top_probability=structured_top_probability,
        structured_entropy=structured_entropy,
        structured_top_candidates=structured_top_candidates,
    )


@torch.no_grad()
def score_linguistic_confidence(
    sampler: SteeredAutoregressiveSampler,
    oracle_messages: list[dict[str, str]],
    answer_text: str,
    prompt: str = LINGUISTIC_PROMPT,
    batch_size: int = 8,
) -> LinguisticConfidenceResult:
    """Score the five verbalized confidence labels under the steered hook.

    Designed to be called after ``direct_elicitation()`` with the answer
    it produced, so the answer turn is not repeated.
    """
    turn2_messages = oracle_messages + [
        {"role": "assistant", "content": answer_text},
        {"role": "user", "content": prompt},
    ]
    turn2_context = _apply_chat_template(sampler.tokenizer, turn2_messages)

    label_ids: list[list[int]] = []
    for label in LINGUISTIC_LABELS:
        ids = sampler.tokenizer.encode(label, add_special_tokens=False)
        if len(ids) == 0:
            ids = sampler.tokenizer.encode(" " + label, add_special_tokens=False)
        if len(ids) == 0:
            raise ValueError(f"Tokenizer returned empty ids for label {label!r}")
        label_ids.append(ids)

    scores = _score_continuations(
        sampler=sampler,
        context=turn2_context,
        continuations=label_ids,
        batch_size=batch_size,
    )
    log_t = torch.tensor(scores, dtype=torch.float32)
    probs_t = torch.softmax(log_t, dim=0)
    probs = [float(p.item()) for p in probs_t]
    expected = float(sum(p * v for p, v in zip(probs, LINGUISTIC_VALUES)))
    top_idx = int(torch.argmax(probs_t).item())
    p_very_high = probs[-1]
    p_high_plus = probs[-1] + probs[-2]

    return LinguisticConfidenceResult(
        labels=list(LINGUISTIC_LABELS),
        label_log_scores=[float(s) for s in scores],
        label_probs=probs,
        top_label=LINGUISTIC_LABELS[top_idx],
        top_label_idx=top_idx,
        expected_value=expected,
        p_very_high=p_very_high,
        p_high_plus=p_high_plus,
        prompt=prompt,
    )
