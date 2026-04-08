"""Steering-coefficient sensitivity for uncertainty quantification.

Sweeps the steering coefficient `c` over a small grid and runs greedy
decoding at each value. The oracle should decode committed activations
stably across the sweep; ambiguous activations should produce different
answers at different magnitudes.

Motivated by Baker et al. 2025 (arXiv:2511.04527), who show that steering
effectiveness correlates with model uncertainty (R≈0.57–0.64). That paper
steers *away* from an answer; here we vary the injection magnitude and
measure decoding stability — the same hypothesis in the oracle setting.

The coefficient reset to 1.0 is done inside a `finally` block so callers
cannot accidentally leave the sampler in a non-default state.
"""

import math
from collections import Counter
from dataclasses import dataclass

from pao.oracle_sampler import SteeredAutoregressiveSampler


@dataclass
class SensitivityResult:
    """Result from the steering-sensitivity method."""
    coefficients: list[float]
    answers: list[str]            # raw greedy answers, one per coefficient
    normalized_answers: list[str]
    per_coef_mean_logprob: list[float]
    mode_answer: str
    mode_frequency: float         # confidence score: fraction agreeing with mode
    entropy: float                # Shannon entropy of the per-sweep answer distribution
    num_unique: int


def _normalize_answer(text: str) -> str:
    return text.strip().lower().rstrip(".!?,;:")


def steering_sensitivity_confidence(
    sampler: SteeredAutoregressiveSampler,
    context: list[int],
    coefficients: list[float],
    max_new_tokens: int = 20,
) -> SensitivityResult:
    """Sweep steering coefficients and measure oracle answer stability.

    The sampler's coefficient is restored to 1.0 (via `finally`) regardless
    of exceptions, so downstream methods in the same `with sampler:` block
    always see the canonical steering strength.

    Args:
        sampler: SteeredAutoregressiveSampler with hook already attached.
        context: Tokenized oracle prompt (including steering token positions).
        coefficients: List of steering coefficients to evaluate.
        max_new_tokens: Maximum tokens per greedy generation.

    Returns:
        SensitivityResult with per-coefficient answers and confidence metrics.
    """
    original_coef = sampler._steering_coefficient
    raw_answers: list[str] = []
    mean_logprobs: list[float] = []

    try:
        for c in coefficients:
            sampler.set_steering_coefficient(c)
            full_seq, log_probs = sampler.greedy_generate(
                context=context,
                max_new_tokens=max_new_tokens,
            )
            generated_ids = full_seq[len(context):]
            text = sampler.tokenizer.decode(generated_ids, skip_special_tokens=True)
            raw_answers.append(text)

            mean_lp = sum(log_probs) / len(log_probs) if log_probs else float("-inf")
            mean_logprobs.append(mean_lp)
    finally:
        sampler.set_steering_coefficient(original_coef)

    normalized = [_normalize_answer(a) for a in raw_answers]
    counts = Counter(normalized)
    total = len(normalized)
    mode_answer, mode_count = counts.most_common(1)[0]
    mode_frequency = mode_count / total

    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log(p)

    return SensitivityResult(
        coefficients=list(coefficients),
        answers=raw_answers,
        normalized_answers=normalized,
        per_coef_mean_logprob=mean_logprobs,
        mode_answer=mode_answer,
        mode_frequency=mode_frequency,
        entropy=entropy,
        num_unique=len(counts),
    )
