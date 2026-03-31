"""Temperature bootstrap for uncertainty quantification.

Runs k oracle generations with nonzero temperature on the same activation,
then computes empirical answer distribution and agreement metrics.
"""

import math
from collections import Counter
from dataclasses import dataclass, field

from pao.oracle_sampler import SteeredAutoregressiveSampler


@dataclass
class BootstrapResult:
    """Result from temperature bootstrap method."""
    samples: list[str]
    answer_counts: dict[str, int]
    mode_answer: str
    mode_frequency: float  # fraction of samples agreeing with mode
    entropy: float  # entropy of empirical answer distribution
    num_unique: int
    temperature: float
    k: int


def _normalize_answer(text: str) -> str:
    """Normalize an oracle answer for comparison."""
    return text.strip().lower().rstrip(".!?,;:")


def _empirical_entropy(counts: Counter, total: int) -> float:
    """Shannon entropy of an empirical distribution."""
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log(p)
    return entropy


def temperature_bootstrap(
    sampler: SteeredAutoregressiveSampler,
    context: list[int],
    k: int = 20,
    temperature: float = 0.7,
    max_new_tokens: int = 20,
) -> BootstrapResult:
    """Run k temperature-sampled generations and compute agreement metrics.

    Args:
        sampler: SteeredAutoregressiveSampler with hook attached.
        context: Tokenized oracle prompt.
        k: Number of samples to draw.
        temperature: Sampling temperature.
        max_new_tokens: Max tokens per sample.

    Returns:
        BootstrapResult with empirical distribution statistics.
    """
    raw_samples = []
    for _ in range(k):
        full_seq, _, _ = sampler.generate_with_logprobs(
            context=context,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            do_sample=True,
        )
        generated_ids = full_seq[len(context):]
        text = sampler.tokenizer.decode(generated_ids, skip_special_tokens=True)
        raw_samples.append(text)

    normalized = [_normalize_answer(s) for s in raw_samples]
    counts = Counter(normalized)
    mode_answer, mode_count = counts.most_common(1)[0]
    mode_frequency = mode_count / k
    entropy = _empirical_entropy(counts, k)

    return BootstrapResult(
        samples=raw_samples,
        answer_counts=dict(counts),
        mode_answer=mode_answer,
        mode_frequency=mode_frequency,
        entropy=entropy,
        num_unique=len(counts),
        temperature=temperature,
        k=k,
    )
