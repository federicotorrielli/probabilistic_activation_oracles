"""Temperature bootstrap for uncertainty quantification.

Runs k oracle generations with nonzero temperature on the same activation,
then computes empirical answer distribution and agreement metrics.

Samples are collapsed to a single *word* via ``extract_predicted_word`` before
counting — this matches how ``is_correct`` is computed at the experiment level
and avoids inflating ``num_unique`` on minor surface-form differences in full
sentence outputs.
"""

import math
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from pao.answer_extraction import extract_predicted_word
from pao.oracle_sampler import SteeredAutoregressiveSampler


@dataclass
class BootstrapResult:
    """Result from temperature bootstrap method."""

    samples: list[str]  # raw decoded texts
    normalized_samples: list[str]  # extracted words
    answer_counts: dict[str, int]
    mode_answer: str  # extracted-word mode
    mode_frequency: float  # fraction of samples agreeing with mode
    entropy: float  # Shannon entropy of empirical distribution
    num_unique: int
    temperature: float
    k: int


def _empirical_entropy(counts: Counter, total: int) -> float:
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
    answer_vocab: Sequence[str],
    k: int = 20,
    temperature: float = 0.7,
    max_new_tokens: int = 20,
) -> BootstrapResult:
    """Run k temperature-sampled generations and compute agreement metrics."""
    raw_samples = sampler.generate_batch_texts(
        context=context,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        num_samples=k,
        do_sample=True,
    )

    normalized = [extract_predicted_word(s, answer_vocab) for s in raw_samples]
    counts = Counter(normalized)
    mode_answer, mode_count = counts.most_common(1)[0]
    mode_frequency = mode_count / k
    entropy = _empirical_entropy(counts, k)

    return BootstrapResult(
        samples=raw_samples,
        normalized_samples=normalized,
        answer_counts=dict(counts),
        mode_answer=mode_answer,
        mode_frequency=mode_frequency,
        entropy=entropy,
        num_unique=len(counts),
        temperature=temperature,
        k=k,
    )
