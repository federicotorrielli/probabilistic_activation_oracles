"""MCMC power sampling for activation oracle uncertainty quantification.

The key novel experiment: apply MCMC power sampling (from reasoning-with-sampling)
to activation-steered oracle models. This produces samples from p^alpha of the
steered oracle, giving sharper and potentially better-calibrated distributions
than temperature sampling.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from pao.answer_extraction import extract_predicted_word
from pao.methods.temperature_bootstrap import _empirical_entropy
from pao.oracle_sampler import (
    SteeredAutoregressiveSampler,
    max_swap_steered,
    mcmc_power_samp_steered,
)


@dataclass
class MCMCResult:
    """Result from a single MCMC power sampling run."""

    generated_text: str
    token_ids: list[int]
    log_probs_norm: list[float]
    log_probs_unnorm: list[float]
    acceptance_ratio: float
    alpha: float  # = 1/temperature
    temperature: float
    mcmc_steps: int
    block_num: int


@dataclass
class MCMCAgreementResult:
    """Result from k independent MCMC samples measuring agreement."""

    samples: list[MCMCResult]
    normalized_samples: list[str]  # extracted words, one per chain
    answer_counts: dict[str, int]
    mode_answer: str  # extracted-word mode
    mode_frequency: float
    entropy: float
    num_unique: int
    mean_acceptance_ratio: float
    k: int
    alpha: float


def mcmc_oracle_sample(
    sampler: SteeredAutoregressiveSampler,
    context: list[int],
    temperature: float = 0.25,
    mcmc_steps: int = 5,
    max_new_tokens: int = 20,
    block_num: int = 4,
    use_max_swap: bool = False,
) -> MCMCResult:
    """Run one MCMC power sampling chain on the steered oracle.

    Args:
        sampler: SteeredAutoregressiveSampler with hook attached.
        context: Tokenized oracle prompt.
        temperature: Proposal temperature (alpha = 1/temperature).
        mcmc_steps: Number of MH steps per block.
        max_new_tokens: Must be divisible by block_num.
        block_num: Number of blocks to divide generation into.
        use_max_swap: If True, use greedy max_swap instead of MH.

    Returns:
        MCMCResult with generated text and diagnostics.
    """
    sample_fn = max_swap_steered if use_max_swap else mcmc_power_samp_steered

    gen, lp_norm, lp_unnorm, acc_ratio = sample_fn(
        sampler=sampler,
        context=context,
        temp=temperature,
        mcmc_steps=mcmc_steps,
        max_new_tokens=max_new_tokens,
        block_num=block_num,
    )

    generated_ids = gen[len(context) :]
    text = sampler.tokenizer.decode(generated_ids, skip_special_tokens=True)

    return MCMCResult(
        generated_text=text,
        token_ids=generated_ids,
        log_probs_norm=lp_norm,
        log_probs_unnorm=lp_unnorm,
        acceptance_ratio=acc_ratio,
        alpha=1.0 / temperature,
        temperature=temperature,
        mcmc_steps=mcmc_steps,
        block_num=block_num,
    )


def mcmc_agreement(
    sampler: SteeredAutoregressiveSampler,
    context: list[int],
    answer_vocab: Sequence[str],
    k: int = 10,
    temperature: float = 0.25,
    mcmc_steps: int = 5,
    max_new_tokens: int = 20,
    block_num: int = 4,
    use_max_swap: bool = False,
) -> MCMCAgreementResult:
    """Run k independent MCMC chains and measure agreement.

    High agreement = low uncertainty. Power sampling's diversity preservation
    means disagreement is informative rather than being caused by mode collapse.

    Args:
        sampler: SteeredAutoregressiveSampler with hook attached.
        context: Tokenized oracle prompt.
        k: Number of independent MCMC chains to run.
        temperature: Proposal temperature.
        mcmc_steps: MH steps per block.
        max_new_tokens: Must be divisible by block_num.
        block_num: Number of generation blocks.
        use_max_swap: Use greedy variant.

    Returns:
        MCMCAgreementResult with agreement statistics.
    """
    results = []
    for _ in range(k):
        result = mcmc_oracle_sample(
            sampler=sampler,
            context=context,
            temperature=temperature,
            mcmc_steps=mcmc_steps,
            max_new_tokens=max_new_tokens,
            block_num=block_num,
            use_max_swap=use_max_swap,
        )
        results.append(result)

    normalized = [
        extract_predicted_word(r.generated_text, answer_vocab) for r in results
    ]
    counts = Counter(normalized)
    mode_answer, mode_count = counts.most_common(1)[0]

    acc_ratios = [r.acceptance_ratio for r in results]
    mean_acc = sum(acc_ratios) / len(acc_ratios)

    return MCMCAgreementResult(
        samples=results,
        normalized_samples=normalized,
        answer_counts=dict(counts),
        mode_answer=mode_answer,
        mode_frequency=mode_count / k,
        entropy=_empirical_entropy(counts, k),
        num_unique=len(counts),
        mean_acceptance_ratio=mean_acc,
        k=k,
        alpha=1.0 / temperature,
    )
