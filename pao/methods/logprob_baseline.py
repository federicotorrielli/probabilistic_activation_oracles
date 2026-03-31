"""Log-probability baseline for uncertainty quantification.

Extracts per-token log-probabilities from the oracle's steered greedy
generation as confidence scores. This is the simplest UQ method and
serves as a strong baseline. The oracle was trained with cross-entropy
loss, so its log-probs should already reflect epistemic uncertainty about
the activation it received.
"""

import math
from dataclasses import dataclass

import torch

from pao.oracle_sampler import SteeredAutoregressiveSampler


@dataclass
class LogProbResult:
    """Result from log-prob baseline method."""

    generated_text: str
    token_ids: list[int]
    per_token_log_probs: list[float]
    mean_log_prob: float
    min_log_prob: float
    sequence_log_prob: float
    first_token_entropy: float
    normalized_prob: float  # exp(mean_log_prob), i.e. geometric mean token probability


def compute_first_token_entropy(
    sampler: SteeredAutoregressiveSampler,
    context: list[int],
) -> float:
    """Compute entropy of the first generated token's distribution."""
    log_probs = sampler.next_token(context)
    probs = torch.exp(log_probs)
    # Entropy: -sum(p * log(p)), only over nonzero probabilities
    entropy = -torch.sum(probs * log_probs).item()
    return entropy


def logprob_confidence(
    sampler: SteeredAutoregressiveSampler,
    context: list[int],
    max_new_tokens: int = 20,
) -> LogProbResult:
    """Run greedy generation and extract log-prob confidence scores.

    Args:
        sampler: A SteeredAutoregressiveSampler with hook already attached.
        context: Tokenized oracle prompt (including steering token positions).
        max_new_tokens: Maximum tokens to generate.

    Returns:
        LogProbResult with confidence metrics.
    """
    # Get first-token entropy before generation
    first_token_entropy = compute_first_token_entropy(sampler, context)

    # Greedy generation with log-probs
    full_seq, log_probs = sampler.greedy_generate(
        context=context,
        max_new_tokens=max_new_tokens,
    )

    generated_ids = full_seq[len(context) :]
    generated_text = sampler.tokenizer.decode(generated_ids, skip_special_tokens=True)

    if len(log_probs) == 0:
        return LogProbResult(
            generated_text=generated_text,
            token_ids=generated_ids,
            per_token_log_probs=[],
            mean_log_prob=float("-inf"),
            min_log_prob=float("-inf"),
            sequence_log_prob=float("-inf"),
            first_token_entropy=first_token_entropy,
            normalized_prob=0.0,
        )

    mean_lp = sum(log_probs) / len(log_probs)
    min_lp = min(log_probs)
    seq_lp = sum(log_probs)

    return LogProbResult(
        generated_text=generated_text,
        token_ids=generated_ids,
        per_token_log_probs=log_probs,
        mean_log_prob=mean_lp,
        min_log_prob=min_lp,
        sequence_log_prob=seq_lp,
        first_token_entropy=first_token_entropy,
        normalized_prob=math.exp(mean_lp),
    )
