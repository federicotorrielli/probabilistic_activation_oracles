"""Log-probability baseline for uncertainty quantification.

Runs greedy decoding on the steered oracle and reports per-token log-probs
alongside the distribution over the *first* generated token. The first-token
distribution is the natural confidence signal for this task: under the chat
template the first assistant content token is the secret word itself, so
``max p(first_token)`` directly measures how concentrated the oracle is on
one candidate word. Mean-log-prob over a 20-token sentence dilutes that
signal with filler tokens, so we keep it as diagnostic metadata only.
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
    first_token_max_prob: float  # primary confidence signal
    geometric_mean_prob: float  # exp(mean_log_prob), diagnostic only


def compute_first_token_stats(
    sampler: SteeredAutoregressiveSampler,
    context: list[int],
) -> tuple[float, float]:
    """Return (entropy, max_prob) of the first generated token distribution."""
    log_probs = sampler.next_token(context)
    probs = torch.exp(log_probs)
    # Entropy of a discrete distribution via sum p*log p.
    entropy = float(-torch.sum(probs * log_probs).item())
    max_prob = float(probs.max().item())
    return entropy, max_prob


def logprob_confidence(
    sampler: SteeredAutoregressiveSampler,
    context: list[int],
    max_new_tokens: int = 20,
) -> LogProbResult:
    """Run greedy generation and extract log-prob confidence scores."""
    first_token_entropy, first_token_max_prob = compute_first_token_stats(
        sampler, context
    )

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
            first_token_max_prob=first_token_max_prob,
            geometric_mean_prob=0.0,
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
        first_token_max_prob=first_token_max_prob,
        geometric_mean_prob=math.exp(mean_lp),
    )
