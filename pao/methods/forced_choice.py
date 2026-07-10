"""Closed-vocabulary forced-choice baseline (M7).

Scores each of the candidate taboo words by teacher-forcing it as the oracle's
answer under the steering hook, then normalises over the candidate set. The
confidence is the normalised probability of the top-scoring word, and the
prediction is that word. This removes the answer-extraction step that the other
methods rely on, so it measures directly whether the oracle's distribution over
the candidate vocabulary is calibrated (requested by ARR reviewer 3).

Each word is scored under two surface forms (with and without a leading space);
we keep the higher-probability form, which matches the whitespace hedge already
used by the log-prob baseline. Raw joint log-probabilities drive the headline
readout; length-normalised scores are kept as metadata.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from pao.oracle_sampler import SteeredAutoregressiveSampler


@dataclass
class ForcedChoiceResult:
    predicted_word: str
    confidence: float  # softmax(raw joint log-probs)[argmax]
    words: list[str]
    word_log_probs: list[float]  # raw joint log-prob per word
    word_probs: list[float]  # softmax over raw joint log-probs
    word_log_probs_norm: list[float]  # length-normalised joint log-prob per word
    predicted_word_norm: str  # argmax under length normalisation (diagnostic)
    n_tokens_per_word: list[int]


def _word_token_ids(word: str, tokenizer) -> list[int]:
    """Token ids for the surface form the model assigns higher probability to.

    We score both ``word`` and `` word`` and let the caller pick; here we just
    return the two candidate encodings.
    """
    forms = []
    for text in (f" {word}", word):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            forms.append(ids)
    return forms


@torch.no_grad()
def _score_continuation(
    sampler: SteeredAutoregressiveSampler,
    context: list[int],
    cont_ids: list[int],
) -> float:
    """Joint log-prob of ``cont_ids`` given ``context`` in one steered forward."""
    full = context + cont_ids
    input_ids = torch.tensor([full], dtype=torch.long, device=sampler.device)
    if input_ids.size(1) > sampler.block_size:
        # ponytail: contexts here are ~200 tokens, well under the block size;
        # guard kept for parity with next_token().
        input_ids = input_ids[:, -sampler.block_size :]
    logits = sampler.model(input_ids).logits[0]  # (L, V)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    c = len(context)
    total = 0.0
    for j, tok in enumerate(cont_ids):
        total += float(log_probs[c - 1 + j, tok].item())
    return total


def forced_choice_confidence(
    sampler: SteeredAutoregressiveSampler,
    context: list[int],
    candidate_words: Sequence[str],
) -> ForcedChoiceResult:
    """Score every candidate word and normalise over the candidate set."""
    tokenizer = sampler.tokenizer
    raw_scores: list[float] = []
    norm_scores: list[float] = []
    n_tokens: list[int] = []

    for word in candidate_words:
        best_lp = float("-inf")
        best_n = 1
        for cont_ids in _word_token_ids(word, tokenizer):
            lp = _score_continuation(sampler, context, cont_ids)
            if lp > best_lp:
                best_lp = lp
                best_n = len(cont_ids)
        raw_scores.append(best_lp)
        norm_scores.append(best_lp / best_n)
        n_tokens.append(best_n)

    scores = torch.tensor(raw_scores, dtype=torch.float64)
    probs = torch.softmax(scores, dim=0).tolist()
    top = int(scores.argmax().item())
    top_norm = int(torch.tensor(norm_scores).argmax().item())

    return ForcedChoiceResult(
        predicted_word=list(candidate_words)[top],
        confidence=float(probs[top]),
        words=list(candidate_words),
        word_log_probs=raw_scores,
        word_probs=probs,
        word_log_probs_norm=norm_scores,
        predicted_word_norm=list(candidate_words)[top_norm],
        n_tokens_per_word=n_tokens,
    )


def _demo() -> None:
    """Self-check the normalisation math with a stub scorer (no model)."""

    # softmax of known log-probs, argmax, and confidence-in-[0,1] invariants.
    raw = [math.log(p) for p in (0.6, 0.3, 0.1)]
    scores = torch.tensor(raw, dtype=torch.float64)
    probs = torch.softmax(scores, dim=0).tolist()
    assert abs(sum(probs) - 1.0) < 1e-9
    assert int(scores.argmax()) == 0
    assert abs(probs[0] - 0.6) < 1e-6, probs
    assert 0.0 <= probs[0] <= 1.0
    print("forced_choice._demo OK", probs)


if __name__ == "__main__":
    _demo()
