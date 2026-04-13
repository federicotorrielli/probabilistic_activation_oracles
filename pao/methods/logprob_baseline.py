"""Log-probability baseline for uncertainty quantification.

Runs greedy decoding on the steered oracle and reports per-token log-probs.

Two confidence variants are reported:

1) ``word_prob_with_offset`` (char-to-token span mapping):
    joint probability of all subword tokens that encode the extracted answer
    word in the generated text.

2) ``word_prob_no_offset`` (offset-free prefix approximation):
    joint probability of the first N generated tokens, where N is the number
    of tokens needed to encode the extracted answer word. This avoids locating
    char offsets in the generated text.

For single-token words, both variants collapse to a first-token probability.
``first_token_max_prob`` and sentence-level ``geometric_mean_prob`` are kept
as diagnostics.
"""

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable

import torch

from pao.answer_extraction import extract_predicted_word
from pao.oracle_sampler import SteeredAutoregressiveSampler


@dataclass
class LogProbResult:
    """Result from log-prob baseline method."""

    generated_text: str
    extracted_word: str
    token_ids: list[int]
    per_token_log_probs: list[float]
    mean_log_prob: float
    min_log_prob: float
    sequence_log_prob: float
    first_token_entropy: float
    first_token_max_prob: float
    word_prob_with_offset: float  # joint prob using char-to-token span mapping
    word_n_tokens_with_offset: int  # number of tokens in mapped span
    word_prob_no_offset: float  # joint prob of first N tokens (offset-free)
    word_n_tokens_no_offset: int  # N used by the offset-free variant
    geometric_mean_prob: float  # exp(mean_log_prob), diagnostic only


def _word_token_count_no_offset(
    word: str,
    tokenizer: Any,
) -> int:
    """Return an offset-free estimate of how many tokens the word spans.

    We only use tokenization length, not where the word appears in the output.
    Taking the minimum of with/without leading-space tokenizations is a small
    hedge against tokenizer-specific whitespace conventions.
    """
    if not word:
        return 0

    lengths: list[int] = []
    for text in (word, f" {word}"):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if token_ids:
            lengths.append(len(token_ids))

    return min(lengths) if lengths else 0


def compute_first_token_stats(
    sampler: SteeredAutoregressiveSampler,
    context: list[int],
) -> tuple[float, float]:
    """Return (entropy, max_prob) of the first generated token distribution."""
    log_probs = sampler.next_token(context)
    probs = torch.exp(log_probs)
    entropy = float(-torch.sum(probs * log_probs).item())
    max_prob = float(probs.max().item())
    return entropy, max_prob


def _find_word_token_span(
    word: str,
    generated_ids: list[int],
    tokenizer: Any,
) -> tuple[int, int]:
    """Return (start, end) token indices for the first occurrence of ``word``.

    Decodes generated tokens one-by-one, tracking character offsets to find
    which tokens contribute to the matched word. Returns (start_idx, end_idx)
    such that ``generated_ids[start:end]`` are the tokens encoding the word,
    or (0, 0) if the word is not found.
    """
    if not word or not generated_ids:
        return 0, 0

    pattern = re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE)

    # Build a mapping: token index -> (char_start, char_end) in accumulated text
    accumulated = ""
    token_char_ranges: list[tuple[int, int]] = []
    for tid in generated_ids:
        piece = _decode_to_str(tokenizer, [tid])
        start = len(accumulated)
        accumulated += piece
        token_char_ranges.append((start, len(accumulated)))

    match = pattern.search(accumulated)
    if match is None:
        return 0, 0

    match_start, match_end = match.start(), match.end()

    # Find the first token whose text overlaps with the match
    tok_start = 0
    for idx, (cs, ce) in enumerate(token_char_ranges):
        if ce > match_start:
            tok_start = idx
            break

    # Find the last token whose text overlaps with the match
    tok_end = tok_start + 1
    for idx in range(tok_start, len(token_char_ranges)):
        tok_end = idx + 1
        if token_char_ranges[idx][1] >= match_end:
            break

    return tok_start, tok_end


def _decode_to_str(tokenizer: Any, token_ids: list[int]) -> str:
    """Decode token ids and coerce tokenizer-specific return types to str."""
    decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
    if isinstance(decoded, list):
        return "".join(decoded)
    return str(decoded)


def logprob_confidence(
    sampler: SteeredAutoregressiveSampler,
    context: list[int],
    answer_vocab: Iterable[str],
    max_new_tokens: int = 20,
) -> LogProbResult:
    """Run greedy generation and extract log-prob confidence scores.

    Args:
        answer_vocab: Candidate answer words passed to ``extract_predicted_word``.
    """
    first_token_entropy, first_token_max_prob = compute_first_token_stats(
        sampler, context
    )

    full_seq, log_probs = sampler.greedy_generate(
        context=context,
        max_new_tokens=max_new_tokens,
    )

    tokenizer = sampler.tokenizer
    generated_ids = full_seq[len(context) :]
    generated_text = str(_decode_to_str(tokenizer, generated_ids))
    extracted_word = extract_predicted_word(generated_text, answer_vocab)

    if len(log_probs) == 0:
        return LogProbResult(
            generated_text=generated_text,
            extracted_word=extracted_word,
            token_ids=generated_ids,
            per_token_log_probs=[],
            mean_log_prob=float("-inf"),
            min_log_prob=float("-inf"),
            sequence_log_prob=float("-inf"),
            first_token_entropy=first_token_entropy,
            first_token_max_prob=first_token_max_prob,
            word_prob_with_offset=0.0,
            word_n_tokens_with_offset=0,
            word_prob_no_offset=0.0,
            word_n_tokens_no_offset=0,
            geometric_mean_prob=0.0,
        )

    mean_lp = sum(log_probs) / len(log_probs)
    min_lp = min(log_probs)
    seq_lp = sum(log_probs)

    # Variant 1 (with offsets): joint probability of the extracted word's
    # subword tokens where the word appears in the generated text.
    tok_start, tok_end = _find_word_token_span(extracted_word, generated_ids, tokenizer)
    word_n_tokens_with_offset = tok_end - tok_start
    if word_n_tokens_with_offset > 0:
        word_log_prob_with_offset = sum(log_probs[tok_start:tok_end])
        word_prob_with_offset = math.exp(word_log_prob_with_offset)
    else:
        # Word not found — fall back to first-token max prob
        word_prob_with_offset = first_token_max_prob

    # Variant 2 (no offsets): joint probability of first N generated tokens,
    # where N is inferred only from word tokenization length.
    word_n_tokens_no_offset = _word_token_count_no_offset(extracted_word, tokenizer)
    if word_n_tokens_no_offset > 0:
        prefix_end = min(word_n_tokens_no_offset, len(log_probs))
        word_log_prob_no_offset = sum(log_probs[:prefix_end])
        word_prob_no_offset = math.exp(word_log_prob_no_offset)
    else:
        word_prob_no_offset = first_token_max_prob

    return LogProbResult(
        generated_text=generated_text,
        extracted_word=extracted_word,
        token_ids=generated_ids,
        per_token_log_probs=log_probs,
        mean_log_prob=mean_lp,
        min_log_prob=min_lp,
        sequence_log_prob=seq_lp,
        first_token_entropy=first_token_entropy,
        first_token_max_prob=first_token_max_prob,
        word_prob_with_offset=word_prob_with_offset,
        word_n_tokens_with_offset=word_n_tokens_with_offset,
        word_prob_no_offset=word_prob_no_offset,
        word_n_tokens_no_offset=word_n_tokens_no_offset,
        geometric_mean_prob=math.exp(mean_lp),
    )
