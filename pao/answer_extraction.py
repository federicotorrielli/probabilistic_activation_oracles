"""Shared answer-extraction helpers.

The activation oracle frequently ignores the "answer with a single word only"
instruction and produces sentences like ``'the secret word is a ship'`` or
``'assistant\\n<think>\\n\\nThe secret word is "ship."'``. Comparing the raw
decoded string to the target word via exact match reports ~0% accuracy even
when the correct word is clearly present in the output.

``extract_predicted_word`` finds the *earliest* taboo-vocabulary word that
appears in the decoded text (word-boundary match, case-insensitive). If no
vocabulary word is present it falls back to the first alphabetic token, so
we can still count agreement across samples on non-vocabulary outputs.
"""

import re
from typing import Iterable

_WORD_RE = re.compile(r"[a-z]+")


def extract_predicted_word(text: str, vocab: Iterable[str]) -> str:
    """Return the earliest vocab word appearing in ``text``, or a fallback token.

    Args:
        text: Raw decoded oracle output.
        vocab: Iterable of candidate answer words (lowercased internally).

    Returns:
        The matching vocab word (lowercased), or the first alphabetic token
        if no vocab word is present, or an empty string if the text contains
        no alphabetic characters at all.
    """
    text_lower = text.lower()
    earliest_pos: int | None = None
    earliest_word: str = ""
    for raw_word in vocab:
        w = raw_word.lower()
        if not w:
            continue
        match = re.search(r"\b" + re.escape(w) + r"\b", text_lower)
        if match is None:
            continue
        if earliest_pos is None or match.start() < earliest_pos:
            earliest_pos = match.start()
            earliest_word = w

    if earliest_word:
        return earliest_word

    fallback = _WORD_RE.search(text_lower)
    return fallback.group(0) if fallback else ""
