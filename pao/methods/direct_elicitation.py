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

from pao.oracle_sampler import SteeredAutoregressiveSampler


@dataclass
class ElicitationResult:
    """Result from direct confidence elicitation."""

    answer_text: str
    confidence_text: str
    parsed_confidence: float  # 0-1 scale
    raw_confidence_value: Optional[float]  # raw number from model (0-100)


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
        return float(match.group(1))

    return None


def direct_elicitation(
    sampler: SteeredAutoregressiveSampler,
    context: list[int],
    confidence_prompt: str = "On a scale of 0 to 100, how confident are you in your answer? Reply with just the number.",
    max_new_tokens: int = 20,
    max_confidence_tokens: int = 10,
) -> ElicitationResult:
    """Two-turn elicitation: generate answer, then ask for confidence.

    The same steering hook stays active for both turns, so the oracle
    "sees" the same activation context when self-assessing confidence.

    Args:
        sampler: SteeredAutoregressiveSampler with hook attached.
        context: Tokenized oracle prompt for the main question.
        confidence_prompt: Follow-up prompt for confidence.
        max_new_tokens: Max tokens for the answer.
        max_confidence_tokens: Max tokens for the confidence response.

    Returns:
        ElicitationResult with parsed confidence.
    """
    # Turn 1: Generate the answer
    full_seq, _, _ = sampler.generate_with_logprobs(
        context=context,
        temperature=1.0,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    answer_ids = full_seq[len(context) :]
    answer_text = sampler.tokenizer.decode(answer_ids, skip_special_tokens=True)

    # Turn 2: Append the answer and confidence prompt, generate again
    # Build: [original_context] + [answer] + [confidence_prompt]
    confidence_suffix = f" {answer_text}\n{confidence_prompt}"
    confidence_ids = sampler.tokenizer.encode(
        confidence_suffix, add_special_tokens=False
    )
    turn2_context = full_seq + confidence_ids

    turn2_seq, _, _ = sampler.generate_with_logprobs(
        context=turn2_context,
        temperature=1.0,
        max_new_tokens=max_confidence_tokens,
        do_sample=False,
    )

    confidence_response_ids = turn2_seq[len(turn2_context) :]
    confidence_text = sampler.tokenizer.decode(
        confidence_response_ids, skip_special_tokens=True
    )

    # Parse confidence
    raw_value = parse_confidence_number(confidence_text)
    if raw_value is not None:
        # Clamp to [0, 100] and normalize to [0, 1]
        parsed_confidence = max(0.0, min(100.0, raw_value)) / 100.0
    else:
        # Fallback: 0.5 (maximum uncertainty) if we can't parse
        parsed_confidence = 0.5

    return ElicitationResult(
        answer_text=answer_text,
        confidence_text=confidence_text,
        parsed_confidence=parsed_confidence,
        raw_confidence_value=raw_value,
    )
