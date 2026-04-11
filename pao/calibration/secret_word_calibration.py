"""Secret word calibration protocol.

The taboo task has 20 secret words with separate target LoRAs. This module
implements a calibration test that measures how well each UQ method's
confidence scores correspond to actual accuracy.

For a perfectly calibrated method:
- Confidence ~1.0 for the correct word's activation
- Confidence ~0.0 for incorrect words' activations
- Average confidence across all 20 words should be ~1/20 for wrong words

The controlled-N variant uses subsets of size N in {2, 5, 10, 20} to test
calibration at different difficulty levels.
"""

import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from pao.calibration.metrics import (
    CalibrationResult,
    expected_calibration_error,
    negative_log_likelihood,
    print_calibration_summary,
)
from pao.config import TABOO_WORDS, VERBALIZER_PROMPTS_TABOO, ModelConfig


@dataclass
class WordPrediction:
    """A single prediction for one (target_word, context_prompt) pair."""

    target_word: str
    context_prompt: str
    verbalizer_prompt: str
    predicted_answer: str
    confidence: float
    is_correct: bool
    method: str
    method_metadata: dict = field(default_factory=dict)


@dataclass
class CalibrationTestResult:
    """Results from the full calibration protocol for one method."""

    method: str
    predictions: list[WordPrediction]
    calibration: CalibrationResult
    per_word_accuracy: dict[str, float]
    overall_accuracy: float
    nll: float
    n_words: int
    n_prompts: int


def run_calibration_test(
    predict_fn: Callable,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    method_name: str,
    target_words: list[str] = TABOO_WORDS,
    verbalizer_prompts: Optional[list[str]] = None,
    context_prompts: Optional[list[str]] = None,
    model_config: Optional[ModelConfig] = None,
    max_prompts: Optional[int] = None,
) -> CalibrationTestResult:
    """Run the secret word calibration test for a given UQ method.

    Args:
        predict_fn: A callable with signature:
            (model, tokenizer, device, target_word, context_prompt, verbalizer_prompt, model_config)
            -> (predicted_answer: str, confidence: float, metadata: dict)
            The predict_fn is responsible for:
            1. Loading the target LoRA
            2. Collecting activations
            3. Setting up steering and generating with the oracle
            4. Returning (answer, confidence, optional metadata)

        model: The base model (with LoRA support).
        tokenizer: Tokenizer for the model.
        device: Torch device.
        method_name: Name of the UQ method being tested.
        target_words: Words to test (default: all 20 taboo words).
        verbalizer_prompts: Oracle prompts (default: standard 3).
        context_prompts: Prompts to send to target model for activation collection.
        model_config: Model configuration for paths etc.
        max_prompts: Limit number of context prompts per word.

    Returns:
        CalibrationTestResult with per-word and aggregate metrics.
    """
    if verbalizer_prompts is None:
        verbalizer_prompts = VERBALIZER_PROMPTS_TABOO
    if model_config is None:
        model_config = ModelConfig()

    predictions: list[WordPrediction] = []

    total = (
        len(target_words)
        * len(verbalizer_prompts)
        * (len(context_prompts) if context_prompts else 1)
    )
    pbar = tqdm(total=total, desc=f"Calibration [{method_name}]")

    for target_word in target_words:
        prompts_to_use = context_prompts or ["Hint me."]
        if max_prompts is not None:
            prompts_to_use = prompts_to_use[:max_prompts]

        for verbalizer_prompt in verbalizer_prompts:
            for ctx_prompt in prompts_to_use:
                predicted, confidence, metadata = predict_fn(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    target_word=target_word,
                    context_prompt=ctx_prompt,
                    verbalizer_prompt=verbalizer_prompt,
                    model_config=model_config,
                )

                is_correct = (
                    predicted.strip().lower().rstrip(".!?,;:") == target_word.lower()
                )

                predictions.append(
                    WordPrediction(
                        target_word=target_word,
                        context_prompt=ctx_prompt,
                        verbalizer_prompt=verbalizer_prompt,
                        predicted_answer=predicted,
                        confidence=confidence,
                        is_correct=is_correct,
                        method=method_name,
                        method_metadata=metadata,
                    )
                )
                pbar.update(1)

    pbar.close()

    # Compute calibration metrics
    confidences = [p.confidence for p in predictions]
    correctness = [p.is_correct for p in predictions]

    calibration = expected_calibration_error(confidences, correctness)
    nll = negative_log_likelihood(confidences, correctness)

    # Per-word accuracy
    word_correct: dict[str, list[bool]] = defaultdict(list)
    for p in predictions:
        word_correct[p.target_word].append(p.is_correct)

    per_word_accuracy = {
        word: sum(results) / len(results) for word, results in word_correct.items()
    }
    overall_accuracy = sum(correctness) / max(len(correctness), 1)

    return CalibrationTestResult(
        method=method_name,
        predictions=predictions,
        calibration=calibration,
        per_word_accuracy=per_word_accuracy,
        overall_accuracy=overall_accuracy,
        nll=nll,
        n_words=len(target_words),
        n_prompts=len(context_prompts) if context_prompts else 1,
    )


def run_controlled_n_test(
    predict_fn: Callable,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    method_name: str,
    n_values: list[int] = [2, 5, 10, 20],
    context_prompts: Optional[list[str]] = None,
    model_config: Optional[ModelConfig] = None,
    seed: int = 42,
    max_prompts: Optional[int] = None,
) -> dict[int, CalibrationTestResult]:
    """Run calibration test with varying numbers of target words.

    Tests whether the method's calibration degrades gracefully as the
    number of possible answers increases.

    Returns:
        Dict mapping N -> CalibrationTestResult.
    """
    rng = random.Random(seed)
    results = {}

    for n in n_values:
        if n > len(TABOO_WORDS):
            continue

        # Sample N words
        words = rng.sample(TABOO_WORDS, n)
        print(f"\n--- Controlled-N test: N={n}, words={words} ---")

        result = run_calibration_test(
            predict_fn=predict_fn,
            model=model,
            tokenizer=tokenizer,
            device=device,
            method_name=f"{method_name}_N{n}",
            target_words=words,
            context_prompts=context_prompts,
            model_config=model_config,
            max_prompts=max_prompts,
        )
        results[n] = result

        print_calibration_summary(result.calibration, f"{method_name} (N={n})")
        print(f"  Overall accuracy: {result.overall_accuracy:.3f}")
        print(f"  NLL: {result.nll:.4f}")

    return results


def save_calibration_results(
    result: CalibrationTestResult,
    output_path: str,
):
    """Save calibration results to JSON."""
    output = {
        "method": result.method,
        "overall_accuracy": result.overall_accuracy,
        "nll": result.nll,
        "n_words": result.n_words,
        "n_prompts": result.n_prompts,
        "calibration": {
            "ece": result.calibration.ece,
            "brier_score": result.calibration.brier_score,
            "bin_confidences": result.calibration.bin_confidences,
            "bin_accuracies": result.calibration.bin_accuracies,
            "bin_counts": result.calibration.bin_counts,
        },
        "per_word_accuracy": result.per_word_accuracy,
        "predictions": [asdict(p) for p in result.predictions],
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved calibration results to {output_path}")
