"""Calibration metrics for evaluating uncertainty quantification methods.

Provides Expected Calibration Error (ECE), Brier score, and reliability
diagram computation for comparing UQ methods.
"""

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class CalibrationResult:
    """Result from calibration analysis."""
    ece: float  # Expected Calibration Error
    brier_score: float
    bin_confidences: list[float]  # average confidence per bin
    bin_accuracies: list[float]  # average accuracy per bin
    bin_counts: list[int]  # number of samples per bin
    num_bins: int
    n_samples: int


def expected_calibration_error(
    confidences: list[float],
    correctness: list[bool],
    num_bins: int = 10,
) -> CalibrationResult:
    """Compute Expected Calibration Error and bin statistics.

    Args:
        confidences: Predicted confidence scores in [0, 1].
        correctness: Whether each prediction was correct.
        num_bins: Number of equal-width bins.

    Returns:
        CalibrationResult with ECE and per-bin statistics.
    """
    assert len(confidences) == len(correctness)
    n = len(confidences)
    if n == 0:
        return CalibrationResult(
            ece=0.0, brier_score=0.0,
            bin_confidences=[], bin_accuracies=[], bin_counts=[],
            num_bins=num_bins, n_samples=0,
        )

    conf = np.array(confidences, dtype=np.float64)
    correct = np.array(correctness, dtype=np.float64)

    # Brier score: mean squared error between confidence and binary outcome
    brier = float(np.mean((conf - correct) ** 2))

    # Bin edges
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []
    ece = 0.0

    for i in range(num_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == num_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)

        count = int(mask.sum())
        bin_counts.append(count)

        if count > 0:
            avg_conf = float(conf[mask].mean())
            avg_acc = float(correct[mask].mean())
            bin_confidences.append(avg_conf)
            bin_accuracies.append(avg_acc)
            ece += count * abs(avg_acc - avg_conf)
        else:
            bin_confidences.append(0.0)
            bin_accuracies.append(0.0)

    ece /= n

    return CalibrationResult(
        ece=ece,
        brier_score=brier,
        bin_confidences=bin_confidences,
        bin_accuracies=bin_accuracies,
        bin_counts=bin_counts,
        num_bins=num_bins,
        n_samples=n,
    )


def negative_log_likelihood(
    confidences: list[float],
    correctness: list[bool],
    eps: float = 1e-10,
) -> float:
    """Average negative log-likelihood of the correct outcome.

    For correct predictions: -log(confidence)
    For incorrect predictions: -log(1 - confidence)
    """
    if len(confidences) == 0:
        return 0.0

    total = 0.0
    for conf, is_correct in zip(confidences, correctness):
        conf = max(min(conf, 1.0 - eps), eps)
        if is_correct:
            total -= math.log(conf)
        else:
            total -= math.log(1.0 - conf)
    return total / len(confidences)


def agreement_accuracy_correlation(
    confidences: list[float],
    correctness: list[bool],
) -> float:
    """Spearman rank correlation between confidence and correctness."""
    from scipy.stats import spearmanr

    if len(confidences) < 3:
        return 0.0

    rho, _ = spearmanr(confidences, [int(c) for c in correctness])
    return float(rho) if not math.isnan(rho) else 0.0


def print_calibration_summary(result: CalibrationResult, method_name: str = ""):
    """Print a human-readable calibration summary."""
    header = f"Calibration Summary: {method_name}" if method_name else "Calibration Summary"
    print(f"\n{'=' * 60}")
    print(f"  {header}")
    print(f"{'=' * 60}")
    print(f"  Samples:     {result.n_samples}")
    print(f"  ECE:         {result.ece:.4f}")
    print(f"  Brier Score: {result.brier_score:.4f}")
    print(f"{'─' * 60}")
    print(f"  {'Bin':>5} {'Count':>6} {'Avg Conf':>10} {'Avg Acc':>10} {'|Gap|':>10}")
    print(f"{'─' * 60}")
    for i in range(result.num_bins):
        count = result.bin_counts[i]
        if count > 0:
            gap = abs(result.bin_accuracies[i] - result.bin_confidences[i])
            print(
                f"  {i+1:>5} {count:>6} {result.bin_confidences[i]:>10.3f} "
                f"{result.bin_accuracies[i]:>10.3f} {gap:>10.3f}"
            )
    print(f"{'=' * 60}\n")
