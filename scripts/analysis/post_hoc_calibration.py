"""Post-hoc calibration baselines for the PAO paper.

For each (model, method-temperature) row in results/, fit four calibrators
(temperature scaling, Platt, isotonic, beta) on a held-out split and
recompute ECE / Brier / NLL on the test slice.

Two splits are reported side by side:
  - word-disjoint: fit on 10 secret words, evaluate on the other 10
  - random: 50/50 random sample split

Output: results/postcal/{model}/{split}.json summarising per-method metrics.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from pao.calibration.metrics import (
    auroc,
    expected_calibration_error,
    negative_log_likelihood,
)

EPS = 1e-6
RESULTS_ROOT = Path("results")
OUT_ROOT = Path("results/postcal")

# Main run dir plus backfill dirs (forced choice, extra temperatures);
# every *_results.json found in these dirs gets calibrated.
MODELS = {
    "qwen3-8b": [
        Path("results/qwen3-8b/run_3"),
        Path("results/qwen3-8b/forced_choice_backfill"),
    ],
    "qwen3.6-27b": [
        Path("results/qwen3.6-27b/taboo_uq"),
        Path("results/qwen3.6-27b/forced_choice_backfill"),
        Path("results/qwen3.6-27b/extra_temps_backfill"),
    ],
    "gemma-2-9b": [
        Path("results/gemma-2-9b/taboo_uq"),
        Path("results/gemma-2-9b/forced_choice_backfill"),
        Path("results/gemma-2-9b/extra_temps_backfill"),
    ],
    "gemma-3-27b": [
        Path("results/gemma-3-27b/taboo_uq"),
        Path("results/gemma-3-27b/forced_choice_backfill"),
    ],
}


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(p, EPS, 1.0 - EPS)


def _logit(p: np.ndarray) -> np.ndarray:
    p = _clip(p)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


# ---------- Calibrators ----------


@dataclass
class TemperatureScaler:
    """1-parameter logit rescale (Guo et al. 2017, adapted to binary)."""

    T: float = 1.0

    def fit(self, p: np.ndarray, y: np.ndarray) -> "TemperatureScaler":
        z = _logit(p)

        def nll(T: float) -> float:
            if T <= 0:
                return 1e10
            q = _clip(_sigmoid(z / T))
            return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))

        res = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
        self.T = float(res.x)
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        z = _logit(p)
        return _clip(_sigmoid(z / self.T))


class PlattScaler:
    """Logistic regression on logit(p) -> y. Two parameters."""

    def __init__(self) -> None:
        self.clf = LogisticRegression(solver="lbfgs", C=1e6)

    def fit(self, p: np.ndarray, y: np.ndarray) -> "PlattScaler":
        z = _logit(p).reshape(-1, 1)
        # If all labels are identical the classifier is degenerate; fall back to mean
        if len(set(y.tolist())) < 2:
            self._const = float(np.mean(y))
            return self
        self.clf.fit(z, y.astype(int))
        self._const = None
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        if getattr(self, "_const", None) is not None:
            return np.full_like(p, self._const, dtype=float)
        z = _logit(p).reshape(-1, 1)
        return _clip(self.clf.predict_proba(z)[:, 1])


class IsotonicCalibrator:
    """Non-parametric monotone calibration."""

    def __init__(self) -> None:
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, p: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        self.iso.fit(p, y.astype(float))
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        return _clip(self.iso.predict(p))


class BetaCalibrator:
    """Beta calibration (Kull et al. 2017): logistic regression on
    features [log(p), -log(1-p)] -> y."""

    def __init__(self) -> None:
        self.clf = LogisticRegression(solver="lbfgs", C=1e6)

    def fit(self, p: np.ndarray, y: np.ndarray) -> "BetaCalibrator":
        p = _clip(p)
        X = np.column_stack([np.log(p), -np.log(1.0 - p)])
        if len(set(y.tolist())) < 2:
            self._const = float(np.mean(y))
            return self
        self.clf.fit(X, y.astype(int))
        self._const = None
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        if getattr(self, "_const", None) is not None:
            return np.full_like(p, self._const, dtype=float)
        p = _clip(p)
        X = np.column_stack([np.log(p), -np.log(1.0 - p)])
        return _clip(self.clf.predict_proba(X)[:, 1])


CALIBRATORS = {
    "temperature": TemperatureScaler,
    "platt": PlattScaler,
    "isotonic": IsotonicCalibrator,
    "beta": BetaCalibrator,
}


# ---------- Data loading + splits ----------


def load_predictions(path: Path):
    with open(path) as f:
        d = json.load(f)
    preds = d["predictions"]
    rows = []
    for p in preds:
        try:
            c = float(p["confidence"])
        except (KeyError, ValueError, TypeError):
            continue
        is_correct = p.get("is_correct")
        if isinstance(is_correct, str):
            is_correct = is_correct.lower() == "true"
        rows.append(
            {
                "word": p.get("target_word", "?"),
                "conf": c,
                "y": bool(is_correct),
            }
        )
    return rows


# Splits are seeded per split kind, so every method within a model is fit and
# evaluated on the identical split; method differences are then not confounded
# by split randomness.


def split_word_disjoint(rows):
    words = sorted({r["word"] for r in rows})
    rng = np.random.default_rng(1)
    rng.shuffle(words)
    fit_words = set(words[: len(words) // 2])
    fit = [r for r in rows if r["word"] in fit_words]
    test = [r for r in rows if r["word"] not in fit_words]
    return fit, test


def split_random(rows):
    rng = np.random.default_rng(2)
    idx = rng.permutation(len(rows))
    half = len(idx) // 2
    fit = [rows[i] for i in idx[:half]]
    test = [rows[i] for i in idx[half:]]
    return fit, test


SPLITS = {
    "word_disjoint": split_word_disjoint,
    "random_5050": split_random,
}


# ---------- Metric computation ----------


def metrics(confs, correct):
    cr = expected_calibration_error(confs, correct)
    return {
        "ece": cr.ece,
        "brier": cr.brier_score,
        "nll": negative_log_likelihood(confs, correct),
        "auroc": auroc(confs, correct),
        "n": len(confs),
    }


def evaluate_row(rows, fit_rows, test_rows):
    fit_p = np.array([r["conf"] for r in fit_rows], dtype=float)
    fit_y = np.array([1 if r["y"] else 0 for r in fit_rows], dtype=float)
    test_p = np.array([r["conf"] for r in test_rows], dtype=float)
    test_y = np.array([1 if r["y"] else 0 for r in test_rows], dtype=float)

    out = {"uncalibrated": metrics(test_p.tolist(), test_y.astype(bool).tolist())}
    for name, ctor in CALIBRATORS.items():
        cal = ctor().fit(fit_p, fit_y)
        cal_p = cal.predict(test_p)
        out[name] = metrics(cal_p.tolist(), test_y.astype(bool).tolist())
        if name == "temperature":
            out["temperature_T"] = cal.T
    return out


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for model, roots in MODELS.items():
        files = sorted(fp for root in roots for fp in root.glob("*_results.json"))
        for split_name, splitter in SPLITS.items():
            per_method = {}
            for fp in files:
                rows = load_predictions(fp)
                if not rows:
                    continue
                method_key = fp.name.replace("_results.json", "")
                fit_rows, test_rows = splitter(rows)
                per_method[method_key] = evaluate_row(rows, fit_rows, test_rows)
            out_path = OUT_ROOT / model
            out_path.mkdir(parents=True, exist_ok=True)
            with open(out_path / f"{split_name}.json", "w") as f:
                json.dump(per_method, f, indent=2)
            summary[(model, split_name)] = out_path / f"{split_name}.json"
            print(
                f"wrote {out_path / (split_name + '.json')}  ({len(per_method)} methods)"
            )
    print()
    print("Summary written to results/postcal/")


if __name__ == "__main__":
    main()
