"""Bootstrap 95% CIs for ECE / Brier / NLL / AUROC over per-sample data.

For each (model, method-temperature) row in results/, resample with
replacement B times over the 6000 samples and report the central 95%
interval for each metric. Output: results/bootstrap_cis/{model}.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pao.calibration.metrics import (
    auroc,
    expected_calibration_error,
    negative_log_likelihood,
)

B = 1000  # bootstrap resamples
ROOT = Path("results")
OUT = Path("results/bootstrap_cis")

MODELS = {
    "qwen3-8b": Path("results/qwen3-8b/run_3"),
    "qwen3.6-27b": Path("results/qwen3.6-27b/taboo_uq"),
}

METHOD_FILES = [
    "logprob_offset_results.json",
    "logprob_no_offset_results.json",
    "bootstrap_t0p3_results.json",
    "bootstrap_t0p5_results.json",
    "bootstrap_t0p7_results.json",
    "bootstrap_t1p0_results.json",
    "bootstrap_t1p3_results.json",
    "bootstrap_t1p5_results.json",
    "direct_results.json",
    "mcmc_t0p125_results.json",
    "mcmc_t0p25_results.json",
    "mcmc_t0p5_results.json",
    "mcmc_agreement_t0p125_results.json",
    "mcmc_agreement_t0p25_results.json",
    "mcmc_agreement_t0p5_results.json",
    "sensitivity_results.json",
]


def load(path: Path):
    with open(path) as f:
        d = json.load(f)
    preds = d["predictions"]
    confs, correct, accs = [], [], []
    for p in preds:
        try:
            c = float(p["confidence"])
        except (KeyError, ValueError, TypeError):
            continue
        is_correct = p.get("is_correct")
        if isinstance(is_correct, str):
            is_correct = is_correct.lower() == "true"
        confs.append(c)
        correct.append(bool(is_correct))
    return np.array(confs), np.array(correct)


def metrics_on_sample(confs, correct):
    confs = confs.tolist()
    correct = correct.tolist()
    cal = expected_calibration_error(confs, correct)
    return (
        cal.ece,
        cal.brier_score,
        negative_log_likelihood(confs, correct),
        auroc(confs, correct),
        float(np.mean(correct)),
    )


def bootstrap_metrics(confs, correct, B=B, seed=0):
    rng = np.random.default_rng(seed)
    n = len(confs)
    samples = np.empty((B, 5), dtype=float)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        samples[b] = metrics_on_sample(confs[idx], correct[idx])
    point = np.array(metrics_on_sample(confs, correct))
    lo = np.percentile(samples, 2.5, axis=0)
    hi = np.percentile(samples, 97.5, axis=0)
    se = np.std(samples, axis=0, ddof=1)
    names = ["ece", "brier", "nll", "auroc", "acc"]
    return {
        name: {
            "point": float(point[i]),
            "lo": float(lo[i]),
            "hi": float(hi[i]),
            "se": float(se[i]),
        }
        for i, name in enumerate(names)
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for model, root in MODELS.items():
        out = {}
        for fname in METHOD_FILES:
            fp = root / fname
            if not fp.exists():
                continue
            confs, correct = load(fp)
            if len(confs) == 0:
                continue
            key = fname.replace("_results.json", "")
            out[key] = bootstrap_metrics(
                confs, correct, B=B, seed=hash(key) & 0xFFFFFFFF
            )
            print(
                f"{model:14s} {key:30s} ECE={out[key]['ece']['point']:.3f} "
                f"({out[key]['ece']['lo']:.3f}, {out[key]['ece']['hi']:.3f})"
            )
        with open(OUT / f"{model}.json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {OUT / (model + '.json')}\n")


if __name__ == "__main__":
    main()
