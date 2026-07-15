"""Bootstrap sample-count (k) ablation from saved draws.

For each bootstrap results file, recompute the mode-frequency confidence
using only the first k of the 20 saved normalized samples, and report
accuracy / ECE / Brier / NLL / AUROC per k. No inference needed.

Output: results/k_ablation/{model}.json
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from pao.calibration.metrics import (
    auroc,
    expected_calibration_error,
    negative_log_likelihood,
)

KS = [3, 5, 10, 20]
OUT = Path("results/k_ablation")

MODELS = {
    "qwen3-8b": [Path("results/qwen3-8b/run_3")],
    "qwen3.6-27b": [
        Path("results/qwen3.6-27b/taboo_uq"),
        Path("results/qwen3.6-27b/extra_temps_backfill"),
    ],
    "gemma-2-9b": [
        Path("results/gemma-2-9b/taboo_uq"),
        Path("results/gemma-2-9b/extra_temps_backfill"),
    ],
    "gemma-3-27b": [Path("results/gemma-3-27b/taboo_uq")],
}


def metrics_at_k(preds, k):
    confs, correct = [], []
    for p in preds:
        samples = p["method_metadata"]["normalized_samples"][:k]
        if not samples:
            continue
        mode, count = Counter(samples).most_common(1)[0]
        confs.append(count / k)
        correct.append(mode == p["target_word"])
    cal = expected_calibration_error(confs, correct)
    return {
        "acc": float(np.mean(correct)),
        "ece": cal.ece,
        "brier": cal.brier_score,
        "nll": negative_log_likelihood(confs, correct),
        "auroc": auroc(confs, correct),
        "n": len(confs),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for model, roots in MODELS.items():
        out = {}
        files = sorted(fp for root in roots for fp in root.glob("bootstrap_*_results.json"))
        for fp in files:
            with open(fp) as f:
                preds = json.load(f)["predictions"]
            key = fp.name.replace("_results.json", "")
            out[key] = {str(k): metrics_at_k(preds, k) for k in KS}
            row = out[key]
            print(
                f"{model:14s} {key:18s} "
                + "  ".join(
                    f"k={k}: acc={row[str(k)]['acc']:.3f} ece={row[str(k)]['ece']:.3f} "
                    f"auroc={row[str(k)]['auroc']:.3f}"
                    for k in KS
                )
            )
        with open(OUT / f"{model}.json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {OUT / (model + '.json')}\n")


if __name__ == "__main__":
    main()
