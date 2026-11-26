"""Calibration analyses of the bootstrap temperature and of the main method rows.

1. Label-free bootstrap temperature chosen leave-one-oracle-out.
2. Bootstrap at the temperature tuned on five labeled words, scored on the other
   words, for the hero figure and the main table.
3. Crossed (pigeonhole) 95% intervals and within-word ECE for the main rows.
4. Maximum ECE across oracles for the free-text methods, on paired resamples.

Writes results/calibration_analyses/calibration_analyses.json. Tables: calibration_tables.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from resampling_analyses import ORACLES, auroc, ece, load_grid, resample_index

OUT = Path("results/calibration_analyses")
B = 1000
GRID = ("0p3", "0p5", "0p7", "1p0", "1p3", "1p5")  # temperatures run on all four oracles
FIT_WORDS, SPLITS = 5, 200
FREE_TEXT = ("logprob_offset", "direct_linguistic_p_very_high", "direct")


def boot_key(tag: str) -> str:
    return f"bootstrap_t{tag}"


def temp(tag: str) -> float:
    return float(tag.replace("p", "."))


def full_ece(model: str, key: str) -> float:
    conf, corr, _ = load_grid(model, key)
    return float(ece(conf.ravel(), corr.ravel()))


def loo_temperature(eces: dict[str, dict[str, float]]) -> dict:
    """Pick T with the smallest maximum ECE on three oracles, score it on the fourth."""
    out = {}
    for held in eces:
        rest = [m for m in eces if m != held]
        pick = min(GRID, key=lambda t: max(eces[m][t] for m in rest))
        out[held] = {"t": temp(pick), "test_ece": eces[held][pick], "t1_ece": eces[held]["1p0"]}
    return out


def tuned_on_words(model: str, rng: np.random.Generator) -> dict:
    """T matched to accuracy on FIT_WORDS labeled words, ECE on the other words."""
    keys = sorted(p.stem.removesuffix("_results") for d in ORACLES[model][1]
                  for p in Path(d).glob("bootstrap_t*_results.json"))
    grids = {k: load_grid(model, k)[:2] for k in keys}
    n_words = next(iter(grids.values()))[0].shape[0]
    eces, aurocs, picks = [], [], []
    for _ in range(SPLITS):
        perm = rng.permutation(n_words)
        fit, test = perm[:FIT_WORDS], perm[FIT_WORDS:]
        k = min(grids, key=lambda k: abs(grids[k][0][fit].mean() - grids[k][1][fit].mean()))
        c, r = grids[k][0][test].ravel(), grids[k][1][test].ravel()
        eces.append(ece(c, r))
        aurocs.append(auroc(c, r))
        picks.append(temp(k.removeprefix("bootstrap_t")))
    return {
        "ece_mean": float(np.mean(eces)), "ece_p5_p95": np.percentile(eces, [5, 95]).tolist(),
        "auroc_mean": float(np.mean(aurocs)),
        "picked_t": {str(t): picks.count(t) / SPLITS for t in sorted(set(picks))},
    }


def within_word_ece(conf: np.ndarray, corr: np.ndarray) -> float:
    """ECE inside each secret word, mean weighted by samples (all words have 300)."""
    return float(np.mean([ece(conf[w].ravel(), corr[w].ravel()) for w in range(conf.shape[0])]))


def crossed_intervals(model: str, keys: list[str], rng: np.random.Generator) -> dict:
    data = {k: load_grid(model, k)[:2] for k in keys}
    shape = next(iter(data.values()))[0].shape
    draws = {k: {"ece": [], "auroc": []} for k in keys}
    for _ in range(B):
        idx = resample_index(shape, "crossed", rng)
        for k, (c, r) in data.items():
            draws[k]["ece"].append(ece(c[idx].ravel(), r[idx].ravel()))
            draws[k]["auroc"].append(auroc(c[idx].ravel(), r[idx].ravel()))
    return {
        k: {
            "ece": float(ece(c.ravel(), r.ravel())), "auroc": float(auroc(c.ravel(), r.ravel())),
            "accuracy": float(r.mean()), "within_word_ece": within_word_ece(c, r),
            "ece_ci": np.percentile(draws[k]["ece"], [2.5, 97.5]).tolist(),
            "auroc_ci": np.percentile(draws[k]["auroc"], [2.5, 97.5]).tolist(),
            "ece_draws": draws[k]["ece"],
        }
        for k, (c, r) in data.items()
    }


def max_ece_shares(intervals: dict, boot: str) -> dict:
    """Share of paired resamples where the bootstrap's maximum ECE across oracles is lower."""
    def max_over_oracles(key):
        return np.max([intervals[m][key]["ece_draws"] for m in intervals], axis=0)
    b = max_over_oracles(boot)
    return {k: float(np.mean(b < max_over_oracles(k))) for k in FREE_TEXT}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    eces = {m: {t: full_ece(m, boot_key(t)) for t in GRID} for m in ORACLES}
    loo = loo_temperature(eces)
    rng = np.random.default_rng(0)
    tuned = {m: tuned_on_words(m, rng) for m in ORACLES}
    loo_t = max(set(v["t"] for v in loo.values()), key=[v["t"] for v in loo.values()].count)
    loo_boot = boot_key(f"{loo_t}".replace(".", "p"))
    keys = ["forced_choice", "logprob_offset", "bootstrap_t1p0", loo_boot, *FREE_TEXT[1:]]
    intervals = {m: crossed_intervals(m, keys, np.random.default_rng(1)) for m in ORACLES}
    shares = {boot: max_ece_shares(intervals, boot) for boot in ("bootstrap_t1p0", loo_boot)}
    slim = {m: {k: {x: y for x, y in v.items() if x != "ece_draws"} for k, v in d.items()}
            for m, d in intervals.items()}
    result = {"grid_ece": eces, "loo_temperature": loo, "tuned_on_5_words": tuned,
              "main_keys": keys, "main_rows": slim, "max_ece_share_bootstrap_lower": shares}
    (OUT / "calibration_analyses.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"loo": loo, "tuned": tuned, "shares": shares}, indent=1))


if __name__ == "__main__":
    main()
