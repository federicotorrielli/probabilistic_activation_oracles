"""Four-oracle paper figures: pareto (ECE vs AUROC) and reliability diagrams.

Reads results/*/comparison_summary.json and the stored per-method reliability
bins; writes paper/figs/pareto_four.pdf|png and paper/figs/reliability_four.pdf|png.
Designed at 13in wide for a \\textwidth figure* (~0.48 scale), so base fonts
of 14-16pt land at ~7pt in print.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

FIGS = Path("paper/figs")

ORACLES = {
    "qwen3-8b": ("Qwen3-8B", ["results/qwen3-8b/run_3", "results/qwen3-8b/forced_choice_backfill"]),
    "qwen3.6-27b": (
        "Qwen3.6-27B",
        [
            "results/qwen3.6-27b/taboo_uq",
            "results/qwen3.6-27b/forced_choice_backfill",
            "results/qwen3.6-27b/extra_temps_backfill",
        ],
    ),
    "gemma-2-9b": (
        "Gemma-2-9B",
        [
            "results/gemma-2-9b/taboo_uq",
            "results/gemma-2-9b/forced_choice_backfill",
            "results/gemma-2-9b/extra_temps_backfill",
        ],
    ),
    "gemma-3-27b": (
        "Gemma-3-27B",
        ["results/gemma-3-27b/taboo_uq", "results/gemma-3-27b/forced_choice_backfill"],
    ),
}

# Validated categorical palette (dataviz reference order); identity is always
# also carried by direct labels or the legend, never color alone.
BLUE, GREEN, MAGENTA, YELLOW = "#2a78d6", "#008300", "#e87ba4", "#eda100"
INK, SLATE, MUTED, HAIR = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

# Pareto points: (method key, short label, mechanism-group color)
PARETO_METHODS = [
    ("forced_choice", "Forced choice", BLUE),
    ("logprob_offset", "Log-prob", BLUE),
    ("bootstrap_t1p0", "Bootstrap 1.0", GREEN),
    ("bootstrap_t1p5", "Bootstrap 1.5", GREEN),
    ("mcmc_agreement_t0p5", "MCMC agree", GREEN),
    ("direct_linguistic_p_very_high", "Constr. label", MAGENTA),
    ("direct", "Direct", MAGENTA),
    ("sensitivity", "Steering sens.", YELLOW),
    ("mcmc_t0p125", "MCMC accept", YELLOW),
]

GROUPS = [
    ("Answer scoring", BLUE),
    ("Resampling agreement", GREEN),
    ("Self-report", MAGENTA),
    ("Perturbation", YELLOW),
]

# Reliability panels show one method per palette slot.
RELIABILITY_METHODS = {
    "qwen3-8b": [
        ("forced_choice", "Forced choice", BLUE),
        ("bootstrap_t1p0", "Bootstrap $T{=}1.0$", GREEN),
        ("logprob_offset", "Log-prob", MAGENTA),
        ("direct", "Direct (numeric)", YELLOW),
    ],
    "qwen3.6-27b": [
        ("forced_choice", "Forced choice", BLUE),
        ("bootstrap_t1p5", "Bootstrap $T{=}1.5$", GREEN),
        ("logprob_offset", "Log-prob", MAGENTA),
        ("direct", "Direct (numeric)", YELLOW),
    ],
    "gemma-2-9b": [
        ("forced_choice", "Forced choice", BLUE),
        ("bootstrap_t1p5", "Bootstrap $T{=}1.5$", GREEN),
        ("logprob_offset", "Log-prob", MAGENTA),
        ("direct", "Direct (numeric)", YELLOW),
    ],
    "gemma-3-27b": [
        ("forced_choice", "Forced choice", BLUE),
        ("bootstrap_t1p3", "Bootstrap $T{=}1.3$", GREEN),
        ("logprob_offset", "Log-prob", MAGENTA),
        ("direct", "Direct (numeric)", YELLOW),
    ],
}

# Per-point label nudges (dx, dy in data units, horizontal alignment),
# hand-tuned per panel so no two labels collide.
DEFAULT_OFFSET = (0.018, 0.012, "left")
LABEL_OFFSETS = {
    ("qwen3-8b", "bootstrap_t1p0"): (0.0, 0.025, "center"),
    ("qwen3-8b", "bootstrap_t1p5"): (-0.02, -0.05, "center"),
    ("qwen3-8b", "logprob_offset"): (0.018, 0.02, "left"),
    ("qwen3-8b", "mcmc_agreement_t0p5"): (-0.02, -0.05, "center"),
    ("qwen3-8b", "direct_linguistic_p_very_high"): (0.0, 0.025, "center"),
    ("qwen3-8b", "sensitivity"): (0.0, -0.05, "center"),
    ("qwen3.6-27b", "direct_linguistic_p_very_high"): (-0.015, -0.045, "right"),
    ("qwen3.6-27b", "bootstrap_t1p5"): (-0.022, -0.02, "right"),
    ("qwen3.6-27b", "logprob_offset"): (0.02, -0.028, "left"),
    ("qwen3.6-27b", "bootstrap_t1p0"): (0.02, 0.015, "left"),
    ("gemma-2-9b", "logprob_offset"): (0.0, -0.055, "center"),
    ("gemma-2-9b", "bootstrap_t1p5"): (0.0, 0.025, "center"),
    ("gemma-2-9b", "bootstrap_t1p0"): (0.018, 0.015, "left"),
    ("gemma-2-9b", "sensitivity"): (-0.015, -0.05, "right"),
    ("gemma-2-9b", "direct"): (0.0, 0.025, "center"),
    ("gemma-3-27b", "direct_linguistic_p_very_high"): (-0.02, -0.02, "right"),
    ("gemma-3-27b", "bootstrap_t1p5"): (0.02, -0.025, "left"),
    ("gemma-3-27b", "bootstrap_t1p0"): (0.02, 0.005, "left"),
    ("gemma-3-27b", "logprob_offset"): (0.02, 0.03, "left"),
    ("gemma-3-27b", "forced_choice"): (0.02, 0.02, "left"),
    ("gemma-3-27b", "direct"): (0.02, 0.02, "left"),
    ("gemma-3-27b", "mcmc_t0p125"): (0.0, -0.05, "center"),
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 15,
        "axes.titlesize": 17,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 14,
        "axes.edgecolor": "#c3c2b7",
        "axes.linewidth": 0.9,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.labelcolor": SLATE,
        "text.color": INK,
        "savefig.dpi": 200,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def load_summary(model: str) -> dict:
    out = {}
    for d in ORACLES[model][1]:
        fp = Path(d) / "comparison_summary.json"
        if fp.exists():
            out.update(json.load(open(fp)))
    return out


def load_bins(model: str, method: str):
    for d in ORACLES[model][1]:
        fp = Path(d) / f"{method}_results.json"
        if fp.exists():
            cal = json.load(open(fp))["calibration"]
            return cal["bin_confidences"], cal["bin_accuracies"], cal["bin_counts"]
    raise FileNotFoundError(f"{model}/{method}")


def style_axis(ax):
    ax.grid(True, color=HAIR, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def pareto():
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.2), sharex=True, sharey=True)
    for ax, model in zip(axes.flat, ORACLES):
        s = load_summary(model)
        style_axis(ax)
        for key, label, color in PARETO_METHODS:
            x, y = s[key]["ece"], s[key]["auroc"]
            ax.scatter(x, y, s=130, color=color, edgecolor="white", linewidth=1.4, zorder=3)
            dx, dy, ha = LABEL_OFFSETS.get((model, key), DEFAULT_OFFSET)
            ax.annotate(
                label, (x, y), xytext=(x + dx, y + dy), fontsize=13, color=SLATE, ha=ha
            )
        ax.set_title(ORACLES[model][0], color=INK)
        ax.set_xlim(-0.03, 0.85)
        ax.set_ylim(0.35, 1.0)
    for ax in axes[1]:
        ax.set_xlabel("ECE (lower is better)")
    for ax in axes[:, 0]:
        ax.set_ylabel("AUROC (higher is better)")
    handles = [
        plt.Line2D(
            [], [], marker="o", linestyle="", markersize=11, color=c,
            markeredgecolor="white", label=n,
        )
        for n, c in GROUPS
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.015))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIGS / "pareto_four.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "pareto_four.png", bbox_inches="tight")
    plt.close(fig)


def reliability():
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.2), sharex=True, sharey=True)
    for ax, model in zip(axes.flat, ORACLES):
        style_axis(ax)
        ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1.1, linestyle="--", zorder=2)
        for key, label, color in RELIABILITY_METHODS[model]:
            confs, accs, counts = load_bins(model, key)
            pts = [(c, a) for c, a, n in zip(confs, accs, counts) if n > 0]
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=color, linewidth=2.2, marker="o", markersize=8,
                    markeredgecolor="white", markeredgewidth=1.1, label=label, zorder=3)
        ax.set_title(ORACLES[model][0], color=INK)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    for ax in axes[1]:
        ax.set_xlabel("Mean confidence in bin")
    for ax in axes[:, 0]:
        ax.set_ylabel("Accuracy in bin")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    labels = [l.replace("$T{=}1.0$", "tuned $T$") for l in labels]
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.015))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIGS / "reliability_four.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "reliability_four.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    FIGS.mkdir(parents=True, exist_ok=True)
    pareto()
    reliability()
    print("wrote pareto_four and reliability_four under paper/figs/")
