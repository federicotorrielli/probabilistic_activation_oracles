"""Build the per-layer accuracy figure for Appendix E.

Reads ``results/{preset}/layer_sweep/sweep.json`` for each of the three
presets and produces a 3-panel matplotlib figure (8B / 27B / Gemma 4 31B)
of per-layer task accuracy on the secret-word task.

Trained-layer indices are marked; for hybrid-attention models we color
sliding vs. full-attention layers separately so the spike pattern is
legible.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from transformers import AutoConfig
from pao.hf_utils import get_text_config, resolve_oracle_layers


PRESETS = [
    ("qwen3-8b", "Qwen3-8B (36 layers, full attention)"),
    ("qwen3.6-27b", "Qwen3.6-27B (64 layers, hybrid linear/full)"),
]


def _attention_per_layer(model_name: str) -> list[str] | None:
    """Return ``layer_types`` if the architecture records it; else None."""
    try:
        cfg = get_text_config(AutoConfig.from_pretrained(model_name))
    except Exception:
        return None
    return list(getattr(cfg, "layer_types", []) or []) or None


def _trained_layers(model_name: str) -> list[int]:
    try:
        cfg = get_text_config(AutoConfig.from_pretrained(model_name))
    except Exception:
        return []
    _inject, percents = resolve_oracle_layers(cfg, [25, 50, 75])
    n = cfg.num_hidden_layers
    return sorted({int(round(p / 100 * n)) for p in percents if 0 <= p <= 100})


def _load(preset: str) -> dict | None:
    path = Path(f"results/{preset}/layer_sweep/sweep.json")
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _accuracy_per_layer(payload: dict) -> dict[int, float]:
    """Mean accuracy across (word, context) pairs per layer."""
    by_layer: dict[int, list[bool]] = {}
    for row in payload["rows"]:
        by_layer.setdefault(int(row["layer"]), []).append(bool(row["is_correct"]))
    return {k: float(np.mean(v)) for k, v in by_layer.items()}


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.0), sharey=True)

    for ax, (preset, title) in zip(axes, PRESETS):
        payload = _load(preset)
        if payload is None:
            ax.set_title(f"{title}\n[no data]")
            ax.set_xlabel("layer index")
            continue

        model_name = payload["model_name"]
        num_layers = int(payload["num_layers"])
        acc = _accuracy_per_layer(payload)
        x = np.array(sorted(acc.keys()))
        y = np.array([acc[i] for i in x])

        layer_types = _attention_per_layer(model_name) or []
        # Default: single color (full attention).
        bar_colors = ["#4338ca"] * len(x)
        if len(layer_types) == num_layers:
            for j, idx in enumerate(x):
                t = layer_types[int(idx)].lower()
                if "sliding" in t or "linear" in t:
                    bar_colors[j] = "#94a3b8"  # slate

        ax.bar(x, y, color=bar_colors, width=0.85, edgecolor="none")

        # Mark trained layers
        trained = _trained_layers(model_name)
        for tl in trained:
            ax.axvline(tl, color="#c2410c", lw=0.9, ls="--", alpha=0.75, zorder=0)

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("layer index", fontsize=10)
        ax.set_xlim(-0.7, num_layers - 0.3)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.25, lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("task accuracy (secret-word recovery)", fontsize=10)

    # Legend on the right-most axis only
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_handles = [
        Patch(color="#4338ca", label="full attention"),
        Patch(color="#94a3b8", label="sliding / linear attention"),
        Line2D([0], [0], color="#c2410c", lw=1.2, ls="--",
               label="verbalizer training layer"),
    ]
    axes[-1].legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=8,
        frameon=False,
    )

    fig.suptitle(
        "Where the secret-word direction is probe-readable: per-layer "
        "accuracy of the log-prob baseline.",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    out = Path("paper/figs/layer_sweep.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    print(f"wrote {out} and {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
