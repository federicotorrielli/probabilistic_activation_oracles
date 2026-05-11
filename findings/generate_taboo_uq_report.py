#!/usr/bin/env python3
"""Generate a compact report for the latest taboo UQ run."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results"
FINDINGS_ROOT = ROOT / "findings"

METRICS = [
    ("Accuracy", "accuracy", True, "pct"),
    ("ECE", "ece", False, "pct"),
    ("Brier", "brier_score", False, "float"),
    ("NLL", "nll", False, "float"),
    ("AUROC", "auroc", True, "float"),
]

SELECTED_METHODS = [
    "bootstrap_t1p0",
    "bootstrap_t0p7",
    "bootstrap_t0p5",
    "logprob_offset",
    "mcmc_agreement_t0p5",
    "direct",
]

TREND_METHODS = [
    "bootstrap_t1p0",
    "bootstrap_t0p7",
    "bootstrap_t0p5",
    "logprob_offset",
    "mcmc_agreement_t0p5",
    "mcmc_t0p125",
    "direct",
]

FAMILY_COLORS = {
    "Bootstrap": "#2a9d8f",
    "Log-prob": "#457b9d",
    "Direct": "#e76f51",
    "Steering": "#8d6a9f",
    "MCMC accept": "#f4a261",
    "MCMC agreement": "#577590",
}

METHOD_COLORS = {
    "bootstrap_t1p0": "#2a9d8f",
    "bootstrap_t0p7": "#43aa8b",
    "bootstrap_t0p5": "#90be6d",
    "logprob_offset": "#277da1",
    "mcmc_agreement_t0p5": "#577590",
    "mcmc_t0p125": "#f4a261",
    "direct": "#e76f51",
}


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def method_family(method: str) -> str:
    if method.startswith("bootstrap_"):
        return "Bootstrap"
    if method.startswith("logprob_"):
        return "Log-prob"
    if method == "direct":
        return "Direct"
    if method == "sensitivity":
        return "Steering"
    if method.startswith("mcmc_agreement_"):
        return "MCMC agreement"
    if method.startswith("mcmc_"):
        return "MCMC accept"
    return "Other"


def pretty_method(method: str) -> str:
    if method == "direct":
        return "Direct self-report"
    if method == "sensitivity":
        return "Steering sensitivity"
    if method == "logprob_offset":
        return "Log-prob + offset"
    if method == "logprob_no_offset":
        return "Log-prob"
    if method.startswith("bootstrap_t"):
        return f"Bootstrap T={temperature_label(method.removeprefix('bootstrap_t'))}"
    if method.startswith("mcmc_agreement_t"):
        return f"MCMC agreement T={temperature_label(method.removeprefix('mcmc_agreement_t'))}"
    if method.startswith("mcmc_t"):
        return f"MCMC accept T={temperature_label(method.removeprefix('mcmc_t'))}"
    return method.replace("_", " ")


def short_method(method: str) -> str:
    if method == "direct":
        return "direct"
    if method == "sensitivity":
        return "steering"
    if method == "logprob_offset":
        return "logprob+"
    if method == "logprob_no_offset":
        return "logprob"
    if method.startswith("bootstrap_t"):
        return f"boot {temperature_label(method.removeprefix('bootstrap_t'))}"
    if method.startswith("mcmc_agreement_t"):
        return f"agree {temperature_label(method.removeprefix('mcmc_agreement_t'))}"
    if method.startswith("mcmc_t"):
        return f"mcmc {temperature_label(method.removeprefix('mcmc_t'))}"
    return method


def temperature_label(fragment: str) -> str:
    return fragment.replace("p", ".")


def fmt_value(value: float, kind: str) -> str:
    if kind == "pct":
        return f"{100.0 * value:.1f}%"
    return f"{value:.3f}"


def fmt_metric_value(key: str, value: float) -> str:
    for _, metric_key, _, kind in METRICS:
        if key == metric_key:
            return fmt_value(value, kind)
    return f"{value:.3f}"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def discover_latest_run() -> Path:
    summaries = sorted(RESULTS_ROOT.glob("**/comparison_summary.json"))
    if not summaries:
        raise FileNotFoundError(f"No comparison_summary.json found under {RESULTS_ROOT}")
    return max(summaries, key=lambda path: path.stat().st_mtime).parent


def read_run(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    comparison = load_json(run_dir / "comparison_summary.json")
    checkpoint_path = run_dir / "checkpoint.json"
    checkpoint = load_json(checkpoint_path) if checkpoint_path.exists() else {}
    controlled_path = run_dir / "controlled_n_summary.json"
    controlled = load_json(controlled_path) if controlled_path.exists() else None
    predictions_by_method: dict[str, Any] = {}
    for method in comparison:
        path = run_dir / f"{method}_results.json"
        if path.exists():
            predictions_by_method[method] = load_json(path)
    return comparison, predictions_by_method, checkpoint, controlled


def score_rows(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, values in comparison.items():
        row = {
            "method": method,
            "pretty": pretty_method(method),
            "family": method_family(method),
            "avg_rank": 0.0,
        }
        for _, key, _, _ in METRICS:
            row[key] = float(values[key])
        row["n_samples"] = int(values["n_samples"])
        row["conf_given_correct"] = float(values["conf_given_correct"])
        row["conf_given_wrong"] = float(values["conf_given_wrong"])
        rows.append(row)

    ranks_by_metric: dict[str, dict[str, int]] = {}
    for _, key, higher_better, _ in METRICS:
        ordered = sorted(rows, key=lambda row: row[key], reverse=higher_better)
        ranks_by_metric[key] = {row["method"]: rank for rank, row in enumerate(ordered, start=1)}

    for row in rows:
        row["ranks"] = {key: ranks[row["method"]] for key, ranks in ranks_by_metric.items()}
        row["avg_rank"] = sum(row["ranks"].values()) / len(row["ranks"])

    rows.sort(key=lambda row: (row["avg_rank"], row["ece"]))
    return rows


def metric_winners(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    winners = []
    for label, key, higher_better, kind in METRICS:
        ordered = sorted(
            comparison.items(),
            key=lambda item: float(item[1][key]),
            reverse=higher_better,
        )
        best_method, best_values = ordered[0]
        runner_method, runner_values = ordered[1]
        winners.append(
            {
                "metric": label,
                "winner": best_method,
                "winner_value": fmt_value(float(best_values[key]), kind),
                "runner": runner_method,
                "runner_value": fmt_value(float(runner_values[key]), kind),
            }
        )
    return winners


def word_accuracy(predictions: list[dict[str, Any]]) -> dict[str, tuple[int, int, float]]:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for pred in predictions:
        word = str(pred["target_word"])
        counts[word][0] += int(bool(pred["is_correct"]))
        counts[word][1] += 1
    return {
        word: (correct, total, correct / total if total else math.nan)
        for word, (correct, total) in counts.items()
    }


def binned_reliability(predictions: list[dict[str, Any]], bins: int = 10) -> dict[str, np.ndarray]:
    conf = np.array([float(p["confidence"]) for p in predictions], dtype=float)
    correct = np.array([bool(p["is_correct"]) for p in predictions], dtype=float)
    conf = np.clip(conf, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.searchsorted(edges, conf, side="right") - 1
    bucket = np.clip(bucket, 0, bins - 1)

    mean_conf = np.full(bins, np.nan)
    accuracy = np.full(bins, np.nan)
    count = np.zeros(bins, dtype=int)
    for idx in range(bins):
        mask = bucket == idx
        count[idx] = int(mask.sum())
        if count[idx]:
            mean_conf[idx] = float(conf[mask].mean())
            accuracy[idx] = float(correct[mask].mean())
    centers = (edges[:-1] + edges[1:]) / 2.0
    return {
        "centers": centers,
        "mean_conf": mean_conf,
        "accuracy": accuracy,
        "count": count,
    }


def set_common_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def plot_rank_heatmap(rows: list[dict[str, Any]], out: Path) -> None:
    methods = [row["method"] for row in rows]
    rank_matrix = np.array([[row["ranks"][key] for _, key, _, _ in METRICS] for row in rows])

    fig_h = max(5.8, 0.36 * len(rows) + 1.5)
    fig, ax = plt.subplots(figsize=(8.2, fig_h))
    im = ax.imshow(rank_matrix, cmap="RdYlGn_r", vmin=1, vmax=len(rows), aspect="auto")
    ax.set_xticks(np.arange(len(METRICS)), [label for label, _, _, _ in METRICS])
    ax.set_yticks(np.arange(len(methods)), [pretty_method(method) for method in methods])
    ax.set_title("Metric rank heatmap, lower rank is better")

    for i in range(rank_matrix.shape[0]):
        for j in range(rank_matrix.shape[1]):
            rank = int(rank_matrix[i, j])
            color = "white" if rank > len(rows) * 0.58 else "black"
            ax.text(j, i, str(rank), ha="center", va="center", color=color, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("rank")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_pareto(comparison: dict[str, Any], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    for family in FAMILY_COLORS:
        family_methods = [
            method for method in comparison if method_family(method) == family
        ]
        if not family_methods:
            continue
        x = [100.0 * comparison[method]["ece"] for method in family_methods]
        y = [comparison[method]["auroc"] for method in family_methods]
        size = [80 + 380 * comparison[method]["accuracy"] for method in family_methods]
        ax.scatter(
            x,
            y,
            s=size,
            alpha=0.82,
            label=family,
            color=FAMILY_COLORS[family],
            edgecolor="white",
            linewidth=0.7,
        )
        for method, xx, yy in zip(family_methods, x, y, strict=True):
            ax.annotate(short_method(method), (xx, yy), xytext=(5, 4), textcoords="offset points", fontsize=7)

    ax.set_xlabel("ECE, lower is better")
    ax.set_ylabel("AUROC, higher is better")
    ax.set_title("Calibration vs ranking power")
    ax.legend(loc="lower left", frameon=False, ncols=2)
    ax.set_xlim(left=0)
    ax.set_ylim(0.48, 0.87)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_confidence_split(rows: list[dict[str, Any]], out: Path) -> None:
    ordered = sorted(rows, key=lambda row: row["auroc"], reverse=True)
    y = np.arange(len(ordered))
    correct = np.array([100.0 * row["conf_given_correct"] for row in ordered])
    wrong = np.array([100.0 * row["conf_given_wrong"] for row in ordered])

    fig, ax = plt.subplots(figsize=(8.6, max(5.8, 0.34 * len(ordered) + 1.4)))
    for yi, x0, x1 in zip(y, wrong, correct, strict=True):
        ax.plot([x0, x1], [yi, yi], color="#b8b8b8", linewidth=1.6, zorder=1)
    ax.scatter(wrong, y, color="#d62828", label="wrong", s=38, zorder=2)
    ax.scatter(correct, y, color="#2a9d8f", label="correct", s=38, zorder=2)
    ax.set_yticks(y, [pretty_method(row["method"]) for row in ordered])
    ax.invert_yaxis()
    ax.set_xlabel("Mean confidence")
    ax.set_title("Confidence separation: correct vs wrong answers")
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlim(0, 102)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_reliability_grid(
    predictions_by_method: dict[str, Any],
    comparison: dict[str, Any],
    out: Path,
) -> None:
    methods = [method for method in SELECTED_METHODS if method in predictions_by_method]
    ncols = 3
    nrows = math.ceil(len(methods) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.8, 3.45 * nrows), sharex=True, sharey=True)
    axes_arr = np.atleast_1d(axes).ravel()

    for ax, method in zip(axes_arr, methods, strict=False):
        data = binned_reliability(predictions_by_method[method]["predictions"], bins=10)
        valid = data["count"] > 0
        sizes = 24 + 360 * data["count"][valid] / max(data["count"].max(), 1)
        color = METHOD_COLORS.get(method, FAMILY_COLORS.get(method_family(method), "#444444"))
        ax.plot([0, 1], [0, 1], color="#777777", linewidth=1, linestyle="--")
        ax.scatter(
            data["mean_conf"][valid],
            data["accuracy"][valid],
            s=sizes,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            alpha=0.9,
        )
        ax.plot(data["mean_conf"][valid], data["accuracy"][valid], color=color, linewidth=1.4, alpha=0.75)
        ax.set_title(
            f"{pretty_method(method)}\n"
            f"ECE {100 * comparison[method]['ece']:.1f}%, "
            f"acc {100 * comparison[method]['accuracy']:.1f}%"
        )
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("confidence")
        ax.set_ylabel("observed accuracy")

    for ax in axes_arr[len(methods) :]:
        ax.axis("off")

    fig.suptitle("Reliability fingerprints, point size is bin count", y=1.01, fontsize=13)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_controlled_n(controlled: dict[str, Any], out: Path) -> None:
    n_keys = sorted(controlled, key=lambda key: int(key.removeprefix("N")))
    ns = [int(key.removeprefix("N")) for key in n_keys]

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.7), sharex=True)
    for method in TREND_METHODS:
        if not all(method in controlled[n_key]["methods"] for n_key in n_keys):
            continue
        color = METHOD_COLORS.get(method, FAMILY_COLORS.get(method_family(method), "#444444"))
        ece = [100.0 * controlled[n_key]["methods"][method]["ece"] for n_key in n_keys]
        acc = [100.0 * controlled[n_key]["methods"][method]["accuracy"] for n_key in n_keys]
        label = short_method(method)
        axes[0].plot(ns, ece, marker="o", linewidth=1.8, color=color, label=label)
        axes[1].plot(ns, acc, marker="o", linewidth=1.8, color=color, label=label)

    axes[0].set_title("ECE as target-word set grows")
    axes[0].set_ylabel("ECE (%)")
    axes[1].set_title("Accuracy as target-word set grows")
    axes[1].set_ylabel("accuracy (%)")
    for ax in axes:
        ax.set_xlabel("number of target words")
        ax.set_xticks(ns)
    axes[1].legend(frameon=False, loc="best", ncols=1)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_word_accuracy(predictions_by_method: dict[str, Any], out: Path) -> tuple[str, str]:
    methods = [method for method in SELECTED_METHODS if method in predictions_by_method]
    per_method = {
        method: word_accuracy(predictions_by_method[method]["predictions"])
        for method in methods
    }
    all_words = sorted({word for values in per_method.values() for word in values})
    word_mean = {
        word: float(np.mean([per_method[method][word][2] for method in methods if word in per_method[method]]))
        for word in all_words
    }
    words = sorted(all_words, key=lambda word: word_mean[word], reverse=True)
    matrix = np.array([[per_method[method][word][2] for method in methods] for word in words])

    fig, ax = plt.subplots(figsize=(8.8, max(6.0, 0.34 * len(words) + 1.8)))
    im = ax.imshow(100.0 * matrix, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(np.arange(len(methods)), [short_method(method) for method in methods], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(words)), words)
    ax.set_title("Per-word accuracy by method")
    for i, word in enumerate(words):
        for j, method in enumerate(methods):
            value = 100.0 * matrix[i, j]
            ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=7, color="black" if value < 72 else "white")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("accuracy")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    best_method = "bootstrap_t1p0" if "bootstrap_t1p0" in per_method else methods[0]
    word_rows = sorted(
        per_method[best_method].items(),
        key=lambda item: item[1][2],
        reverse=True,
    )
    top = ", ".join(f"{word} {100 * value[2]:.0f}%" for word, value in word_rows[:4])
    bottom = ", ".join(f"{word} {100 * value[2]:.0f}%" for word, value in word_rows[-4:])
    return top, bottom


def controlled_winner_rows(controlled: dict[str, Any] | None) -> list[dict[str, str]]:
    if not controlled:
        return []
    rows = []
    for n_key in sorted(controlled, key=lambda key: int(key.removeprefix("N"))):
        methods = controlled[n_key]["methods"]
        ece_method = min(methods, key=lambda method: methods[method]["ece"])
        brier_method = min(methods, key=lambda method: methods[method]["brier_score"])
        auroc_method = max(methods, key=lambda method: methods[method]["auroc"])
        acc_method = max(methods, key=lambda method: methods[method]["accuracy"])
        rows.append(
            {
                "n": n_key.removeprefix("N"),
                "samples": str(next(iter(methods.values()))["n_samples"]),
                "ece": f"{pretty_method(ece_method)} ({100 * methods[ece_method]['ece']:.1f}%)",
                "brier": f"{pretty_method(brier_method)} ({methods[brier_method]['brier_score']:.3f})",
                "auroc": f"{pretty_method(auroc_method)} ({methods[auroc_method]['auroc']:.3f})",
                "accuracy": f"{pretty_method(acc_method)} ({100 * methods[acc_method]['accuracy']:.1f}%)",
            }
        )
    return rows


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def build_report(
    run_dir: Path,
    comparison: dict[str, Any],
    rows: list[dict[str, Any]],
    checkpoint: dict[str, Any],
    controlled: dict[str, Any] | None,
    image_paths: dict[str, Path],
    word_top: str,
    word_bottom: str,
    out_path: Path,
) -> None:
    rel_run = run_dir.relative_to(ROOT)
    run_id = "/".join(rel_run.parts)
    timestamp = checkpoint.get("metadata", {}).get("timestamp")
    total_expected = checkpoint.get("metadata", {}).get("total_expected")
    config_hash = checkpoint.get("metadata", {}).get("config_hash")
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    winners = metric_winners(comparison)
    by_metric = {row["metric"]: row for row in winners}
    best_cal = by_metric["ECE"]["winner"]
    best_brier = by_metric["Brier"]["winner"]
    best_nll = by_metric["NLL"]["winner"]
    best_acc = by_metric["Accuracy"]["winner"]
    direct = comparison.get("direct", {})
    raw_mcmc_methods = [method for method in comparison if method.startswith("mcmc_t")]
    worst_mcmc_ece = max((comparison[method]["ece"] for method in raw_mcmc_methods), default=math.nan)

    image_rel = {
        name: path.relative_to(out_path.parent).as_posix()
        for name, path in image_paths.items()
    }

    score_table_rows = []
    for row in rows:
        score_table_rows.append(
            [
                row["pretty"],
                row["family"],
                fmt_value(row["accuracy"], "pct"),
                fmt_value(row["ece"], "pct"),
                fmt_value(row["brier_score"], "float"),
                fmt_value(row["nll"], "float"),
                fmt_value(row["auroc"], "float"),
                f"{row['avg_rank']:.1f}",
            ]
        )

    winner_table_rows = [
        [
            row["metric"],
            pretty_method(row["winner"]),
            row["winner_value"],
            pretty_method(row["runner"]),
            row["runner_value"],
        ]
        for row in winners
    ]

    controlled_rows = controlled_winner_rows(controlled)
    controlled_table = ""
    if controlled_rows:
        controlled_table = markdown_table(
            ["N", "samples", "best ECE", "best Brier", "best AUROC", "best accuracy"],
            [[row["n"], row["samples"], row["ece"], row["brier"], row["auroc"], row["accuracy"]] for row in controlled_rows],
        )

    logprob = comparison.get("logprob_offset", {})
    logprob_sentence = ""
    if logprob:
        logprob_sentence = (
            f"Log-prob + offset is the strongest ranker (AUROC {logprob['auroc']:.3f}) "
            f"but is underconfident: mean confidence is "
            f"{100 * logprob['conf_given_correct']:.1f}% on correct answers and "
            f"{100 * logprob['conf_given_wrong']:.1f}% on wrong answers."
        )

    direct_sentence = ""
    if direct:
        direct_sentence = (
            f"Direct self-report is almost always near-certain "
            f"({100 * direct['conf_given_correct']:.1f}% correct vs "
            f"{100 * direct['conf_given_wrong']:.1f}% wrong), which leaves it with "
            f"ECE {100 * direct['ece']:.1f}% and AUROC {direct['auroc']:.3f}."
        )

    mcmc_sentence = ""
    if raw_mcmc_methods:
        mcmc_sentence = (
            f"Raw MCMC acceptance-ratio confidence is also overconfident "
            f"(worst raw MCMC ECE {100 * worst_mcmc_ece:.1f}%). "
            f"The agreement variant helps, but its best N=20 ECE is still "
            f"{100 * min(comparison[m]['ece'] for m in comparison if m.startswith('mcmc_agreement_')):.1f}%."
        )

    report = f"""# Taboo UQ latest-run compact report

Generated: {generated}

Run: `{run_id}`  
Checkpoint timestamp: `{timestamp}`  
Samples: `{total_expected or comparison[next(iter(comparison))]["n_samples"]}` per full-method summary  
Config hash: `{config_hash}`

## Bottom line

The latest run says the cleanest confidence signal is simple temperature bootstrap at T=1.0. It wins ECE, Brier, and NLL on the full 20-word run: ECE {100 * comparison[best_cal]['ece']:.1f}%, Brier {comparison[best_brier]['brier_score']:.3f}, NLL {comparison[best_nll]['nll']:.3f}. Accuracy is not the differentiator: the best accuracy method is {pretty_method(best_acc)} at {100 * comparison[best_acc]['accuracy']:.1f}%, while {pretty_method(best_cal)} is {100 * comparison[best_cal]['accuracy']:.1f}%.

{logprob_sentence}

{direct_sentence}

{mcmc_sentence}

Per-word accuracy is uneven for the calibrated winner: top words are {word_top}; weakest words are {word_bottom}.

## Metric winners

{markdown_table(["Metric", "winner", "value", "runner-up", "value"], winner_table_rows)}

## Full scorecard

Sorted by average rank across accuracy, ECE, Brier, NLL, and AUROC.

{markdown_table(["method", "family", "acc", "ECE", "Brier", "NLL", "AUROC", "avg rank"], score_table_rows)}

![Rank heatmap]({image_rel["rank_heatmap"]})

## Calibration vs discrimination

The main tradeoff is visible here: log-prob separates correct from wrong answers well, but its probabilities are too small; direct confidence and raw MCMC acceptance ratios are high without enough separation; T=1.0 bootstrap is the best calibrated compromise.

![ECE vs AUROC]({image_rel["pareto"]})

![Confidence split]({image_rel["confidence_split"]})

## Reliability fingerprints

Each point is one confidence bin; larger points have more examples. The diagonal is perfect calibration.

![Reliability selected]({image_rel["reliability"]})

## Controlled target-set size

{controlled_table}

![Controlled N trends]({image_rel["controlled_n"]})

## Word-level behavior

The calibrated method is not uniformly mediocre: some words are easy across methods while others remain hard. This matters because aggregate calibration can hide per-word failure modes.

![Word accuracy heatmap]({image_rel["word_accuracy"]})
"""

    out_path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None, help="Run directory containing comparison_summary.json")
    parser.add_argument("--output-dir", type=Path, default=FINDINGS_ROOT, help="Directory for the report")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve() if args.run_dir else discover_latest_run()
    comparison, predictions_by_method, checkpoint, controlled = read_run(run_dir)
    rows = score_rows(comparison)

    safe_run_id = slugify("_".join(run_dir.relative_to(RESULTS_ROOT).parts))
    out_dir = args.output_dir.resolve()
    assets_dir = out_dir / f"{safe_run_id}_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{safe_run_id}_compact_report.md"

    set_common_style()
    image_paths = {
        "rank_heatmap": assets_dir / "rank_heatmap.png",
        "pareto": assets_dir / "pareto_ece_auroc.png",
        "confidence_split": assets_dir / "confidence_split.png",
        "reliability": assets_dir / "reliability_selected.png",
        "controlled_n": assets_dir / "controlled_n_trends.png",
        "word_accuracy": assets_dir / "word_accuracy_heatmap.png",
    }

    plot_rank_heatmap(rows, image_paths["rank_heatmap"])
    plot_pareto(comparison, image_paths["pareto"])
    plot_confidence_split(rows, image_paths["confidence_split"])
    plot_reliability_grid(predictions_by_method, comparison, image_paths["reliability"])
    if controlled:
        plot_controlled_n(controlled, image_paths["controlled_n"])
    else:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No controlled_n_summary.json found", ha="center", va="center")
        ax.axis("off")
        fig.savefig(image_paths["controlled_n"], bbox_inches="tight")
        plt.close(fig)
    word_top, word_bottom = plot_word_accuracy(predictions_by_method, image_paths["word_accuracy"])

    build_report(
        run_dir=run_dir,
        comparison=comparison,
        rows=rows,
        checkpoint=checkpoint,
        controlled=controlled,
        image_paths=image_paths,
        word_top=word_top,
        word_bottom=word_bottom,
        out_path=out_path,
    )

    print(out_path)
    for name, path in image_paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
