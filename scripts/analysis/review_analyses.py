"""Analyses added for the ARR resubmission.

1. Resampling that respects the data structure: whole secret words, and
   words x context prompts x verbalizer prompts together (pigeonhole
   bootstrap, Owen 2007).
2. Accuracy when only the most confident answers are kept (coverage).
3. Where the free-text errors come from, and how many forced choice fixes.
4. The bootstrap temperature rule, with T chosen on a few labeled secret words.
5. The failure types of Jakkli et al. (2026) inside the free-text errors.

Writes results/review_analyses/*.json, four LaTeX tables and one figure in
the paper directory.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

PAPER = Path("paper/Probabilistic_Activation_Oracles__EMNLP_2026_")
OUT = Path("results/review_analyses")
B = 1000

ORACLES = {
    "qwen3-8b": ("Qwen3-8B", ["results/qwen3-8b/run_3", "results/qwen3-8b/forced_choice_backfill"], "bootstrap_t1p0"),
    "qwen3.6-27b": (
        "Qwen3.6-27B",
        [
            "results/qwen3.6-27b/taboo_uq",
            "results/qwen3.6-27b/forced_choice_backfill",
            "results/qwen3.6-27b/extra_temps_backfill",
        ],
        "bootstrap_t2p0",
    ),
    "gemma-2-9b": (
        "Gemma-2-9B",
        [
            "results/gemma-2-9b/taboo_uq",
            "results/gemma-2-9b/forced_choice_backfill",
            "results/gemma-2-9b/extra_temps_backfill",
        ],
        "bootstrap_t1p75",
    ),
    "gemma-3-27b": (
        "Gemma-3-27B",
        ["results/gemma-3-27b/taboo_uq", "results/gemma-3-27b/forced_choice_backfill"],
        "bootstrap_t1p3",
    ),
}

# (key, display, color); "boot" is resolved to the tuned temperature per oracle.
BLUE, GREEN, MAGENTA, YELLOW, PURPLE = "#2a78d6", "#008300", "#e87ba4", "#eda100", "#7b4fb3"
INK, SLATE, MUTED, HAIR = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
METHODS = [
    ("forced_choice", "Forced choice", BLUE),
    ("boot", "Bootstrap (tuned $T$)", GREEN),
    ("logprob_offset", "Log probability", MAGENTA),
    ("direct_linguistic_p_very_high", "Constrained label scoring", PURPLE),
    ("direct", "Direct (numeric)", YELLOW),
]

# (better, worse, metric): "better" should win on this metric.
ORDERINGS = [
    ("forced_choice", "logprob_offset", "auroc", "Forced choice $>$ log probability"),
    ("forced_choice", "boot", "auroc", "Forced choice $>$ bootstrap"),
    ("logprob_offset", "direct", "auroc", "Log probability $>$ direct"),
    ("boot", "direct", "auroc", "Bootstrap $>$ direct"),
    ("direct_linguistic_p_very_high", "direct", "auroc", "Constrained label $>$ direct"),
    ("boot", "direct", "ece", "Bootstrap $<$ direct"),
    ("forced_choice", "direct", "ece", "Forced choice $<$ direct"),
]


def find(model: str, key: str) -> Path:
    for d in ORACLES[model][1]:
        fp = Path(d) / f"{key}_results.json"
        if fp.exists():
            return fp
    raise FileNotFoundError(f"{model}/{key}")


def load_grid(model: str, key: str):
    """Records of one method arranged as a word x context x verbalizer grid."""
    preds = json.load(open(find(model, key)))["predictions"]
    words = sorted({p["target_word"] for p in preds})
    ctxs = sorted({p["context_prompt"] for p in preds})
    verbs = sorted({p["verbalizer_prompt"] for p in preds})
    conf = np.full((len(words), len(ctxs), len(verbs)), np.nan)
    corr = np.zeros_like(conf, dtype=bool)
    recs = np.empty(conf.shape, dtype=object)
    for p in preds:
        i = words.index(p["target_word"]), ctxs.index(p["context_prompt"]), verbs.index(p["verbalizer_prompt"])
        conf[i] = float(p["confidence"])
        c = p["is_correct"]
        corr[i] = c if isinstance(c, bool) else str(c).lower() == "true"
        recs[i] = p
    assert not np.isnan(conf).any(), f"{model}/{key} has missing samples"
    return conf, corr, recs


def ece(conf, corr, bins=10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, bins - 1)
    n = np.bincount(idx, minlength=bins)
    gap = np.abs(np.bincount(idx, corr, bins) - np.bincount(idx, conf, bins))
    return gap.sum() / len(conf) if n.sum() else 0.0


def auroc(conf, corr):
    pos = corr.sum()
    neg = len(corr) - pos
    if pos == 0 or neg == 0:
        return 0.5
    return (rankdata(conf)[corr].sum() - pos * (pos + 1) / 2) / (pos * neg)


METRIC = {"ece": ece, "auroc": auroc}


def resample_index(shape, scheme, rng):
    w, c, v = shape
    iw = rng.integers(0, w, w)
    if scheme == "word":
        return np.ix_(iw, np.arange(c), np.arange(v))
    return np.ix_(iw, rng.integers(0, c, c), rng.integers(0, v, v))


def resampling(model: str, data: dict) -> dict:
    out = {}
    for scheme in ("word", "crossed"):
        rng = np.random.default_rng(0)
        draws = {k: {"ece": [], "auroc": []} for k in data}
        for _ in range(B):
            idx = resample_index(data["direct"][0].shape, scheme, rng)
            for k, (conf, corr, _) in data.items():
                cf, cr = conf[idx].ravel(), corr[idx].ravel()
                draws[k]["ece"].append(ece(cf, cr))
                draws[k]["auroc"].append(auroc(cf, cr))
        draws = {k: {m: np.array(v) for m, v in d.items()} for k, d in draws.items()}
        share = {}
        for better, worse, met, _ in ORDERINGS:
            a, b = draws[better][met], draws[worse][met]
            share[f"{better}>{worse}:{met}"] = float(np.mean(a > b if met == "auroc" else a < b))
        cis = {
            k: {m: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for m, v in d.items()}
            for k, d in draws.items()
        }
        out[scheme] = {"share": share, "ci": cis}
    return out


def coverage_curve(conf, corr, rng):
    order = np.lexsort((rng.random(len(conf)), -conf))  # equal confidences in random order
    hits = np.cumsum(corr[order])
    return hits / np.arange(1, len(conf) + 1)


def errors(data: dict, boot_t1: tuple) -> dict:
    _, lp_corr, lp_recs = data["logprob_offset"]
    _, fc_corr, _ = data["forced_choice"]
    lp_recs, lp_corr, fc_corr = lp_recs.ravel(), lp_corr.ravel(), fc_corr.ravel()
    wrong = ~lp_corr
    off_list = np.array([r["method_metadata"]["extracted_word_source"] != "answer_vocab" for r in lp_recs])
    never = np.array([
        r["target_word"].lower() not in r["method_metadata"]["normalized_samples"] for r in boot_t1[2].ravel()
    ])
    return {
        "n_wrong": int(wrong.sum()),
        "off_list_share_of_wrong": float((wrong & off_list).sum() / wrong.sum()),
        "fc_right_on_off_list": float(fc_corr[wrong & off_list].mean()),
        "fc_wrong_where_lp_right": int((lp_corr & ~fc_corr).sum()),
        "never_sampled_share": float(never.mean()),
        "fc_right_on_never_sampled": float(fc_corr.ravel()[never].mean()),
    }


def pct(x: float, nd: int = 0) -> str:
    return f"{100 * x:.{nd}f}"


def resampling_table(res: dict) -> str:
    SHORT = ["Q-8B", "Q-27B", "G-9B", "G-27B"]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small \setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{ll rrrr rrrr}",
        r"\toprule",
        r" & & \multicolumn{4}{c}{Resample secret words} & \multicolumn{4}{c}{Resample words, contexts, questions} \\",
        r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
        "Metric & Ordering & " + " & ".join(SHORT * 2) + r" \\",
        r"\midrule",
    ]
    for better, worse, met, label in ORDERINGS:
        k = f"{better}>{worse}:{met}"
        cells = [pct(res[m][s]["share"][k], 1) for s in ("word", "crossed") for m in ORACLES]
        lines.append(f"{met.upper()} & {label} & " + " & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Share of 1000 resamples (\%) in which each comparison gives the same result."
        r" Left: we resample whole secret words, so that the 300 samples of a word are kept together."
        r" Right: we resample secret words, context prompts, and verbalizer prompts together"
        r" \citep{owen_pigeonhole_2007}. The bootstrap is at the tuned temperature of each oracle."
        r" Constrained label scoring uses the $P(\mathrm{very\ high})$ readout. Q-8B: Qwen3-8B,"
        r" Q-27B: Qwen3.6-27B, G-9B: Gemma-2-9B, G-27B: Gemma-3-27B.}",
        r"\label{tab:resampling}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


def errors_table(errs: dict) -> str:
    rows = [
        ("Incorrect answers in free text that name none of the 20 words", "off_list_share_of_wrong"),
        ("\\quad among these, forced choice finds the secret word", "fc_right_on_off_list"),
        ("Samples where no bootstrap answer names the secret word", "never_sampled_share"),
        ("\\quad among these, forced choice finds the secret word", "fc_right_on_never_sampled"),
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small \setlength{\tabcolsep}{3pt}",
        r"\begin{tabularx}{\linewidth}{X rrrr}",
        r"\toprule",
        " & " + " & ".join(rf"\rotatebox{{70}}{{{ORACLES[m][0]}}}" for m in ORACLES) + r" \\",
        r"\midrule",
    ]
    for label, key in rows:
        lines.append(f"{label} & " + " & ".join(pct(errs[m][key]) for m in ORACLES) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabularx}",
        r"\caption{Where the forced choice gain comes from (\%). Top: greedy answers in free text"
        r" (M1). Bottom: the twenty bootstrap answers at $T{=}1.0$ (M2).}",
        r"\label{tab:errors}",
        r"\end{table}",
    ]
    return "\n".join(lines)


FIT_SIZES = (1, 5, 10)
SPLITS = 200


def temperature_selection(model: str) -> dict:
    """Pick T where mean agreement matches accuracy on fit words; ECE on the other words."""
    keys = sorted({fp.stem.removesuffix("_results") for d in ORACLES[model][1]
                   for fp in Path(d).glob("bootstrap_t*_results.json")})
    grids = {k: load_grid(model, k)[:2] for k in keys}
    n_words = next(iter(grids.values()))[0].shape[0]
    rng = np.random.default_rng(0)
    out = {"t1": [], **{n: [] for n in FIT_SIZES}}
    for _ in range(SPLITS):
        perm = rng.permutation(n_words)
        for n in FIT_SIZES:
            fit, test = perm[:n], perm[n:]
            gap = {k: abs(c[fit].mean() - r[fit].mean()) for k, (c, r) in grids.items()}
            c, r = grids[min(gap, key=gap.get)]
            out[n].append(ece(c[test].ravel(), r[test].ravel()))
        c, r = grids["bootstrap_t1p0"]
        out["t1"].append(ece(c[perm[10:]].ravel(), r[perm[10:]].ravel()))
    best = min(keys, key=lambda k: ece(grids[k][0].ravel(), grids[k][1].ravel()))
    return {
        "t1": float(np.mean(out["t1"])),
        **{f"fit{n}": float(np.mean(out[n])) for n in FIT_SIZES},
        "best": float(ece(grids[best][0].ravel(), grids[best][1].ravel())),
        "best_t": float(best.removeprefix("bootstrap_t").replace("p", ".")),
    }


def fmt3(x: float) -> str:
    return f"{x:.3f}".removeprefix("0")


def tsel_table(tsel: dict) -> str:
    cols = ["t1"] + [f"fit{n}" for n in FIT_SIZES] + ["best"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small \setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l rrrrr}",
        r"\toprule",
        r" & No labels & \multicolumn{3}{c}{Labeled words} & Best \\",
        r"\cmidrule(lr){3-5}",
        r"Oracle & $T{=}1.0$ & 1 & 5 & 10 & $T$ \\",
        r"\midrule",
    ]
    for m in ORACLES:
        lines.append(ORACLES[m][0] + " & " + " & ".join(fmt3(tsel[m][c]) for c in cols) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{ECE of the bootstrap when the temperature is chosen with the rule of"
        r" \cref{sec:results-temperature} on 1, 5, or 10 labeled secret words, computed on the"
        r" other words and averaged over 200 random choices of the labeled words. \emph{No labels}:"
        r" the default $T{=}1.0$. \emph{Best $T$}: the temperature of the grid with the lowest ECE"
        r" on all 20 words.}",
        r"\label{tab:tsel}",
        r"\end{table}",
    ]
    return "\n".join(lines)


GROUPS = ("context", "question", "other")


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def failure_group(answer: str, rec: dict) -> str:
    """Incorrect answer that copies the oracle question, the context prompt, or neither."""
    a = (re.findall(r"[a-z]+", answer.lower()) or [""])[0]
    if a and a in words(rec["verbalizer_prompt"]):
        return "question"
    if a and a in words(rec["context_prompt"]):
        return "context"
    return "other"


def failures(model: str) -> dict:
    out = {}
    for key, answer in (("logprob_offset", lambda r: r["method_metadata"]["extracted_word"] or ""),
                        ("bootstrap_t1p0", lambda r: r["predicted_answer"] or "")):
        conf, corr, recs = (x.ravel() for x in load_grid(model, key))
        groups = np.array([failure_group(answer(r), r) if not c else "correct" for r, c in zip(recs, corr)])
        res = {}
        for g in GROUPS:
            mask = corr | (groups == g)
            res[g] = {"share": float((groups == g).sum() / (~corr).sum()), "auroc": float(auroc(conf[mask], corr[mask]))}
        out[key] = res
    return out


def failures_table(fails: dict) -> str:
    names = {"context": "Copies the context prompt", "question": "Copies the oracle question", "other": "Other"}
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small \setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l rrr rrr rrr rrr}",
        r"\toprule",
        " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{ORACLES[m][0]}}}" for m in ORACLES) + r" \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}",
        "Incorrect answer & " + " & ".join([r"\% & LP & BS"] * 4) + r" \\",
        r"\midrule",
    ]
    for g in GROUPS:
        cells = []
        for m in ORACLES:
            f = fails[m]
            cells += [pct(f["logprob_offset"][g]["share"]), fmt3(f["logprob_offset"][g]["auroc"]),
                      fmt3(f["bootstrap_t1p0"][g]["auroc"])]
        lines.append(f"{names[g]} & " + " & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{The failure types of \citet{jakkli_current_2026} inside the incorrect answers in free"
        r" text. \%: share of the incorrect greedy answers of log probability (M1) in each type. LP and BS:"
        r" AUROC of log probability and of the bootstrap at $T{=}1.0$ (M2) when the correct answers are"
        r" compared only with the incorrect answers of this type. An answer copies the context prompt when"
        r" its word is in the context prompt but not in the question of the oracle.}",
        r"\label{tab:failures}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


def coverage_figure(curves: dict):
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 15, "axes.titlesize": 17,
        "axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13,
        "legend.fontsize": 14, "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.9,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": SLATE,
        "text.color": INK, "savefig.dpi": 200,
    })
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6), sharey=True)
    for ax, model in zip(axes, ORACLES):
        ax.grid(True, color=HAIR, linewidth=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for key, label, color in METHODS:
            acc = curves[model][key]
            cov = np.arange(1, len(acc) + 1) / len(acc)
            keep = cov >= 0.05  # the last few answers are too few to read
            ax.plot(cov[keep], acc[keep], color=color, linewidth=2.4, label=label)
        ax.set_title(ORACLES[model][0], color=INK)
        ax.set_xlim(1.0, 0.05)
        ax.set_ylim(0, 1.02)
        ax.set_xticks([1.0, 0.75, 0.5, 0.25], ["100%", "75%", "50%", "25%"])
    axes[0].set_ylabel("Accuracy of kept answers")
    fig.supxlabel("Answers kept, most confident first", color=SLATE, fontsize=15, y=0.1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(PAPER / "figs" / f"coverage_four.{ext}", bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res, errs, curves, tsel, fails = {}, {}, {}, {}, {}
    for model, (_, _, boot) in ORACLES.items():
        data = {k: load_grid(model, boot if k == "boot" else k) for k, _, _ in METHODS}
        res[model] = resampling(model, data)
        errs[model] = errors(data, load_grid(model, "bootstrap_t1p0"))
        tsel[model] = temperature_selection(model)
        fails[model] = failures(model)
        rng = np.random.default_rng(0)
        curves[model] = {k: coverage_curve(c.ravel(), r.ravel(), rng) for k, (c, r, _) in data.items()}
        at = {k: {f"{q}": float(v[int(q / 100 * len(v)) - 1]) for q in (100, 50, 25, 10)} for k, v in curves[model].items()}
        json.dump({"resampling": res[model], "errors": errs[model], "accuracy_at_coverage": at,
                   "temperature_selection": tsel[model], "failures": fails[model]},
                  open(OUT / f"{model}.json", "w"), indent=2)
        print(model, json.dumps(tsel[model]), json.dumps(fails[model]))
    (PAPER / "resampling-tbl.tex").write_text(resampling_table(res) + "\n")
    (PAPER / "errors-tbl.tex").write_text(errors_table(errs) + "\n")
    (PAPER / "tsel-tbl.tex").write_text(tsel_table(tsel) + "\n")
    (PAPER / "failures-tbl.tex").write_text(failures_table(fails) + "\n")
    coverage_figure(curves)


if __name__ == "__main__":
    main()
