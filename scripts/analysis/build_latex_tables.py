"""Emit the LaTeX table fragments that paper/acl_latex.tex \\input{}s.

Every number in the paper's tables comes from this script reading the raw
results JSONs; nothing is hand-copied. Run with --write to (re)generate the
fragments under paper/, or with no args to print them to stdout.

Fragments:
  scorecard-main-tbl.tex        main four-oracle scorecard (ECE/AUROC +- SE)
  full-scorecard-qwen-tbl.tex   all method rows, Qwen pair, 5 metrics
  full-scorecard-gemma-tbl.tex  all method rows, Gemma pair, 5 metrics
  tsweep-tbl.tex                bootstrap temperature sweep, four oracles
  k-ablation-tbl.tex            bootstrap sample-count ablation, four oracles
  perword-tbl.tex               per-word accuracy at bootstrap T=1.0
  postcal-{model}-tbl.tex       post-hoc calibration, one per oracle
  bootstrap-cis-tbl.tex         bootstrap 95% CIs, four oracles
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

PAPER = Path("paper")

MODELS = {
    "qwen3-8b": {
        "display": "Qwen3-8B",
        "dirs": ["results/qwen3-8b/run_3", "results/qwen3-8b/forced_choice_backfill"],
        "boot_star": "bootstrap_t1p0",
    },
    "qwen3.6-27b": {
        "display": "Qwen3.6-27B",
        "dirs": [
            "results/qwen3.6-27b/taboo_uq",
            "results/qwen3.6-27b/forced_choice_backfill",
            "results/qwen3.6-27b/extra_temps_backfill",
        ],
        "boot_star": "bootstrap_t1p5",
    },
    "gemma-2-9b": {
        "display": "Gemma-2-9B",
        "dirs": [
            "results/gemma-2-9b/taboo_uq",
            "results/gemma-2-9b/forced_choice_backfill",
            "results/gemma-2-9b/extra_temps_backfill",
        ],
        "boot_star": "bootstrap_t1p5",
    },
    "gemma-3-27b": {
        "display": "Gemma-3-27B",
        "dirs": ["results/gemma-3-27b/taboo_uq", "results/gemma-3-27b/forced_choice_backfill"],
        "boot_star": "bootstrap_t1p3",
    },
}

METHOD_DISPLAY = {
    "forced_choice": "Forced choice (M7)",
    "logprob_offset": "Log-prob (with offset)",
    "logprob_no_offset": "Log-prob (no offset)",
    "bootstrap_t0p3": r"Bootstrap $T{=}0.3$",
    "bootstrap_t0p5": r"Bootstrap $T{=}0.5$",
    "bootstrap_t0p7": r"Bootstrap $T{=}0.7$",
    "bootstrap_t1p0": r"Bootstrap $T{=}1.0$",
    "bootstrap_t1p3": r"Bootstrap $T{=}1.3$",
    "bootstrap_t1p5": r"Bootstrap $T{=}1.5$",
    "bootstrap_t1p75": r"Bootstrap $T{=}1.75$",
    "bootstrap_t2p0": r"Bootstrap $T{=}2.0$",
    "direct": "Direct (numeric)",
    "direct_linguistic_expected": "Constrained label (expected)",
    "direct_linguistic_p_high_plus": r"Constrained label $P(\mathrm{high}{+})$",
    "direct_linguistic_p_very_high": r"Constrained label $P(\mathrm{very\ high})$",
    "mcmc_t0p125": r"MCMC accept $T{=}0.125$",
    "mcmc_t0p25": r"MCMC accept $T{=}0.25$",
    "mcmc_t0p5": r"MCMC accept $T{=}0.5$",
    "mcmc_agreement_t0p125": r"MCMC agreement $T{=}0.125$",
    "mcmc_agreement_t0p25": r"MCMC agreement $T{=}0.25$",
    "mcmc_agreement_t0p5": r"MCMC agreement $T{=}0.5$",
    "sensitivity": "Steering sensitivity",
}

FULL_ORDER = [
    "forced_choice",
    "logprob_no_offset",
    "logprob_offset",
    "bootstrap_t0p3",
    "bootstrap_t0p5",
    "bootstrap_t0p7",
    "bootstrap_t1p0",
    "bootstrap_t1p3",
    "bootstrap_t1p5",
    "bootstrap_t1p75",
    "bootstrap_t2p0",
    "direct_linguistic_expected",
    "direct_linguistic_p_high_plus",
    "direct_linguistic_p_very_high",
    "mcmc_agreement_t0p125",
    "mcmc_agreement_t0p25",
    "mcmc_agreement_t0p5",
    "sensitivity",
    "mcmc_t0p125",
    "mcmc_t0p25",
    "mcmc_t0p5",
    "direct",
]

MAIN_ORDER = [
    "forced_choice",
    "logprob_offset",
    "bootstrap_t1p0",
    "bootstrap_t1p5",
    "direct_linguistic_p_very_high",
    "mcmc_agreement_t0p5",
    "sensitivity",
    "mcmc_t0p125",
    "direct",
]


def load_summaries(model: str) -> dict:
    out = {}
    for d in MODELS[model]["dirs"]:
        fp = Path(d) / "comparison_summary.json"
        if fp.exists():
            out.update(json.load(open(fp)))
    return out


def load_cis(model: str) -> dict:
    return json.load(open(f"results/bootstrap_cis/{model}.json"))


def fmt(x: float, nd: int = 3) -> str:
    s = f"{x:.{nd}f}"
    return s[1:] if s.startswith("0.") else s


def scorecard_main() -> str:
    sums = {m: load_summaries(m) for m in MODELS}
    cis = {m: load_cis(m) for m in MODELS}
    best_ece = {m: min(sums[m][k]["ece"] for k in MAIN_ORDER if k in sums[m]) for m in MODELS}
    best_aur = {m: max(sums[m][k]["auroc"] for k in MAIN_ORDER if k in sums[m]) for m in MODELS}

    def cell(model, key, metric, best):
        v = sums[model][key][metric]
        se = cis[model][key][{"ece": "ece", "auroc": "auroc"}[metric]]["se"]
        body = rf"{fmt(v)}\,\tiny$\pm${fmt(se)}"
        return rf"\textbf{{{body}}}" if abs(v - best) < 5e-4 else body

    out = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize \setlength{\tabcolsep}{1.0pt}",
        r"\begin{tabular}{l rrr rrr rrr rrr}",
        r"\toprule",
        " & "
        + " & ".join(rf"\multicolumn{{3}}{{c}}{{{MODELS[m]['display']}}}" for m in MODELS)
        + r" \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}",
        "Method & " + " & ".join(["Acc & ECE & AUROC"] * 4) + r" \\",
        r"\midrule",
    ]
    short = {
        "logprob_offset": "Log-prob (offset)",
        "direct_linguistic_p_very_high": r"Constr.\ label $P(\mathrm{v.\,high})$",
    }
    for key in MAIN_ORDER:
        cells = []
        for m in MODELS:
            if key not in sums[m]:
                cells += ["--", "--", "--"]
                continue
            s = sums[m][key]
            cells.append(fmt(s["accuracy"]))
            cells.append(cell(m, key, "ece", best_ece[m]))
            cells.append(cell(m, key, "auroc", best_aur[m]))
        name = short.get(key, METHOD_DISPLAY[key])
        out.append(f"{name} & " + " & ".join(cells) + r" \\")
    out += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Main scorecard on all four oracles, $n{=}6{,}000$ samples per row"
        r" per oracle. ECE and AUROC carry $\pm$ one bootstrap standard error"
        r" (1000 resamples). Lower ECE and higher accuracy and AUROC are better;"
        r" best ECE and AUROC per oracle in \textbf{bold}. Forced choice scores the"
        r" 20 candidate words directly and is the only method that changes the answer"
        r" as well as the confidence. The full method grid is in"
        r" \cref{app:full-scorecard}.}",
        r"\label{tab:scorecard}",
        r"\end{table*}",
    ]
    return "\n".join(out)


def full_scorecard(models: list[str], label: str) -> str:
    sums = {m: load_summaries(m) for m in models}
    keys = [k for k in FULL_ORDER if any(k in sums[m] for m in models)]
    best = {
        (m, met): (min if met in ("ece", "brier_score", "nll") else max)(
            sums[m][k][met] for k in keys if k in sums[m]
        )
        for m in models
        for met in ("accuracy", "ece", "brier_score", "nll", "auroc")
    }

    def cell(m, k, met):
        if k not in sums[m]:
            return "--"
        v = sums[m][k][met]
        nd = 3 if met != "nll" else (3 if v < 10 else 2)
        body = f"{v:.{nd}f}"
        return rf"\textbf{{{body}}}" if abs(v - best[(m, met)]) < 5e-4 else body

    disp = " & ".join(rf"\multicolumn{{5}}{{c{'|' if i == 0 else ''}}}{{{MODELS[m]['display']}}}" for i, m in enumerate(models))
    out = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small \setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrrr|rrrrr}",
        r"\toprule",
        f" & {disp} \\\\",
        r"Method & Acc & ECE & Brier & NLL & AUROC & Acc & ECE & Brier & NLL & AUROC \\",
        r"\midrule",
    ]
    for k in keys:
        cells = []
        for m in models:
            for met in ("accuracy", "ece", "brier_score", "nll", "auroc"):
                cells.append(cell(m, k, met))
        out.append(f"{METHOD_DISPLAY[k]} & " + " & ".join(cells) + r" \\")
    pair = " and ".join(MODELS[m]["display"] for m in models)
    out += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{Full method scorecard on {pair}, $n{{=}}6{{,}}000$ samples per"
        r" row. ECE, Brier, and NLL: lower is better. Accuracy and AUROC: higher is"
        r" better. Best per column in \textbf{bold}. `--': configuration not run on"
        r" this oracle (the extended temperatures were run only where the ECE optimum"
        r" was not interior to the original grid).}",
        rf"\label{{{label}}}",
        r"\end{table*}",
    ]
    return "\n".join(out)


def tsweep() -> str:
    sums = {m: load_summaries(m) for m in MODELS}
    temps = ["0p3", "0p5", "0p7", "1p0", "1p3", "1p5", "1p75", "2p0"]
    best = {m: min(sums[m][f"bootstrap_t{t}"]["ece"] for t in temps if f"bootstrap_t{t}" in sums[m]) for m in MODELS}
    out = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small \setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l rr rr rr rr}",
        r"\toprule",
        " & "
        + " & ".join(rf"\multicolumn{{2}}{{c}}{{{MODELS[m]['display']}}}" for m in MODELS)
        + r" \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
        "$T$ & " + " & ".join(["Acc & ECE"] * 4) + r" \\",
        r"\midrule",
    ]
    for t in temps:
        key = f"bootstrap_t{t}"
        cells = []
        for m in MODELS:
            if key not in sums[m]:
                cells += ["--", "--"]
                continue
            s = sums[m][key]
            e = fmt(s["ece"])
            if abs(s["ece"] - best[m]) < 5e-4:
                e = rf"\textbf{{{e}}}"
            cells += [fmt(s["accuracy"]), e]
        out.append(t.replace("p", ".") + " & " + " & ".join(cells) + r" \\")
    out += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Bootstrap temperature sweep, $k{=}20$ samples per item, best ECE"
        r" per oracle in \textbf{bold}. The extended grid ($T{>}1.5$) was run on the"
        r" two oracles whose ECE was still falling at $T{=}1.5$: the optimum is"
        r" interior on Gemma-2-9B ($T{=}1.75$) and still falling at $T{=}2.0$ on"
        r" Qwen3.6-27B.}",
        r"\label{tab:tsweep}",
        r"\end{table*}",
    ]
    return "\n".join(out)


def k_ablation() -> str:
    data = {m: json.load(open(f"results/k_ablation/{m}.json")) for m in MODELS}
    out = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small \setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l rrr rrr rrr rrr}",
        r"\toprule",
        " & "
        + " & ".join(
            rf"\multicolumn{{3}}{{c}}{{{MODELS[m]['display']} ($T{{=}}{MODELS[m]['boot_star'].split('_t')[1].replace('p', '.')}$)}}"
            for m in MODELS
        )
        + r" \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}",
        "$k$ & " + " & ".join(["Acc & ECE & AUROC"] * 4) + r" \\",
        r"\midrule",
    ]
    for k in ["3", "5", "10", "20"]:
        cells = []
        for m in MODELS:
            row = data[m][MODELS[m]["boot_star"]][k]
            cells += [fmt(row["acc"]), fmt(row["ece"]), fmt(row["auroc"])]
        out.append(f"{k} & " + " & ".join(cells) + r" \\")
    out += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Bootstrap sample-count ablation: mode frequency over the first"
        r" $k$ of the 20 saved draws, at the ECE-optimal temperature of the original"
        r" grid for each oracle. Most of the calibration and ranking gain is in place"
        r" by $k{=}10$ (on Gemma-3-27B the ECE minimum sits at $k{=}10$); $k$ is the"
        r" main cost lever of the bootstrap.}",
        r"\label{tab:kablation}",
        r"\end{table*}",
    ]
    return "\n".join(out)


def perword() -> str:
    accs: dict[str, dict[str, float]] = {}
    for m in MODELS:
        root = Path(MODELS[m]["dirs"][0])
        preds = json.load(open(root / "bootstrap_t1p0_results.json"))["predictions"]
        agg = defaultdict(list)
        for p in preds:
            agg[p["target_word"]].append(bool(p["is_correct"]))
        accs[m] = {w: sum(v) / len(v) for w, v in agg.items()}
    words = sorted(accs["qwen3-8b"], key=lambda w: -accs["qwen3-8b"][w])
    out = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize \setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        "Word & " + " & ".join(MODELS[m]["display"] for m in MODELS) + r" \\",
        r"\midrule",
    ]
    for w in words:
        out.append(f"{w} & " + " & ".join(fmt(accs[m][w]) for m in MODELS) + r" \\")
    out += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Per-word accuracy on Bootstrap $T{=}1.0$ for all four oracles,"
        r" sorted by Qwen3-8B accuracy. Per-word accuracy spans an order of magnitude"
        r" on every oracle.}",
        r"\label{tab:perword-bothmodels}",
        r"\end{table}",
    ]
    return "\n".join(out)


POSTCAL_METHODS = [
    "forced_choice",
    "logprob_offset",
    "bootstrap_t0p7",
    "bootstrap_t1p0",
    "bootstrap_t1p5",
    "direct_linguistic_p_very_high",
    "direct",
    "mcmc_t0p125",
    "mcmc_agreement_t0p5",
    "sensitivity",
]


def postcal_table(model: str) -> str:
    d_word = json.load(open(f"results/postcal/{model}/word_disjoint.json"))
    d_rand = json.load(open(f"results/postcal/{model}/random_5050.json"))
    out = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small \setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrrr|rrrrr}",
        r"\toprule",
        r" & \multicolumn{5}{c|}{Word-disjoint split} & \multicolumn{5}{c}{Random 50/50 split} \\",
        r"Method & Uncal & Temp & Platt & Iso & Beta & Uncal & Temp & Platt & Iso & Beta \\",
        r"\midrule",
    ]
    for m in POSTCAL_METHODS:
        if m not in d_word:
            continue
        w, r = d_word[m], d_rand[m]
        cells = [
            fmt(w["uncalibrated"]["ece"]), fmt(w["temperature"]["ece"]),
            fmt(w["platt"]["ece"]), fmt(w["isotonic"]["ece"]), fmt(w["beta"]["ece"]),
            fmt(r["uncalibrated"]["ece"]), fmt(r["temperature"]["ece"]),
            fmt(r["platt"]["ece"]), fmt(r["isotonic"]["ece"]), fmt(r["beta"]["ece"]),
        ]
        out.append(f"{METHOD_DISPLAY[m]} & " + " & ".join(cells) + r" \\")
    out += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{Post-hoc calibration on {MODELS[model]['display']}: test-set ECE"
        r" after fitting each calibrator on the fit slice. \emph{Word-disjoint}: fit"
        r" on 10 of 20 secret words, evaluate on the other 10. \emph{Random 50/50}:"
        r" random sample-level split, identical across methods. Lower is better.}",
        rf"\label{{tab:postcal-{model.replace('.', '-')}}}",
        r"\end{table*}",
    ]
    return "\n".join(out)


def ci_table() -> str:
    def cell(v):
        return rf"{fmt(v['point'])}\,\scriptsize[{fmt(v['lo'])},{fmt(v['hi'])}]"

    out = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small \setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & ECE & Brier & NLL & AUROC \\",
    ]
    for m in MODELS:
        d = load_cis(m)
        out.append(r"\midrule")
        out.append(rf"\multicolumn{{5}}{{l}}{{\textit{{{MODELS[m]['display']}}}}} \\")
        for key in MAIN_ORDER:
            if key not in d:
                continue
            out.append(
                f"{METHOD_DISPLAY[key]} & "
                f"{cell(d[key]['ece'])} & {cell(d[key]['brier'])} & "
                f"{cell(d[key]['nll'])} & {cell(d[key]['auroc'])} \\\\"
            )
    out += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Bootstrap 95\% CIs (1000 resamples) for the main scorecard rows"
        r" on all four oracles. Point estimate followed by the 2.5/97.5 percentiles."
        r" $n{=}6{,}000$ per row.}",
        r"\label{tab:bootstrap-cis}",
        r"\end{table*}",
    ]
    return "\n".join(out)


FRAGMENTS = {
    "scorecard-main-tbl.tex": scorecard_main,
    "full-scorecard-qwen-tbl.tex": lambda: full_scorecard(
        ["qwen3-8b", "qwen3.6-27b"], "tab:full-scorecard"
    ),
    "full-scorecard-gemma-tbl.tex": lambda: full_scorecard(
        ["gemma-2-9b", "gemma-3-27b"], "tab:full-scorecard-gemma"
    ),
    "tsweep-tbl.tex": tsweep,
    "k-ablation-tbl.tex": k_ablation,
    "perword-tbl.tex": perword,
    "postcal-qwen3-8b-tbl.tex": lambda: postcal_table("qwen3-8b"),
    "postcal-qwen3-6-27b-tbl.tex": lambda: postcal_table("qwen3.6-27b"),
    "postcal-gemma-2-9b-tbl.tex": lambda: postcal_table("gemma-2-9b"),
    "postcal-gemma-3-27b-tbl.tex": lambda: postcal_table("gemma-3-27b"),
    "bootstrap-cis-tbl.tex": ci_table,
}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--write":
        for name, build in FRAGMENTS.items():
            (PAPER / name).write_text(build() + "\n")
            print(f"wrote paper/{name}")
    else:
        for name, build in FRAGMENTS.items():
            print(f"% =========== {name} ===========")
            print(build())
            print()
