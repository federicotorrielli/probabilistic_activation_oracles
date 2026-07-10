"""Emit LaTeX-ready snippets for the appendix tables.

Outputs to stdout:
  * Post-hoc calibration table (ECE pre/post) for both splits, both models
  * Bootstrap CI annotations for the main scorecard rows
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("results")

METHOD_DISPLAY = {
    "logprob_offset": "Log-prob (offset)",
    "logprob_no_offset": "Log-prob (no offset)",
    "bootstrap_t0p3": r"Bootstrap $T{=}0.3$",
    "bootstrap_t0p5": r"Bootstrap $T{=}0.5$",
    "bootstrap_t0p7": r"Bootstrap $T{=}0.7$",
    "bootstrap_t1p0": r"Bootstrap $T{=}1.0$",
    "bootstrap_t1p3": r"Bootstrap $T{=}1.3$",
    "bootstrap_t1p5": r"Bootstrap $T{=}1.5$",
    "direct": "Direct (numeric)",
    "mcmc_t0p125": r"MCMC accept $T{=}0.125$",
    "mcmc_t0p25": r"MCMC accept $T{=}0.25$",
    "mcmc_t0p5": r"MCMC accept $T{=}0.5$",
    "mcmc_agreement_t0p125": r"MCMC agreement $T{=}0.125$",
    "mcmc_agreement_t0p25": r"MCMC agreement $T{=}0.25$",
    "mcmc_agreement_t0p5": r"MCMC agreement $T{=}0.5$",
    "sensitivity": "Steering sensitivity",
}


def calibration_table():
    """Build a LaTeX table: rows = methods, columns = uncal/temp/platt/iso/beta per split per model."""
    methods_to_show = [
        "logprob_offset",
        "bootstrap_t0p7",
        "bootstrap_t1p0",
        "bootstrap_t1p5",
        "direct",
        "mcmc_t0p125",
        "mcmc_agreement_t0p5",
        "sensitivity",
    ]
    # Two tables, one per model. Columns: uncal, temp, platt, iso, beta on word_disjoint, then random_5050.
    out = []
    for model in ["qwen3-8b", "qwen3.6-27b"]:
        d_word = json.load(open(f"results/postcal/{model}/word_disjoint.json"))
        d_rand = json.load(open(f"results/postcal/{model}/random_5050.json"))
        out.append(rf"\begin{{table*}}[t]")
        out.append(r"\centering")
        out.append(r"\small \setlength{\tabcolsep}{4pt}")
        out.append(r"\begin{tabular}{lrrrrr|rrrrr}")
        out.append(r"\toprule")
        out.append(
            r" & \multicolumn{5}{c|}{Word-disjoint split} & \multicolumn{5}{c}{Random 50/50 split} \\"
        )
        out.append(
            r"Method & Uncal & Temp & Platt & Iso & Beta & Uncal & Temp & Platt & Iso & Beta \\"
        )
        out.append(r"\midrule")
        for m in methods_to_show:
            if m not in d_word:
                continue
            w = d_word[m]
            r = d_rand[m]
            out.append(
                f"{METHOD_DISPLAY[m]:30s} & "
                f"{w['uncalibrated']['ece']:.3f} & {w['temperature']['ece']:.3f} & "
                f"{w['platt']['ece']:.3f} & {w['isotonic']['ece']:.3f} & {w['beta']['ece']:.3f} & "
                f"{r['uncalibrated']['ece']:.3f} & {r['temperature']['ece']:.3f} & "
                f"{r['platt']['ece']:.3f} & {r['isotonic']['ece']:.3f} & {r['beta']['ece']:.3f} \\\\"
            )
        out.append(r"\bottomrule")
        out.append(r"\end{tabular}")
        model_caption = {
            "qwen3-8b": "Qwen3-8B",
            "qwen3.6-27b": "Qwen3.6-27B",
        }[model]
        out.append(
            rf"\caption{{Post-hoc calibration on {model_caption}: ECE on the test slice after fitting "
            rf"each calibrator on the fit slice. \emph{{Word-disjoint}}: fit on 10 of 20 secret words, "
            rf"evaluate on the other 10. \emph{{Random 50/50}}: random sample-level split. "
            rf"Lower is better.}}"
        )
        out.append(rf"\label{{tab:postcal-{model.replace('.', '-')}}}")
        out.append(r"\end{table*}")
        out.append("")
    return "\n".join(out)


def ci_table():
    """Bootstrap CIs stacked vertically: models as row blocks, 4 metric columns."""
    methods_to_show = [
        "logprob_offset",
        "bootstrap_t0p5",
        "bootstrap_t0p7",
        "bootstrap_t1p0",
        "bootstrap_t1p3",
        "bootstrap_t1p5",
        "mcmc_agreement_t0p5",
        "sensitivity",
        "mcmc_t0p125",
        "direct",
    ]
    d8 = json.load(open("results/bootstrap_cis/qwen3-8b.json"))
    d27 = json.load(open("results/bootstrap_cis/qwen3.6-27b.json"))

    def cell(v):
        return rf"{v['point']:.3f}\,\scriptsize[{v['lo']:.3f},{v['hi']:.3f}]"

    out = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small \setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & ECE & Brier & NLL & AUROC \\",
        r"\midrule",
        r"\multicolumn{5}{l}{\textit{Qwen3-8B}} \\",
    ]
    for m in methods_to_show:
        if m not in d8:
            continue
        out.append(
            f"{METHOD_DISPLAY[m]:30s} & "
            f"{cell(d8[m]['ece'])} & {cell(d8[m]['brier'])} & "
            f"{cell(d8[m]['nll'])} & {cell(d8[m]['auroc'])} \\\\"
        )
    out.append(r"\midrule")
    out.append(r"\multicolumn{5}{l}{\textit{Qwen3.6-27B}} \\")
    for m in methods_to_show:
        if m not in d27:
            continue
        out.append(
            f"{METHOD_DISPLAY[m]:30s} & "
            f"{cell(d27[m]['ece'])} & {cell(d27[m]['brier'])} & "
            f"{cell(d27[m]['nll'])} & {cell(d27[m]['auroc'])} \\\\"
        )
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(
        r"\caption{Bootstrap 95\% CIs (1000 resamples) for the headline scorecard "
        r"rows. Point estimate followed by the 2.5/97.5 percentile. $n{=}6{,}000$ "
        r"per row.}"
    )
    out.append(r"\label{tab:bootstrap-cis}")
    out.append(r"\end{table*}")
    return "\n".join(out)


def calibration_table_one(model: str) -> str:
    methods_to_show = [
        "logprob_offset",
        "bootstrap_t0p7",
        "bootstrap_t1p0",
        "bootstrap_t1p5",
        "direct",
        "mcmc_t0p125",
        "mcmc_agreement_t0p5",
        "sensitivity",
    ]
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
    for m in methods_to_show:
        if m not in d_word:
            continue
        w = d_word[m]
        r = d_rand[m]
        out.append(
            f"{METHOD_DISPLAY[m]:30s} & "
            f"{w['uncalibrated']['ece']:.3f} & {w['temperature']['ece']:.3f} & "
            f"{w['platt']['ece']:.3f} & {w['isotonic']['ece']:.3f} & {w['beta']['ece']:.3f} & "
            f"{r['uncalibrated']['ece']:.3f} & {r['temperature']['ece']:.3f} & "
            f"{r['platt']['ece']:.3f} & {r['isotonic']['ece']:.3f} & {r['beta']['ece']:.3f} \\\\"
        )
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    model_caption = {"qwen3-8b": "Qwen3-8B", "qwen3.6-27b": "Qwen3.6-27B"}[model]
    out.append(
        rf"\caption{{Post-hoc calibration on {model_caption}: test-set ECE after fitting "
        rf"each calibrator on the fit slice. \emph{{Word-disjoint}}: fit on 10 of 20 secret words, "
        rf"evaluate on the other 10. \emph{{Random 50/50}}: random sample-level split. "
        rf"Lower is better.}}"
    )
    out.append(rf"\label{{tab:postcal-{model.replace('.', '-')}}}")
    out.append(r"\end{table*}")
    return "\n".join(out)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--write":
        # Write standalone .tex fragments expected by acl_latex.tex \input{}
        out_dir = Path("paper")
        (out_dir / "postcal-qwen3-8b-tbl.tex").write_text(
            calibration_table_one("qwen3-8b") + "\n"
        )
        (out_dir / "postcal-qwen3-6-27b-tbl.tex").write_text(
            calibration_table_one("qwen3.6-27b") + "\n"
        )
        (out_dir / "bootstrap-cis-tbl.tex").write_text(ci_table() + "\n")
        print("wrote 3 .tex fragments under paper/")
    else:
        print("% =========== POST-HOC CALIBRATION TABLES ===========")
        print(calibration_table())
        print()
        print("% =========== BOOTSTRAP CI TABLE ===========")
        print(ci_table())
