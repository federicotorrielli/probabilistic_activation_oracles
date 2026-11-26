"""LaTeX tables for the ARR November 2026 revision, built from saved JSON only.

Inputs: results/review_analyses/nov/nov_analyses.json (arr_nov_analyses.py),
results/qwen3-8b/timing.json (pao.experiments.time_methods).
Writes scorecard-main, loo-t, within-word, and cost tables into paper/.
"""

from __future__ import annotations

import json
from pathlib import Path

PAPER = Path("paper")
NOV = json.loads(Path("results/review_analyses/nov/nov_analyses.json").read_text())
TIMING = json.loads(Path("results/qwen3-8b/timing.json").read_text())["seconds_per_sample"]
ORACLES = {"qwen3-8b": "Qwen3-8B", "qwen3.6-27b": "Qwen3.6-27B", "gemma-2-9b": "Gemma-2-9B",
           "gemma-3-27b": "Gemma-3-27B"}
NAMES = {
    "forced_choice": "Forced choice (M5)",
    "logprob_offset": "Log probability (M1)",
    "bootstrap_t1p0": r"Bootstrap $T{=}1.0$ (M2)",
    "bootstrap_t1p3": r"Bootstrap $T{=}1.3$ (M2)",
    "direct_linguistic_p_very_high": r"Constr.\ label $P(\mathrm{v.\,high})$ (M4)",
    "direct": "Direct numeric (M3)",
}


def fmt(x: float) -> str:
    return f"{x:.3f}".removeprefix("0")


def interval(v: float, lo_hi: list[float]) -> str:
    return rf"{fmt(v)}\,\tiny[{fmt(lo_hi[0])},{fmt(lo_hi[1])}]"


def table(env: str, colspec: str, header: list[str], rows: list[str], caption: str, label: str,
          resize: bool = False) -> str:
    body = [rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", *header, r"\midrule", *rows, r"\bottomrule",
            r"\end{tabular}"]
    if resize:
        body = [r"\resizebox{\linewidth}{!}{%", *body, "}"]
    return "\n".join([rf"\begin{{{env}}}[t]", r"\centering", r"\small \setlength{\tabcolsep}{3pt}", *body,
                      rf"\caption{{{caption}}}", rf"\label{{{label}}}", rf"\end{{{env}}}"]) + "\n"


def scorecard() -> str:
    rows = []
    for key, name in NAMES.items():
        cells = []
        for m in ORACLES:
            s = NOV["main_rows"][m][key]
            cells += [fmt(s["accuracy"]), interval(s["ece"], s["ece_ci"]), interval(s["auroc"], s["auroc_ci"])]
        rows.append(f"{name} & " + " & ".join(cells) + r" \\")
    tuned = []
    for m in ORACLES:
        t = NOV["tuned_on_5_words"][m]
        tuned += ["--", interval(t["ece_mean"], t["ece_p5_p95"]), fmt(t["auroc_mean"])]
    rows.append(r"Bootstrap, $T$ tuned on 5 words (M2) & " + " & ".join(tuned) + r" \\")
    header = [" & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{d}}}" for d in ORACLES.values()) + r" \\",
              r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}",
              "Method & " + " & ".join(["Acc & ECE & AUROC"] * 4) + r" \\"]
    caption = (r"Accuracy, ECE, and AUROC of the five methods on all four oracles, with $6{,}000$ samples per row"
               r" and oracle. A lower ECE and a higher accuracy and AUROC are better. Brackets give $95\%$ intervals"
               r" from 1000 resamples of secret words, context prompts, and verbalizer prompts together"
               r" \citep{owen_pigeonhole_2007}. In the last row, the temperature is chosen on 5 labeled secret"
               r" words and the ECE is computed on the other 15. The row gives the mean over 200 choices of the"
               r" labeled words, with the 5th and 95th percentiles. The full grid of methods is in"
               r" \cref{app:full-scorecard}.")
    return table("table*", "l rrr rrr rrr rrr", header, rows, caption, "tab:scorecard", resize=True)


def loo() -> str:
    loo = NOV["loo_temperature"]
    rows = [
        "Chosen $T$ & " + " & ".join(f"{loo[m]['t']:.1f}" for m in ORACLES) + r" \\",
        "ECE at the chosen $T$ & " + " & ".join(fmt(loo[m]["test_ece"]) for m in ORACLES) + r" \\",
        r"ECE at $T{=}1.0$ & " + " & ".join(fmt(loo[m]["t1_ece"]) for m in ORACLES) + r" \\",
    ]
    header = [" & " + " & ".join(ORACLES.values()) + r" \\"]
    caption = (r"Temperature of the bootstrap without labels on the oracle under test. For each oracle, $T$ is the"
               r" temperature of the grid with the smallest maximum ECE over the other three oracles.")
    return table("table", "l rrrr", header, rows, caption, "tab:loo-t", resize=True)


def within_word() -> str:
    rows = [f"{name} & " + " & ".join(fmt(NOV["main_rows"][m][key]["within_word_ece"]) for m in ORACLES) + r" \\"
            for key, name in NAMES.items()]
    header = [" & " + " & ".join(ORACLES.values()) + r" \\"]
    caption = (r"ECE computed inside each secret word and averaged over the 20 words. This value does not include"
               r" the differences in difficulty between words.")
    return table("table", "l rrrr", header, rows, caption, "tab:within-word", resize=True)


def cost() -> str:
    spec = [
        ("Log probability (M1)", "1 greedy answer", "log_probability"),
        ("Bootstrap, $k{=}20$ (M2)", "20 sampled answers, one batch", "bootstrap_k20"),
        ("Direct numeric (M3)", "2 greedy answers", "direct_numeric"),
        ("Constr.\\ label (M4)", "M3 answer + 5 scored labels", "label_scoring"),
        ("Forced choice (M5)", "20 words scored, 2 spellings each", "forced_choice"),
    ]
    rows = [f"{name} & {work} & {TIMING[key]:.2f}" + r" \\" for name, work, key in spec]
    header = [r"Method & Work per sample & Seconds \\"]
    caption = (r"Cost of each method on Qwen3-8B, on one B200 GPU. Seconds per sample are the mean over 49 samples."
               r" Constrained label scoring reuses the answer of M3, so its time adds to the time of M3. Forced"
               r" choice scores the candidates one at a time in our code.")
    return table("table", "l l r", header, rows, caption, "tab:cost", resize=True)


def main() -> None:
    for name, build in {"scorecard-main-tbl.tex": scorecard, "loo-t-tbl.tex": loo,
                        "within-word-tbl.tex": within_word, "cost-tbl.tex": cost}.items():
        (PAPER / name).write_text(build())
        print("wrote", PAPER / name)


if __name__ == "__main__":
    main()
