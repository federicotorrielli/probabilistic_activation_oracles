# Probabilistic Activation Oracles

An *activation oracle* (Karvonen et al. 2025) is a second LLM trained to read the internal activations of a first LLM and describe them in natural language. The oracle answers — but it does not say how sure it is. This repository benchmarks six methods for attaching a calibrated confidence number to every oracle prediction, on the 20-word "taboo" secret-word task, across multiple base models (Qwen3-8B, Qwen3.6-27B, Gemma-2-9B, Gemma-3-27B, Llama-3.1-8B).

## Headline result

On Qwen3-8B at full 6000-sample completion (`results/qwen3-8b/run_3/`), the cleanest confidence signal is **temperature bootstrap mode-frequency at $T{=}1.0$**: ECE 5.7%, Brier 0.163, NLL 0.498, AUROC 0.829. The strongest discriminator (highest AUROC) is the log-prob baseline at 0.840, but it is severely underconfident in absolute terms. Direct numeric self-report ("on a scale of 0 to 100, how sure are you?") is near-uninformative (ECE 58.2%, AUROC 0.516). Raw MCMC acceptance ratio is similarly overconfident (ECE 55.1% at its worst temperature). The MCMC-agreement variant recovers some ranking power but is dominated by bootstrap at matched cost. See `findings/qwen3-8b_run_3_compact_report.md` for the full scorecard.

## Install

Python >=3.13, managed with [uv](https://docs.astral.sh/uv/). Torch is pinned to `2.11.0` against the `pytorch-cu130` index; the flash-attn 2 wheel is hard-coded for `cu130 / torch2.11 / cp313`.

```bash
git clone --recurse-submodules <repo-url>
cd probabilistic_activation_oracles
uv sync --extra flash-attn --reinstall
```

`--extra flash-attn` is mandatory — the default `uv sync` silently omits it and every GPU run breaks. `--reinstall` keeps it consistent after pulls.

For Qwen3.6-27B inference, stay on **flash-attn 2**. flash-attn-4 `4.0.0b12` silently corrupts activations on the 27B forward path at seq>~170 (8B is unaffected). See `findings/attention_backends_2026-05-11.md`.

## Quickstart

Run the main experiment on a preset (each preset bundles a base model + verbalizer LoRA + target-LoRA template):

```bash
uv run python -m pao.experiments.run_taboo_uq \
  --preset qwen3-8b \
  --max-prompts 5            # limit context prompts for fast iteration
```

Output defaults to `results/<preset>/taboo_uq/`. Runs auto-resume from `<output-dir>/checkpoint.json`; a config hash over `CODE_VERSION` and every hyperparameter prevents silent resumes across incompatible code. After a completed run, rename `taboo_uq/` to `run_N/` before launching the next one, or you will resume into it.

Available presets (`pao/config.py`): `qwen3-8b`, `qwen3-32b`, `qwen3.6-27b`, `gemma-2-9b`, `gemma-3-27b`, `llama-3.1-8b`.

Other entry points:

```bash
uv run python -m pao.experiments.layer_sweep --preset qwen3.6-27b
uv run python -m pao.experiments.preflight_taboo --preset qwen3-8b
uv run python -m pao.experiments.mini_direct_variants --preset qwen3-8b

uv run python scripts/analysis/bootstrap_cis.py
uv run python scripts/analysis/post_hoc_calibration.py
uv run python scripts/analysis/build_latex_tables.py
uv run python scripts/figures/layer_sweep.py
```

## The six methods

| # | method                         | confidence signal                          | cost (gens/example) |
| - | ------------------------------ | ------------------------------------------ | ------------------: |
| 1 | log-prob baseline              | joint prob. of extracted answer            |                   1 |
| 2 | temperature bootstrap          | mode frequency over $k{=}20$ samples       |                  20 |
| 3 | direct elicitation             | parsed numeric self-report                 |                   2 |
| 4 | single MCMC power sampling     | raw MH acceptance ratio                    |                ~B·S |
| 5 | MCMC agreement                 | mode frequency over $k{=}10$ chains        |              ~10·BS |
| 6 | steering-coefficient sweep     | mode frequency over 5 dial settings        |                   5 |

Methods 4–6 are introduced in this work. Each method is a function returning a `WordPrediction(predicted_word, confidence, metadata)` triple over a shared interface (`pao/calibration/secret_word_calibration.py`); implementations live in `pao/methods/`.

### The taboo protocol

We use 20 LoRA-finetuned variants of a base model, each with a different secret word (ship, wave, song, snow, rock, moon, jump, green, flame, flag, dance, cloud, clock, chair, salt, book, blue, gold, leaf, smile). For a context prompt we (a) collect activations from the target LoRA at the preset's readout layer, (b) inject those activations into the oracle at layer 1 via a forward hook that preserves the residual norm, (c) ask the oracle for the secret word using each of the six UQ methods, and (d) score the resulting `(prediction, confidence)` pairs with ECE (10-bin), Brier, AUROC, and NLL. A *controlled-N* variant restricts to subsets of $N \in \{2, 5, 10, 20\}$ words with a fixed seed to test how calibration degrades as the task gets harder.

The per-preset readout layer is empirical: Qwen3.6-27B reads at 75% depth, not the usual 50% (at 50% accuracy collapses to ~6%).

## Reproducing the results

Per-method, per-preset JSONs land in `results/<preset>/taboo_uq/` (or `run_N/` after rename). The committed `results/qwen3-8b/run_3/` and `results/qwen3.6-27b/taboo_uq/` are the canonical 6000-sample completions referenced above. Re-derive the bootstrap CIs, post-hoc calibration sweep, and LaTeX summary tables from these JSONs with:

```bash
uv run python scripts/analysis/bootstrap_cis.py
uv run python scripts/analysis/post_hoc_calibration.py
uv run python scripts/analysis/build_latex_tables.py
uv run python scripts/figures/layer_sweep.py
```
