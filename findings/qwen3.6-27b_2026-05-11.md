# Qwen3.6-27B activation oracle — debugging notes

**Date:** 2026-05-11
**Status:** Resolved. Fix: read activations from layer 48 (`selected_layer_percent=75`) instead of layer 32.
**Outcome:** Qwen3.6-27B mean accuracy rises from 6.25% to 40.62%, matching Qwen3-8B's 41.05%.

---

## What was happening

The first Qwen3.6-27B run reported 0% accuracy across all 16 UQ methods at
901 / 6000 iterations. Smoke-test outputs looked like base-model template
echoes (`"The secret word is 'NAME_1'"`, etc). Retraining the verbalizer
(`EvilScript/activation-oracle-Qwen3_6-27B`) did not help. A second run at
2150 / 6000 iterations reached approximately 7% mean, with a striking
per-word pattern: `moon` ~34%, `snow` and `green` at low double digits,
`ship` / `song` / `wave` at exactly 0 / 300.

## What it actually was

**pao (and the upstream trainer's own eval) defaulted to reading
activations from the middle of the stack (`selected_layer_percent=50`,
i.e. layer 32).** For Qwen3.6-27B that is the wrong layer. The verbalizer
was trained on activations from `layer_percents=[25, 50, 75]`, i.e. layers
[16, 32, 48], but at inference time only the **layer-48 (75%)** readout
carries usable secret-word signal.

Sweeping the selected layer percent on a fixed verbalizer plus target
adapters, with all flash-attn extras installed for correct numerics:

| selected_layer_percent | active_layer | mean accuracy (20 words, 8 prompts each) |
|---|---|---|
| 25 | 16 | 0.00% (all 20 words at 0/8) |
| 50 | 32 | 6.25% (only 5 of 20 non-zero) |
| **75** | **48** | **40.62%** (19 of 20 non-zero) |

Qwen3-8B's baseline on the same 20-word set is 41.05%. Layer 48 closes
the gap entirely.

## The fix

`pao/config.py`: per-preset readout-layer override, set to 75 for the
`qwen3.6-27b` preset. Existing presets (Qwen3-8B, Qwen3-32B,
gemma-4-31b) keep 50. Single config change, no retraining, no other code
edits needed for this fix.

After applying the fix, re-running pao on Qwen3.6-27B takes approximately
48 hours on a B200-class GPU (6000 iterations at ~29 s per iteration at
27B scale). Output mean is expected to land around 40% across the 20
words, with the same per-method spread observed on Qwen3-8B.

---

## How we got here (so the dead ends are documented)

The investigation went through three wrong hypotheses before landing on
the right one. They are recorded below because each took real time and
the "evidence" for them looks compelling at first glance.

### Dead end 1: "The LoRA loader is broken"

**Symptom that misled.** transformers 5.8's `model.load_adapter(...)`
prints a verbose `LOAD REPORT` listing every wrapper as `MISSING`:

```
model.layers.{0...62}.linear_attn.in_proj_a.lora_A.EvilScript/...weight | MISSING |
...
Notes:
- MISSING: those params were newly initialized because missing from the checkpoint.
```

It looks like the LoRA weights were never loaded.

**What is actually happening.** The report is emitted for each
`load_adapter` call. It lists every adapter wrapper in the model that is
not present in the *current* safetensors file, including wrappers that
were correctly populated by a *previous* `load_adapter` call for a
different adapter. They are not actually missing.

**Verification.** Compared post-load parameter norms against safetensors
ground truth:

- `layers.0.linear_attn.in_proj_a.lora_A.weight` (verbalizer):
  safetensors norm 4.9520, post-load norm 4.9520
- Same key for the `taboo-ship` adapter: post-load norm 3.2794 (within
  normal Kaiming-init range, matching safetensors)
- 480 `linear_attn` parameters for the ship adapter all populated
  correctly

**Takeaway.** Do not trust the `MISSING` report. Verify with parameter
norms.

### Dead end 2: "PEFT name-mapping mismatch (`lora_A.weight` vs `lora_A.default.weight`)"

**Symptom that misled.** Inspection of `adapter_model.safetensors`
showed keys like
`base_model.model.model.layers.0.linear_attn.in_proj_a.lora_A.weight` (no
`.default.` infix), while PEFT 0.19's runtime model state dict expects
`...lora_A.{adapter_name}.weight`. Theory: transformers's PEFT
integration was doing strict-match without the rename and treating all
keys as missing.

**What is actually happening.** PEFT's loader does the rename
automatically during `load_state_dict`. The 992 keys in each safetensors
file load correctly under both `model.load_adapter(...)` (transformers
integration) and `PeftModel.from_pretrained(...)` (PEFT direct). The
naming convention difference is irrelevant.

**Verification.** Ran the same loader sequence via two paths
(pao-style and PEFT-direct), inspected post-load norms; both produce the
same correct values.

**Takeaway.** The `.default.` vs no-infix difference is handled
transparently by PEFT 0.19. Do not write code that tries to work around
it.

### Dead end 3: "Per-word target adapter quality is uneven"

**Symptom that misled.** On the partial Qwen3.6-27B checkpoint, accuracy
was sharply bimodal across words: `moon` ~34%, `snow` ~13%,
`ship` / `song` / `wave` at 0 / 300. The upstream trainer's own eval
(its `taboo_open_ended_eval.py` code path, run on local checkpoints)
reproduced the same per-word pattern. That ruled out pao but pointed at
per-word variance in the target adapters or verbalizer.

**What is actually happening.** All 8 target adapters tested have
statistically indistinguishable weights (`lora_A` mean norm ~3.33,
`lora_B` mean norm ~0.32, 992 keys). All 20 target adapters move the
residual stream by approximately 40 – 55% of the base norm at layer 32.
The bimodality is not adapter quality; layer 32 simply happens to have
leaky signal for `moon` (and to a lesser extent `blue`, `snow`, `green`,
`book`) but not for the rest. With layer 48 read, 19 of 20 words light
up.

**Takeaway.** Bimodal failure across nominally equivalent training
artifacts is a signal to question the readout, not the artifacts.

### What finally cracked it

Two diagnostics done in parallel:

1. **Layer sweep** over `selected_layer_percent ∈ {25, 50, 75}` on all
   20 words via the trainer's own eval. Layer 48 (`lp=75`) jumped from
   6.25% mean to 40.62%.
2. **Training data audit** of the trainer's `sft.py`. The verbalizer is
   trained on `act_layers=[16, 32, 48]` (three layers) using activations
   from `latentqa + classification + past_lens` mixes
   (approximately 360k samples, 1 epoch). The training collects
   activations under `model.disable_adapter()` (i.e. base model, no
   LoRA). The verbalizer is trained to read from any of the three
   layers, but at inference time only one is selected. The interesting
   empirical finding is that for Qwen3.6-27B's 64-layer hybrid
   linear / full-attention stack, only the deepest of those three (layer
   48) carries reliable signal.

---

## Evidence inventory

### Per-word accuracy at the recovered layer (`lp=75`, `act_layer=48`)

| word | 27B @ lp=75 (n=8) | 8B run_3 (n=300) | delta |
|---|---|---|---|
| moon | 62.5% | 85.3% | -23 |
| blue | 75.0% | 20.3% | +55 |
| snow | 62.5% | 75.3% | -13 |
| green | 62.5% | 50.0% | +13 |
| book | 62.5% | 33.7% | +29 |
| gold | 62.5% | 32.7% | +30 |
| rock | 50.0% | 14.7% | +35 |
| chair | 50.0% | 57.0% | -7 |
| salt | 50.0% | 47.7% | +2 |
| jump | 37.5% | 65.7% | -28 |
| dance | 37.5% | 35.7% | +2 |
| cloud | 37.5% | 23.0% | +15 |
| smile | 37.5% | 61.7% | -24 |
| ship | 25.0% | 66.0% | -41 |
| wave | 25.0% | 17.3% | +8 |
| flame | 25.0% | 41.0% | -16 |
| flag | 25.0% | 48.3% | -23 |
| song | 12.5% | 14.0% | -1 |
| clock | 12.5% | 14.7% | -2 |
| leaf | 0.0% | 25.0% | -25 |
| **mean** | **40.62%** | **41.05%** | **-0.43** |

Qwen3.6-27B at `lp=75` is competitive with Qwen3-8B word-for-word.
`blue`, `rock`, `gold`, `book`, `green`, and `cloud` actually do better
at 27B than at 8B. Only `leaf` remains stuck at 0%.

### Steering vector magnitudes

To rule out "target adapter does nothing," layer-32 activations were
captured under one chat-formatted context prompt, with each target
adapter alternately enabled. Base residual norm at the last 10 tokens
averaged 64.4. The per-word steering deltas (LoRA-on minus LoRA-off,
mean per-token norm over the last 10 tokens):

| word | delta norm | fraction of base |
|---|---|---|
| moon | 34.54 | 55% |
| song | 28.52 | 46% |
| green | 26.84 | 43% |
| wave | 25.55 | 42% |
| ship | 25.00 | 41% |

All adapters perturb the residual stream by 40 – 55% of its base
magnitude. Nothing is "dead." Pairwise cosine similarity of steering
directions is 0.61 – 0.77, with `moon` slightly more distinct than the
others. Variance in cosine similarity is not what causes the per-word
performance variance; the readout layer is.

### Adapter weight statistics

All 8 target adapters tested plus the verbalizer (probed once):

| repo | mean `lora_A` norm | mean `lora_B` norm | keys |
|---|---|---|---|
| verbalizer | 4.891 | 0.982 | 992 |
| taboo-moon | 3.331 | 0.323 | 992 |
| taboo-ship | 3.334 | 0.329 | 992 |
| taboo-song | 3.333 | 0.329 | 992 |
| taboo-wave | 3.330 | 0.319 | 992 |
| taboo-green | 3.330 | 0.320 | 992 |
| taboo-jump | 3.334 | 0.326 | 992 |
| taboo-rock | 3.332 | 0.329 | 992 |
| taboo-snow | 3.330 | 0.322 | 992 |

Statistically indistinguishable. No undertrained outlier. The same
conclusion holds for the other 12 target adapters checked indirectly via
the `lp=75` sweep (all produce non-zero accuracy except `leaf`).

---

## Configuration and environment notes

### Verbalizer training (from the trainer's `sft.py` and `configs/sft_config.py`)

- Model: `Qwen/Qwen3.6-27B`
- LoRA: r=64, alpha=128, dropout=0.05, target_modules=`all-linear`
  (~100M+ trainable params over ~640 modules at 64 layers)
- Hook injection layer: 1
- Activation read layers: [16, 32, 48] (25 / 50 / 75% of 64)
- Training mix: latentqa (~100k) + classification (~60k across 10
  datasets) + past_lens (~200k) = approximately 360k samples
- Epochs: 1
- Batch size 16, lr=1e-5, grad-accum=1
- Periodic eval during training: effectively off
  (`eval_steps=9_999_999`, `eval_on_start=False`)
- Activations collected under `model.disable_adapter()` (base model, no
  LoRA active)

**Train / inference distribution shift.** Training activations come from
the bare base model. Inference activations come from base plus the
target taboo LoRA active, which perturbs the stream by 40 – 55% of base
norm. This shift is inherent to the activation-oracle paradigm. It works
on Qwen3-8B and on Qwen3.6-27B *if the right layer is read*. The right
layer is empirically determined.

### Reproducibility prerequisites

- A Blackwell-class GPU (B200 was used for this report) with enough HBM
  to hold the model plus one verbalizer LoRA plus one target LoRA.
- An environment with `transformers >= 5.8`, PEFT 0.19.x, and the
  `flash-attn` optional extra installed.
- The `flash-attn` extra is required for two distinct reasons:
  (a) `flash_attention_2` for the full-attention layers, and
  (b) `flash-linear-attention` for the Qwen3.6 `Qwen3_5GatedDeltaNet`
  blocks. Without the extras, the linear-attention layers fall back to a
  pure-torch implementation that is approximately 10x slower and
  produces different numerics from what the verbalizer was trained
  against. Evaluation results without the extras are not comparable.

### Pre-existing pao changes (already in master)

These were applied before this investigation but are load-bearing for
the layer-48 fix to take effect cleanly:

- `pao/hf_utils.py` `resolve_oracle_layers`: gates the Gemma-4-style
  layer snapping on the presence of `"sliding_attention"` in
  `layer_types`. For Qwen3.6 (linear plus full hybrid), no snapping
  occurs; percents `[25, 50, 75]` pass through to `[16, 32, 48]`.
  Injection layer = 1.
- `pao/hf_utils.py` `resolve_attention_implementation`:
  Qwen3.6 / `qwen3_6` returns `flash_attention_2`, matching the
  trainer's backend. Other Qwen3 variants get `flash_attention_4`.
  (See `attention_backends_2026-05-11.md` for the empirical basis of
  this choice. The non-default `flash_attention_4` selection on
  Qwen3.6-27B introduces a numerical regression that must be avoided.)

These changes by themselves did not fix the 0% problem; they just
remove unrelated inconsistencies between training and inference. Keep
them.

### Upstream gotcha for anyone running `taboo_open_ended_eval.py` directly

The trainer's eval uses `dummy_config = LoraConfig()` (line 179) with no
`target_modules`. This crashes on Qwen3.6 + PEFT 0.19.1 with
`ValueError: Please specify target_modules or target_parameters`.
Workaround: `LoraConfig(target_modules=["q_proj"])` — which is what pao
already does.
