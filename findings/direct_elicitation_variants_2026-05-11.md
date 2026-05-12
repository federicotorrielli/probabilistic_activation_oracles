# Appendix B — Direct-elicitation prompt variants (mini-study)

Generated: 2026-05-11

> **Summary for the main paper.** We tried five different options for
> eliciting confidence from the steered oracle on a small sample
> (n = 30 per variant). The verbalized-linguistic variant clearly wins
> on ranking (AUROC 0.957) while the zero-shot baseline is degenerate
> (AUROC 0.500, model always says ~100). Full per-variant table and
> per-sample analysis below. Treat absolute numbers as indicative —
> a full sweep is needed before any claim is settled.

Model: `qwen3-8b` (Qwen3-8B + taboo LoRA + steered activations)
Slice: 6 words × 1 verbalizer × 5 context prompts = **30 samples per variant**
Code: `pao/experiments/mini_direct_variants.py`
Raw data: `results/qwen3-8b/mini_direct_variants_2026-05-11/`

## Motivation

In the full `run_3` sweep, the existing direct-elicitation method (ask the model
to rate its confidence 0–100 in a follow-up turn) gives 99.8% confidence on
correct answers and 99.5% on wrong ones, for ECE 58% and AUROC 0.516 — the
worst-ranked of all 16 methods. This appendix asks whether a different
elicitation prompt can break the saturation.

## Variants

All five variants score the **same generated answer** under the same activation
steering hook, so the comparison is purely about the elicitation prompt.

| ID | Name | Mechanic |
|---|---|---|
| V0 | `zero_shot` (control) | "On a scale of 0 to 100, how confident are you?" |
| V1 | `few_shot_numeric` | 4 in-context demos in unrelated domains (capitals, arithmetic) at confidences 95/70/40/15 |
| V2 | `verbalized_linguistic` | "Reply with exactly one of: very low / low / medium / high / very high." Constrained logit scoring over the 5 labels. |
| V3 | `p_true` | "Question: Q. Proposed answer: A. Is this correct? Reply Yes or No." Confidence = P(Yes) over Yes/No logits (Kadavath et al.). |
| V4 | `hedged` | "Be honest — you may be wrong, and many models are overconfident. On 0–100, how confident?" |

CoT-then-number was excluded for token cost.

## Scorecard

Accuracy is 0.37 across all variants because they share the same answer turn.

| variant | mean conf on correct | mean conf on wrong | gap | ECE | Brier | **AUROC** |
|---|---:|---:|---:|---:|---:|---:|
| zero_shot | 1.000 | 1.000 | 0.000 | 0.633 | 0.633 | 0.500 |
| few_shot_numeric | 0.950 | 0.905 | 0.045 | 0.555 | 0.543 | 0.526 |
| **verbalized_linguistic** | **0.876** | **0.852** | **0.025** | **0.494** | **0.465** | **0.957** |
| p_true | 0.428 | 0.470 | −0.042 | 0.388 | 0.315 | 0.440 |
| hedged | 0.909 | 0.900 | 0.009 | 0.537 | 0.516 | 0.591 |

### Alternative readouts for V2

The verbalized variant ships the full 5-label distribution. Mapping to a
single confidence number can be done several ways:

| readout | gap | ECE | Brier | AUROC |
|---|---:|---:|---:|---:|
| expected value (current) | 0.025 | 0.494 | 0.465 | 0.957 |
| **P(very high) only** | **0.077** | 0.530 | 0.447 | 0.957 |
| P(high) + P(very high) | 0.024 | 0.604 | 0.586 | 0.952 |

The expected-value mapping (0.1 / 0.3 / 0.5 / 0.7 / 0.9) preserves the
ranking but compresses everything to a narrow [0.82, 0.89] band because
the top label is *always* "very high". The **raw P(very high)** is a
sharper readout — same AUROC, 3× the mean-confidence gap. ECE is slightly
worse but trivially fixable with temperature/affine calibration over a
held-out slice.

## What the per-sample data shows

### V2 wins via probability mass, not label choice

Every single sample picked "very high" as the top label. The signal lives
in the *mass* the model places on that label:

| | P(very_high) on correct | P(very_high) on wrong |
|---|---|---|
| range | 0.86–0.95 | 0.79–0.89 |
| mean | 0.914 | 0.837 |

The model "knows" — the activations carry enough signal that the
distribution over linguistic labels tilts more toward "very high" when
the predicted word matches the target. Numeric prompts can't access this
because the model commits to one number; linguistic-label scoring keeps
the full distribution.

### V3 fails because the model has semantic priors over "secret words"

The 8 highest P(Yes) wrong answers all use generic plausible nouns:

| word target | proposed answer | P(Yes) | is correct? |
|---|---|---:|---|
| moon | `secret` | 0.92 | No |
| song | `secret` | 0.91 | No |
| clock | `secret` | 0.89 | No |
| clock | `secret` | 0.87 | No |
| clock | `name` | 0.68 | No |
| smile | `password` | 0.59 | No |
| song | `password` | 0.56 | No |
| snow | `password` | 0.56 | No |

The meta-judgment turn engages the model's prior over what's a plausible
taboo answer ("secret"/"password" are highly plausible in the abstract)
rather than the activation-grounded answer. The steering hook is attached
to the meta-judgment forward pass, but P(Yes|"is 'secret' correct?")
appears dominated by semantic plausibility, not activation match. Two
likely paths to rescue p_true, untested here:

- Reframe as: "Given the hidden context, does this answer match? Yes/No."
- Re-render the original oracle prompt as the prefix instead of a free-standing
  judgment turn, so the steering positions land somewhere meaningful.

### V1 and V4 are weak nudges of V0

Few-shot demos drop the modal answer from 100 to ~95 and produce a 4.5pp
gap, but the model still anchors near-saturation. Hedged framing produces
a 0.9pp gap. Neither fundamentally breaks the "always certain" pattern.

## Recommendation (for the main code path)

**Port V2 (verbalized linguistic) into `pao/methods/direct_elicitation.py`
as a new readout mode, behind a `SamplingConfig` flag.** Keep the current
numeric direct method too — they answer different questions.

Concretely:

1. Add a `linguistic` mode to `direct_elicitation()` that runs the 5-label
   scoring (reusing `_score_continuations` exactly as the mini does).
2. Emit at least three readouts in `ElicitationResult`: expected value,
   P(very_high), P(high)+P(very_high). Let the analysis pick the best one
   per slice.
3. Add `direct_linguistic` (and possibly `direct_linguistic_p_max`) to the
   method list in `run_taboo_uq.py` so it appears in the next full sweep.
4. Do **not** port V3 (p_true) as-is — needs a reframing pass first.

Before merging, confirm on a larger slice (≥ 12 words × 5 ctx) that the
0.957 AUROC holds. The mini-study has only 11 correct / 19 wrong; the
AUROC has wide confidence bounds at n=30.

Calibration follow-up: temperature/affine scaling on V2's P(very_high) over
a held-out word slice. With a ranking that strong, even simple Platt
scaling should bring ECE under 10%.

## Caveats

- n = 30 per variant. AUROC at this size has SE ≈ 0.05. Treat 0.957 as
  "looks great, verify with a full sweep" not "settled."
- All 5 context prompts come from `taboo_direct_test.txt` lines 1–5,
  which are all jailbreak/role-bypass framings. Confidence patterns may
  shift on more neutral prompts.
- One verbalizer prompt only ("Answer with a single word only…"). The other
  two in `VERBALIZER_PROMPTS_TABOO` weren't exercised.
- The accuracy of 0.37 is below run_3's full-slice 0.41–0.42 because the
  word set is hand-picked to mix easy and hard words. This doesn't affect
  the relative comparison between variants.
