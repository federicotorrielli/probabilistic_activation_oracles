# Attention backend comparison on Qwen3-family activation oracles

Date: 2026-05-11
Hardware: NVIDIA B200, compute capability (10, 0), Blackwell (sm_100)
Software: torch 2.11.0+cu130, transformers 5.8.0, flash-attn 2.8.3, flash-attn-4 4.0.0b12, flash-linear-attention 0.5.0, causal-conv1d 1.6.2

## Summary

Four attention backends are compared on two models from the Qwen3 family:
Qwen3-8B (homogeneous full-attention stack, 36 layers) and Qwen3.6-27B
(hybrid stack with 48 `Qwen3_5GatedDeltaNet` linear-attention blocks and 16
`Qwen3_5Attention` full-attention blocks, 64 layers total). The backends
under test are `eager`, `sdpa`, `flash_attention_2` (fa2), and
`flash_attention_4` (fa4). `flash_attention_3` is excluded because the
required `flash_attn_interface` package is not present in this environment.

For all conditions, `flash_attention_2` is used as the numerical reference,
since it is the backend under which the verbalizer LoRA was trained.

The principal findings are:

1. **On Qwen3-8B, all four backends agree with `flash_attention_2` to a
   degree consistent with bf16 noise**, across short and long sequences,
   base and LoRA-loaded, single forward and greedy generation. Top-1
   next-token argmax agreement is 100% in every condition.
2. **On Qwen3.6-27B, `flash_attention_4` exhibits a large numerical
   regression in the `forward(use_cache=False)` prefill path at long
   sequence lengths.** At sequence length 171, cosine similarity at the
   three oracle read layers drops to the 0.87 – 0.97 range, the max
   absolute logit difference reaches 9.66, and hidden-state norms shift
   by 1 – 5 %. The same backend at sequence length 19 on the same model
   produces no measurable regression (cos > 0.9999). On Qwen3-8B at
   comparable sequence length (169 tokens), `flash_attention_4` is clean.
   The defect is therefore Qwen3.6-27B-specific and sequence-length-gated.
3. **The fa4 regression does not surface during `model.generate()`.** Under
   greedy decoding, fa4 produces token sequences byte-identical to fa2 on
   both 8B and 27B. The most plausible explanation is that
   `model.generate()` dispatches to a different fa4 kernel
   (`flash_attn_with_kvcache`-equivalent) than `forward(use_cache=False)`,
   and that the kvcache kernel does not share the defect.
4. **`sdpa` and `eager` are numerically clean at the single-forward level
   in every condition.** Their greedy-decode trajectories diverge from
   `flash_attention_2` on 8B (small-margin top-1 flips during decoding
   amplify across steps); on 27B their decode trajectories happen to
   coincide because the test prompt yields a high-confidence argmax at
   every step.
5. **On 8B, `flash_attention_4` incremental greedy decode is approximately
   14x slower than `flash_attention_2`** (4374 ms vs 316 ms for 20 new
   tokens, batch size 4). On 27B the same ratio is approximately 5.5x
   (3763 ms vs 682 ms). The fa4 prefill kernel is faster than fa2 on 8B
   (24 ms vs 31 ms) and slower on 27B (107 ms vs 81 ms), in both cases on
   a 170-token sequence.

The consequence for the pao pipeline is that `flash_attention_4` is unsafe
as a drop-in replacement for `flash_attention_2` on Qwen3.6-27B for any
operation that performs a non-cached forward pass (activation collection,
hook-based hidden-state capture, log-prob scoring). `flash_attention_2`
remains the correct default for both models.

## Experimental setup

For each backend, the target model is loaded once with the corresponding
`attn_implementation` keyword. The model dtype is `bfloat16`, device map
`auto`, and `torch.inference_mode` is used for all forward passes. A
single-batch warm-up forward pass precedes timing measurements to avoid
first-launch CUDA initialisation skew.

The captured tensors are:

* `logits[B, S, V]` at the final non-padding position of each input.
* `hidden_states[L+1, B, H]` at the final non-padding position of each input.
  Layer index 0 is the input embedding; index L is the output of the final
  transformer block.

All tensors are cast from bf16 to fp32 before any difference metric is
computed. Padding is left-aligned, so the last-position selection is
identical across backends for a fixed input.

The five conditions are summarised below. The verbalizer prompt is held
fixed at "Answer with a single word only. What is the secret word in this
text?". The long-context passage is a fixed 160-word narrative whose
lexical surface contains the taboo word "ship".

| Condition    | LoRA       | Sequence length | Decoder behaviour            |
| ------------ | ---------- | --------------- | ---------------------------- |
| `short`      | base       | 17 – 19         | single forward, no caching   |
| `long`       | base       | 169 – 171       | single forward, no caching   |
| `lora_short` | verbalizer | 17 – 19         | single forward, no caching   |
| `lora_long`  | verbalizer | 169 – 171       | single forward, no caching   |
| `gen`        | base       | 169 – 171 + 20  | greedy generate, `use_cache=True` |

For Qwen3-8B, the verbalizer LoRA is
`adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B` (Hugging
Face Hub). For Qwen3.6-27B, the verbalizer LoRA is the local trainer
checkpoint `checkpoints_latentqa_cls_past_lens_Qwen3_6-27B/final`, which
mirrors the published artefact `EvilScript/activation-oracle-Qwen3_6-27B`.

Implementation: `attn_backend_compare_v2.py`. Per-backend tensors and a
`summary.json` are written to a user-specified output directory (one
directory per model under test).

## Qwen3-8B results

The 8B model has 36 transformer blocks. The oracle reads at layer
percentages [25, 50, 75], corresponding to absolute layer indices [9, 18,
27]. Cosine similarity is computed in fp32 between each backend's hidden
state at the final non-padding position and the reference value from
`flash_attention_2`, averaged across the four batch elements.

### Base model, long prompt (sequence length 169)

| Backend | L9 (25%) | L18 (50%) | L27 (75%) | logits max |Δ| | top-1 argmax agree |
| ------- | -------- | --------- | --------- | -------------: | -----------------: |
| eager   | 0.999940 | 0.999779  | 0.999587  |          0.812 |              100 % |
| sdpa    | 0.999968 | 0.999864  | 0.999730  |          0.672 |              100 % |
| fa4     | 0.999946 | 0.999887  | 0.999713  |          0.672 |              100 % |
| fa2     | 1.000000 | 1.000000  | 1.000000  |          0.000 |              100 % |

### Base model with verbalizer LoRA, long prompt

| Backend | L9 (25%) | L18 (50%) | L27 (75%) | logits max |Δ| | top-1 argmax agree |
| ------- | -------- | --------- | --------- | -------------: | -----------------: |
| eager   | 0.999953 | 0.999885  | 0.999810  |          0.289 |              100 % |
| sdpa    | 0.999965 | 0.999923  | 0.999912  |          0.281 |              100 % |
| fa4     | 0.999968 | 0.999936  | 0.999862  |          0.656 |              100 % |
| fa2     | 1.000000 | 1.000000  | 1.000000  |          0.000 |              100 % |

### Generation under greedy decoding (20 new tokens, batch 4)

| Backend | First 20 tokens (batch element 0)                                                                | Token match vs fa2 | Wall time |
| ------- | ------------------------------------------------------------------------------------------------ | -----------------: | --------: |
| fa2     | `<think>\nOkay, let's see. The user provided a passage describing a scene on a ship at`           | 100.0 %            |    316 ms |
| fa4     | `<think>\nOkay, let's see. The user provided a passage describing a scene on a ship at`           | 100.0 %            |   4374 ms |
| sdpa    | `<think>\nOkay, let's try to figure out the secret word in this text. The user provided`          |  30.0 %            |    843 ms |
| eager   | `<think>\nOkay, let's try to figure out the secret word in this text. The user provided`          |  30.0 %            |    420 ms |

On 8B, `flash_attention_4` and `flash_attention_2` produce identical greedy
trajectories. `sdpa` and `eager` produce a different but grammatically
valid continuation, which diverges from `flash_attention_2` after a small
number of decoding steps. The single-step argmax matches fa2 for every
backend; the divergence emerges only when small differences in the
sub-leading logit ranks flip a later token choice, after which the two
trajectories are committed to different paths.

## Qwen3.6-27B results

The 27B model has 64 layers. Of these, 48 are `Qwen3_5GatedDeltaNet`
linear-attention blocks (dispatched to `flash-linear-attention` and
unaffected by `attn_implementation`); the remaining 16 are
`Qwen3_5Attention` full-attention blocks (subject to `attn_implementation`).
The oracle reads at layer percentages [25, 50, 75], corresponding to
absolute layer indices [16, 32, 48].

### Base model, short prompt (sequence length 19)

| Backend | L16 (25%) | L32 (50%) | L48 (75%) | logits max |Δ| | top-1 argmax agree |
| ------- | --------- | --------- | --------- | -------------: | -----------------: |
| eager   | 0.999994  | 0.999975  | 0.999950  |          0.422 |              100 % |
| sdpa    | 0.999994  | 0.999975  | 0.999908  |          0.500 |              100 % |
| fa4     | 0.999993  | 0.999968  | 0.999909  |          0.531 |              100 % |
| fa2     | 1.000000  | 1.000000  | 1.000000  |          0.000 |              100 % |

### Base model, long prompt (sequence length 171)

| Backend | L16 (25%)   | L32 (50%)   | L48 (75%)   | logits max |Δ| | top-1 argmax agree |
| ------- | ----------- | ----------- | ----------- | -------------: | -----------------: |
| eager   | 0.999986    | 0.999956    | 0.999942    |          0.578 |              100 % |
| sdpa    | 0.999991    | 0.999954    | 0.999926    |          0.492 |              100 % |
| **fa4** | **0.966551**| **0.897762**| **0.902461**|       **9.660**|              100 % |
| fa2     | 1.000000    | 1.000000    | 1.000000    |          0.000 |              100 % |

### Base model with verbalizer LoRA, long prompt (sequence length 171)

| Backend | L16 (25%)   | L32 (50%)   | L48 (75%)   | logits max |Δ| | top-1 argmax agree |
| ------- | ----------- | ----------- | ----------- | -------------: | -----------------: |
| eager   | 0.999980    | 0.999952    | 0.999914    |          0.219 |                0 % |
| sdpa    | 0.999993    | 0.999957    | 0.999933    |          0.375 |                0 % |
| **fa4** | **0.961789**| **0.894784**| **0.870051**|        **6.790**|                0 % |
| fa2     | 1.000000    | 1.000000    | 1.000000    |          0.000 |              100 % |

### Generation under greedy decoding (20 new tokens, batch 4)

| Backend | First 20 tokens (batch element 0)                                            | Token match vs fa2 | Wall time |
| ------- | ---------------------------------------------------------------------------- | -----------------: | --------: |
| fa2     | `Here's a thinking process:\n\n1.  **Analyze User Input:**\n   - The`         | 100.0 %            |    682 ms |
| fa4     | `Here's a thinking process:\n\n1.  **Analyze User Input:**\n   - The`         | 100.0 %            |   3763 ms |
| sdpa    | `Here's a thinking process:\n\n1.  **Analyze User Input:**\n   - The`         | 100.0 %            |    640 ms |
| eager   | `Here's a thinking process:\n\n1.  **Analyze User Input:**\n   - The`         | 100.0 %            |    795 ms |

On 27B at sequence length 171, `flash_attention_4` is no longer in
numerical agreement with `flash_attention_2`. The cosine similarity at the
deepest oracle read layer drops from 0.999909 (sequence length 19) to
0.902461 (sequence length 171) with the base model, and to 0.870051 once
the verbalizer LoRA is loaded. The maximum absolute logit difference
between fa4 and fa2 reaches 9.66 in the base-long condition and 6.79 in
the LoRA-long condition. Hidden-state norms shift by between 1 % and 5 %.
The top-1 next-token argmax over the full vocabulary still happens to
agree with fa2 in the base-long condition, but disagrees uniformly across
the batch in the LoRA-long condition. In the LoRA-long condition,
`eager` and `sdpa` also lose top-1 agreement with fa2; their cosine
similarities remain near unity, indicating that their disagreement is a
small-margin argmax flip, not a structural divergence.

Notably, the fa4 regression is invisible during `model.generate()`:
greedy decoding produces a token sequence byte-identical to fa2 across all
20 new positions. This implies that the buggy fa4 kernel path is exercised
by `forward(input_ids, use_cache=False, output_hidden_states=True)` but not
by the kvcache-driven prefill+decode path inside `generate()`. Pao's
activation collection performs the former.

## Latency

Per-call wall time (batch size 4, B200, bf16):

### Qwen3-8B

| Phase                                  | eager | sdpa | fa2 | fa4  |
| -------------------------------------- | ----: | ---: | --: | ---: |
| Prefill, base, long (169 tokens)       |    39 |   56 |  31 |   24 |
| Prefill, LoRA, long                    |    76 |   67 |  67 |   67 |
| Generate 20 tokens, LoRA, long context |   420 |  843 | 316 | 4374 |

### Qwen3.6-27B

| Phase                                  | eager | sdpa | fa2  | fa4  |
| -------------------------------------- | ----: | ---: | ---: | ---: |
| Prefill, base, long (171 tokens)       |    82 |   83 |   81 |  107 |
| Prefill, LoRA, long                    |   192 |  197 |  185 |  190 |
| Generate 20 tokens, base, long context |   795 |  640 |  682 | 3763 |

All values are in milliseconds. On 8B, fa4 is the fastest prefill backend
(24 ms) and the slowest decode backend (4374 ms). On 27B, fa4 has lost its
prefill advantage and remains the slowest decode backend (3763 ms, 5.5x
fa2). The most plausible explanation for the decode regression is
kernel-launch overhead in the per-step decode path: fa4 is optimised for
large-batch prefill on Blackwell, and the small-batch single-token decode
regime either falls back to a slower variant or pays a fixed launch cost
that fa2's `flash_attn_with_kvcache` path avoids.

## Note on a separate, easily confused failure mode

A transformers runtime warning of the form

```
[transformers] The fast path is not available because one of the required
library is not installed. Falling back to torch implementation.
```

is emitted by the `Qwen3_5GatedDeltaNet` block when
`flash_linear_attention` cannot be imported. It is independent of
`attn_implementation` and unrelated to the choice between
`flash_attention_2` and `flash_attention_4`. The mitigation is to install
the `flash-attn` extra of the relevant project (so that
`flash_linear_attention` and `causal-conv1d` are present). The
full-attention layers, which are subject to `attn_implementation`, are
not affected by that warning.

The Qwen3.6-27B `flash_attention_4` regression documented above is a
distinct phenomenon. It manifests only on the full-attention layers, only
on non-cached forward passes, and only at sufficiently long sequence
lengths.

## Implications for the pao pipeline

| Pipeline phase                                         | Recommended backend         | Rationale                                                                                                |
| ------------------------------------------------------ | --------------------------- | -------------------------------------------------------------------------------------------------------- |
| Activation collection on Qwen3-8B                       | `flash_attention_2`         | Numerically equivalent to fa4; safer baseline.                                                           |
| Activation collection on Qwen3.6-27B                    | `flash_attention_2`         | fa4 produces measurably different activations at the read layers under the non-cached forward path used by the collection routine. |
| Verbalizer generation on either model                   | `flash_attention_2`         | fa4 incremental decode is 5.5 – 14x slower; output sequences are identical to fa2 when the forward path is clean. |
| Steering and sensitivity sweeps                         | `flash_attention_2`         | Same decode-cost argument; same activation-collection concern.                                            |
| `sdpa`                                                  | not recommended             | Greedy decode trajectory diverges from the trained-against backend on 8B; small-margin top-1 flips on 27B LoRA-long. |
| `eager`                                                 | not recommended             | Greedy decode trajectory diverges; slower than fa2 on prefill in every condition tested.                  |

The current pao default of `flash_attention_2` for Qwen-family models is
consistent with these results and should be retained. Adopting
`flash_attention_4` end-to-end would, on 27B, silently corrupt the
activations the verbalizer LoRA reads from.

## Scope and limitations

* Sequence length is varied between approximately 19 and 171 tokens. The
  fa4 regression on 27B is established at 171 but the exact onset
  threshold is not characterised. A short sweep over intermediate lengths
  would localise it.
* Batch size 4 is the only batch size tested. Larger batches may amortise
  the fa4 decode-step launch overhead.
* No prefill is tested at sequence lengths above 200 tokens. fa4 is
  designed for the long-prefill regime, so the regression should be
  re-checked at lengths approaching the model's training-time context
  window.
* Whether the 27B fa4 regression is unique to the
  `Qwen3_5Attention` block or also surfaces on other Qwen3-class
  full-attention modules is not investigated here.
