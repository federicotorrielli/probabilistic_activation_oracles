"""Per-layer log-prob sweep for Appendix E.

Reads activations at every layer index of the target stack and runs the
log-prob baseline through the oracle for each layer. Produces
``results/{preset}/layer_sweep/sweep.json`` with per-word per-layer
accuracy + mean log-prob.

Compared with the main UQ runner this is intentionally cheap:
  * only the log-prob baseline (no bootstrap / MCMC / sensitivity)
  * one verbalizer prompt (the first in ``VERBALIZER_PROMPTS_TABOO``)
  * a small context subset (default ``--n-contexts 5``)

For each (word, context) we collect activations at every layer in one
forward pass, then loop over layers for the oracle generation.

Usage:
    uv run python -m pao.experiments.layer_sweep \
        --preset qwen3-8b --n-contexts 5 --output-dir results/qwen3-8b/layer_sweep

Resumes per-word from ``sweep.json`` if it exists.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm
from transformers import AutoConfig

from pao.answer_extraction import extract_predicted_word
from pao.config import (
    AO_ROOT,
    MODEL_PRESETS,
    TABOO_WORDS,
    VERBALIZER_PROMPTS_TABOO,
    ExperimentConfig,
    ModelConfig,
)
from pao.hf_utils import (
    SPECIAL_TOKEN,
    encode_messages,
    find_pattern_in_tokens,
    get_hf_submodule,
    get_introspection_prefix,
    get_text_config,
    load_lora_adapter,
    set_seed,
)
from pao.methods.logprob_baseline import logprob_confidence
from pao.oracle_sampler import SteeredAutoregressiveSampler

from pao.experiments.run_taboo_uq import setup_model


@dataclass
class SweepRow:
    word: str
    layer: int
    context_idx: int
    extracted_word: str
    is_correct: bool
    mean_log_prob: float
    first_token_max_prob: float
    word_prob_no_offset: float


def _collect_all_layer_acts(
    model,
    tokenizer,
    device,
    cfg: ModelConfig,
    target_adapter: str,
    verbalizer_adapter: Optional[str],
    context_prompt: str,
    num_layers: int,
) -> tuple[dict[int, torch.Tensor], int]:
    """One forward pass with the target adapter, hooks on every layer.

    The hooks ``.detach().clone()`` each captured tensor: without that copy,
    later decoder layers mutate the residual stream in-place and earlier
    captures (which are mere references into the same buffer) silently get
    corrupted by the time the function returns. ``collect_activations_
    multiple_layers`` in ``hf_utils`` gets away with no clone because it
    EarlyStops right after capturing a single layer.
    """
    formatted_prompt = [{"role": "user", "content": context_prompt}]
    inputs_BL = encode_messages(
        tokenizer=tokenizer,
        message_dicts=[formatted_prompt],
        add_generation_prompt=True,
        enable_thinking=False,
        device=device,
    )

    model.set_adapter(target_adapter)
    submodules = {i: get_hf_submodule(model, i) for i in range(num_layers)}
    module_to_layer = {sm: i for i, sm in submodules.items()}
    acts_by_layer: dict[int, torch.Tensor] = {}

    def hook(module, _inputs, outputs):
        layer = module_to_layer[module]
        out = outputs[0] if isinstance(outputs, tuple) else outputs
        acts_by_layer[layer] = out.detach().clone()

    handles = [sm.register_forward_hook(hook) for sm in submodules.values()]
    try:
        with torch.no_grad():
            _ = model(**inputs_BL)
    finally:
        for h in handles:
            h.remove()

    seq_len = next(iter(acts_by_layer.values())).shape[1]

    if verbalizer_adapter is not None:
        model.set_adapter(verbalizer_adapter)

    return acts_by_layer, seq_len


def _run_logprob_at_layer(
    model,
    tokenizer,
    device,
    cfg: ModelConfig,
    layer_idx: int,
    acts_BLD: torch.Tensor,
    seq_len: int,
    verbalizer_prompt: str,
    word: str,
    max_new_tokens: int,
) -> SweepRow | None:
    """Build the oracle prompt for ``layer_idx``, inject, run log-prob."""
    start = max(0, seq_len + cfg.segment_start_idx)
    end = (
        seq_len + cfg.segment_end_idx
        if cfg.segment_end_idx <= 0
        else cfg.segment_end_idx
    )
    positions_rel = list(range(start, end))
    num_positions = len(positions_rel)
    if num_positions == 0:
        return None

    steering_vectors = [acts_BLD[0, positions_rel, :]]

    oracle_user_content = (
        get_introspection_prefix(layer_idx, num_positions) + verbalizer_prompt
    )
    oracle_messages = [{"role": "user", "content": oracle_user_content}]
    oracle_ids = tokenizer.apply_chat_template(
        oracle_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors=None,
        return_dict=False,
        padding=False,
        enable_thinking=False,
    )
    if not isinstance(oracle_ids, list) or (
        oracle_ids and not isinstance(oracle_ids[0], int)
    ):
        raise TypeError("apply_chat_template returned unexpected shape")

    steering_positions = [
        find_pattern_in_tokens(oracle_ids, SPECIAL_TOKEN, num_positions, tokenizer)
    ]
    injection_submodule = get_hf_submodule(model, cfg.injection_layer)

    sampler = SteeredAutoregressiveSampler(
        model=model,
        tokenizer=tokenizer,
        device=device,
        submodule=injection_submodule,
        steering_vectors=steering_vectors,
        positions=steering_positions,
        steering_coefficient=1.0,
        dtype=cfg.dtype,
    )

    with sampler:
        lp = logprob_confidence(
            sampler=sampler,
            context=oracle_ids,
            answer_vocab=TABOO_WORDS,
            max_new_tokens=max_new_tokens,
        )

    extracted = extract_predicted_word(lp.generated_text, TABOO_WORDS)
    return SweepRow(
        word=word,
        layer=layer_idx,
        context_idx=-1,  # filled by caller
        extracted_word=extracted,
        is_correct=extracted.lower() == word.lower(),
        mean_log_prob=lp.mean_log_prob,
        first_token_max_prob=lp.first_token_max_prob,
        word_prob_no_offset=lp.word_prob_no_offset,
    )


def _save_sweep(out_path: Path, payload: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.json")
    with open(tmp, "w") as f:
        json.dump(payload, f)
    tmp.replace(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True, choices=sorted(MODEL_PRESETS))
    ap.add_argument(
        "--n-contexts",
        type=int,
        default=5,
        help="Number of context prompts to use per word (5 is the standard sweep budget).",
    )
    ap.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated layer indices to sweep. Default = all.",
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=15,
        help="Max tokens for log-prob baseline generation.",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Defaults to results/{preset}/layer_sweep.",
    )
    ap.add_argument(
        "--no-resume",
        action="store_true",
        help="Discard existing sweep.json and start fresh.",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)

    cfg = ModelConfig.from_preset(args.preset)

    base_config = AutoConfig.from_pretrained(cfg.model_name)
    text_cfg = get_text_config(base_config)
    num_layers = text_cfg.num_hidden_layers

    if args.layers:
        layers = [int(x) for x in args.layers.split(",") if x.strip()]
    else:
        layers = list(range(num_layers))

    output_dir = Path(args.output_dir or f"results/{args.preset}/layer_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = output_dir / "sweep.json"

    # Resume
    completed_words: set[str] = set()
    rows: list[dict] = []
    if sweep_path.exists() and not args.no_resume:
        try:
            with open(sweep_path) as f:
                prior = json.load(f)
            if prior.get("preset") == args.preset and prior.get("num_layers") == num_layers:
                rows = prior.get("rows", [])
                completed_words = {r["word"] for r in rows}
                print(
                    f"Resuming: {len(completed_words)} words already swept "
                    f"({len(rows)} rows). Skipping those."
                )
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Could not resume from {sweep_path}: {exc}; starting fresh.")
            rows = []

    # Load contexts
    ctx_file = AO_ROOT / "datasets/taboo/taboo_direct_test.txt"
    with open(ctx_file) as f:
        all_contexts = [line.strip() for line in f if line.strip()]
    contexts = all_contexts[: args.n_contexts]
    verbalizer_prompt = VERBALIZER_PROMPTS_TABOO[0]

    print(
        f"[layer_sweep] preset={args.preset} model={cfg.model_name} "
        f"layers={len(layers)} (0..{num_layers - 1}) words={len(TABOO_WORDS)} "
        f"contexts={len(contexts)} verbalizer_prompts=1"
    )

    # Load model once
    model, tokenizer, device, verbalizer_adapter = setup_model(cfg)

    words_to_run = [w for w in TABOO_WORDS if w not in completed_words]
    pbar_total = len(words_to_run) * len(contexts) * len(layers)
    pbar = tqdm(total=pbar_total, desc="layer-sweep")

    try:
        for word in words_to_run:
            target_lora_path = cfg.target_lora_template.format(word=word)
            target_adapter = load_lora_adapter(model, target_lora_path)

            for ctx_idx, ctx in enumerate(contexts):
                acts_by_layer, seq_len = _collect_all_layer_acts(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    cfg=cfg,
                    target_adapter=target_adapter,
                    verbalizer_adapter=verbalizer_adapter,
                    context_prompt=ctx,
                    num_layers=num_layers,
                )

                for layer_idx in layers:
                    row = _run_logprob_at_layer(
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        cfg=cfg,
                        layer_idx=layer_idx,
                        acts_BLD=acts_by_layer[layer_idx],
                        seq_len=seq_len,
                        verbalizer_prompt=verbalizer_prompt,
                        word=word,
                        max_new_tokens=args.max_new_tokens,
                    )
                    pbar.update(1)
                    if row is None:
                        continue
                    row.context_idx = ctx_idx
                    rows.append(
                        {
                            "word": row.word,
                            "layer": row.layer,
                            "context_idx": row.context_idx,
                            "extracted_word": row.extracted_word,
                            "is_correct": row.is_correct,
                            "mean_log_prob": row.mean_log_prob,
                            "first_token_max_prob": row.first_token_max_prob,
                            "word_prob_no_offset": row.word_prob_no_offset,
                        }
                    )

            # Checkpoint after each completed word
            payload = {
                "preset": args.preset,
                "model_name": cfg.model_name,
                "num_layers": num_layers,
                "swept_layers": layers,
                "n_words": len(TABOO_WORDS),
                "n_contexts": len(contexts),
                "verbalizer_prompt": verbalizer_prompt,
                "max_new_tokens": args.max_new_tokens,
                "seed": args.seed,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "rows": rows,
            }
            _save_sweep(sweep_path, payload)
    finally:
        pbar.close()

    print(f"\n[layer_sweep] done. {len(rows)} rows -> {sweep_path}")


if __name__ == "__main__":
    main()
