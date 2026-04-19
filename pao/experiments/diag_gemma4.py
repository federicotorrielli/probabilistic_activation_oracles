"""Diagnostic for Gemma 4 oracle gibberish output.

Isolates adapter activation, chat-template rendering, and steering from each
other. Run on the remote B200 with:

    uv run python -m pao.experiments.diag_gemma4 --preset gemma-4-e2b

Prints:
  1. Model class + module hierarchy sanity
  2. Active adapter(s) and count of matched LoRA modules
  3. Rendered oracle prompt (token ids + decoded)
  4. Verbalizer-only generation (no steering) on a trivial prompt
  5. Verbalizer-only generation (no steering) on the oracle prompt
  6. Full steered generation on the oracle prompt
"""

from __future__ import annotations

import argparse

import torch
from peft import LoraConfig

from pao.config import MODEL_PRESETS, TABOO_WORDS, VERBALIZER_PROMPTS_TABOO, ModelConfig
from pao.experiments.run_taboo_uq import prepare_activation_and_sampler, setup_model
from pao.hf_utils import (
    SPECIAL_TOKEN,
    get_hf_submodule,
    get_introspection_prefix,
    get_text_config,
    load_lora_adapter,
)


def count_lora_modules(model, adapter_name: str) -> int:
    """Count modules carrying an lora_A submodule for adapter_name."""
    n = 0
    for _name, mod in model.named_modules():
        if hasattr(mod, "lora_A") and adapter_name in getattr(mod, "lora_A", {}):
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True, choices=sorted(MODEL_PRESETS))
    args = ap.parse_args()

    cfg = ModelConfig.from_preset(args.preset)

    # -- 1. load model ----------------------------------------------------------
    model, tokenizer, device, verbalizer_adapter = setup_model(cfg)

    print("\n=== 1. MODEL STRUCTURE ===")
    print("class:", type(model).__name__)
    print("base_model_name_or_path:", model.config._name_or_path)
    print("has text_config:", hasattr(model.config, "text_config"))
    text_cfg = get_text_config(model)
    print("num_hidden_layers:", text_cfg.num_hidden_layers)

    # -- 2. adapter activation --------------------------------------------------
    print("\n=== 2. ADAPTER STATE ===")
    print("active_adapters:", model.active_adapters())
    print("peft_config keys:", list(model.peft_config.keys()))
    print(
        "verbalizer adapter name:",
        verbalizer_adapter,
    )
    n_verb = count_lora_modules(model, verbalizer_adapter) if verbalizer_adapter else 0
    print(f"LoRA modules carrying {verbalizer_adapter!r}:", n_verb)
    # Also load one target adapter to count modules
    tgt_path = cfg.target_lora_template.format(word="ship")
    tgt_name = load_lora_adapter(model, tgt_path)
    n_tgt = count_lora_modules(model, tgt_name)
    print(f"LoRA modules carrying {tgt_name!r}:", n_tgt)

    # Expected: num_layers * 7 - (num_kv_shared_layers * 2) for verbalizer
    # Gemma4 E2B: text_cfg.num_kv_shared_layers might exist
    kv_shared = getattr(text_cfg, "num_kv_shared_layers", 0)
    print("num_kv_shared_layers:", kv_shared)
    expected = text_cfg.num_hidden_layers * 7 - kv_shared * 2
    print(f"expected LoRA modules (q,k,v,o,gate,up,down × layers − kv-shared × 2): {expected}")

    # -- 3. render oracle prompt -----------------------------------------------
    print("\n=== 3. ORACLE PROMPT RENDER ===")
    num_positions = 10
    act_layer = int(text_cfg.num_hidden_layers * (cfg.selected_layer_percent / 100))
    oracle_user_content = (
        get_introspection_prefix(act_layer, num_positions)
        + VERBALIZER_PROMPTS_TABOO[0]
    )
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": oracle_user_content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    print("rendered str:", repr(rendered)[:300])
    oracle_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": oracle_user_content}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors=None,
        return_dict=False,
        padding=False,
        enable_thinking=False,
    )
    print("n_tokens:", len(oracle_ids))
    print("last 20 tokens decoded:", repr(tokenizer.decode(oracle_ids[-20:])))

    # -- 4. verbalizer-only on trivial prompt ----------------------------------
    print("\n=== 4. VERBALIZER-ONLY / TRIVIAL PROMPT ===")
    model.set_adapter(verbalizer_adapter)
    trivial_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 2 + 2?"}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors=None,
        return_dict=False,
        padding=False,
        enable_thinking=False,
    )
    trivial = torch.tensor([trivial_ids], device=device)
    with torch.no_grad():
        out = model.generate(
            trivial,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen = tokenizer.decode(out[0, trivial.shape[1]:], skip_special_tokens=False)
    print("generation:", repr(gen))

    # -- 5. verbalizer-only on oracle prompt (NO STEERING) ---------------------
    print("\n=== 5. VERBALIZER-ONLY / ORACLE PROMPT (no steering) ===")
    ids = torch.tensor([oracle_ids], device=device)
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=False)
    print("generation:", repr(gen))

    # -- 6. full steered generation --------------------------------------------
    print("\n=== 6. FULL STEERED (oracle + steering hook) ===")
    ctx_prompts_path = (
        __import__("pao.config", fromlist=["AO_ROOT"]).AO_ROOT
        / "datasets/taboo/taboo_direct_test.txt"
    )
    with open(ctx_prompts_path) as f:
        ctx = next(line.strip() for line in f if line.strip())
    sampler, oracle_ids_pipeline = prepare_activation_and_sampler(
        model=model,
        tokenizer=tokenizer,
        device=device,
        cfg=cfg,
        target_adapter=tgt_name,
        verbalizer_adapter=verbalizer_adapter,
        context_prompt=ctx,
        verbalizer_prompt=VERBALIZER_PROMPTS_TABOO[0],
    )
    with sampler:
        texts = sampler.generate_batch_texts(
            context=oracle_ids_pipeline,
            temperature=0.0,
            max_new_tokens=15,
            num_samples=1,
            do_sample=False,
        )
    print("target_word expected: ship")
    print("steered generation:", repr(texts[0]))


if __name__ == "__main__":
    main()
