"""Low-level HF helpers: model/tokenizer/LoRA loading, activation
collection, residual-stream steering, and chat-template prompt encoding."""

from __future__ import annotations

import contextlib
import random
from typing import Callable

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Introspection prompt layout
# ---------------------------------------------------------------------------

SPECIAL_TOKEN = " ?"


def get_introspection_prefix(sae_layer: int, num_positions: int) -> str:
    prefix = f"Layer: {sae_layer}\n"
    prefix += SPECIAL_TOKEN * num_positions
    prefix += " \n"
    return prefix


def find_pattern_in_tokens(
    token_ids: list[int],
    special_token_str: str,
    num_positions: int,
    tokenizer: AutoTokenizer,
) -> list[int]:
    start_idx = 0
    end_idx = len(token_ids)
    special_token_id = tokenizer.encode(special_token_str, add_special_tokens=False)
    assert len(special_token_id) == 1, (
        f"Expected single token, got {len(special_token_id)}"
    )
    special_token_id = special_token_id[0]
    positions = []

    for i in range(start_idx, end_idx):
        if len(positions) == num_positions:
            break
        if token_ids[i] == special_token_id:
            positions.append(i)

    assert len(positions) == num_positions, (
        f"Expected {num_positions} positions, got {len(positions)}"
    )
    assert positions[-1] - positions[0] == num_positions - 1, (
        f"Positions are not consecutive: {positions}"
    )

    final_pos = positions[-1] + 1
    final_tokens = token_ids[final_pos : final_pos + 2]
    final_str = tokenizer.decode(final_tokens, skip_special_tokens=False)
    assert "\n" in final_str, f"Expected newline in {final_str}"

    return positions


# ---------------------------------------------------------------------------
# Model / tokenizer / seed helpers
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and torch for reproducible runs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(
    model_name: str,
    dtype: torch.dtype,
    attn_implementation: str = "auto",
    **model_kwargs,
) -> AutoModelForCausalLM:
    print("🧠 Loading model...")

    attn = resolve_attention_implementation(model_name, attn_implementation)
    print(f"  attention: {attn}")

    kwargs: dict = {
        "device_map": "auto",
        "attn_implementation": attn,
        "dtype": dtype,
        **model_kwargs,
    }

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    return model


def resolve_attention_implementation(model_name: str, requested: str = "auto") -> str:
    """Resolve the attention backend to pass to Transformers.

    ``sdpa`` is not an eager fallback: PyTorch dispatches it to fused GPU
    scaled-dot-product attention kernels when supported. It is the fastest
    known-working backend for Qwen3 under the current Transformers 5.6 stack,
    where the explicit FA2 integration can crash with a missing attention-sink
    tensor.
    """
    if requested != "auto":
        return requested

    model_name_lower = model_name.lower()
    if "gemma" in model_name_lower:
        return "eager"
    if "qwen3" in model_name_lower:
        return "flash_attention_4"
    return "flash_attention_2"


def load_tokenizer(
    model_name: str,
) -> AutoTokenizer:
    # Load tokenizer
    print("📦 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token_id = tokenizer.eos_token_id
    return tokenizer


# ---------------------------------------------------------------------------
# Message encoding / LoRA loading
# ---------------------------------------------------------------------------


def encode_messages(
    tokenizer: AutoTokenizer,
    message_dicts: list[list[dict[str, str]]],
    add_generation_prompt: bool,
    enable_thinking: bool,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    messages = []
    for source in message_dicts:
        rendered = tokenizer.apply_chat_template(
            source,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
        messages.append(rendered)
    inputs_BL = tokenizer(
        messages, return_tensors="pt", add_special_tokens=False, padding=True
    ).to(device)
    return inputs_BL


def sanitize_lora_name(lora_path: str) -> str:
    return lora_path.replace(".", "_")


def load_lora_adapter(model: AutoModelForCausalLM, lora_path: str) -> str:
    sanitized_lora_name = sanitize_lora_name(lora_path)

    if sanitized_lora_name not in model.peft_config:
        print(f"Loading LoRA: {lora_path}")
        model.load_adapter(
            lora_path,
            adapter_name=sanitized_lora_name,
            is_trainable=False,
            low_cpu_mem_usage=True,
        )

    return sanitized_lora_name


# ---------------------------------------------------------------------------
# Activation collection
# ---------------------------------------------------------------------------


class EarlyStopException(Exception):
    """Custom exception for stopping model forward pass early."""

    pass


def collect_activations_multiple_layers(
    model: AutoModelForCausalLM,
    submodules: dict[int, torch.nn.Module],
    inputs_BL: dict[str, torch.Tensor],
    min_offset: int | None,
    max_offset: int | None,
) -> dict[int, torch.Tensor]:
    if min_offset is not None:
        assert max_offset is not None, (
            "max_offset must be provided if min_offset is provided"
        )
        assert max_offset < min_offset, "max_offset must be less than min_offset"
        assert min_offset < 0, "min_offset must be less than 0"
        assert max_offset < 0, "max_offset must be less than 0"
    else:
        assert max_offset is None, (
            "max_offset must not be provided if min_offset is not provided"
        )

    activations_BLD_by_layer = {}

    module_to_layer = {submodule: layer for layer, submodule in submodules.items()}

    max_layer = max(submodules.keys())

    def gather_target_act_hook(module, inputs, outputs):
        layer = module_to_layer[module]

        if isinstance(outputs, tuple):
            activations_BLD_by_layer[layer] = outputs[0]
        else:
            activations_BLD_by_layer[layer] = outputs

        if min_offset is not None:
            activations_BLD_by_layer[layer] = activations_BLD_by_layer[layer][
                :, max_offset:min_offset, :
            ]

        if layer == max_layer:
            raise EarlyStopException("Early stopping after capturing activations")

    handles = []

    for layer, submodule in submodules.items():
        handles.append(submodule.register_forward_hook(gather_target_act_hook))

    try:
        # Use the selected context manager
        with torch.no_grad():
            _ = model(**inputs_BL)
    except EarlyStopException:
        pass
    except Exception as e:
        print(f"Unexpected error during forward pass: {str(e)}")
        raise
    finally:
        for handle in handles:
            handle.remove()

    return activations_BLD_by_layer


def get_text_config(model_or_config):
    """Return the config object that holds text-model attributes.

    Multimodal models (Gemma 3/4) nest text params (`num_hidden_layers`,
    `hidden_size`, `max_position_embeddings`, ...) under `config.text_config`.
    Flat decoder configs (Qwen, Llama, Mistral, Gemma 2) expose them directly
    on `config`. This helper returns whichever object carries them so callers
    don't have to branch per architecture. Accepts either a loaded model or
    a `PretrainedConfig` directly.
    """
    cfg = getattr(model_or_config, "config", model_or_config)
    return getattr(cfg, "text_config", cfg)


def resolve_oracle_layers(
    text_cfg,
    layer_percents: list[int] | None = None,
) -> tuple[int, list[int]]:
    """Pick the read/inject layers the activation oracle was trained on.

    Mirrors the rule used by the AO trainer in `nl_probes/gemma4_sft.py`:

    * For models exposing `layer_types` with at least one `"full_attention"`
      entry (Gemma 4's mixed sliding/full attention stack), each requested
      percent is snapped to the nearest full-attention layer, and the integer
      percent that maps to that layer via `int(num_layers * p / 100)` is
      returned. The injection layer is the first full-attention layer.
    * For every other architecture (Qwen, Llama, Gemma 2, Gemma 3 text-only,
      Mistral, ...) the percents pass through unchanged and the injection
      layer defaults to 1, matching the historical AO recipe.

    This is the single rule shared between training and inference: as long as
    the trainer used the same rule, pao reads from the layer the oracle was
    actually trained on.
    """
    if layer_percents is None:
        layer_percents = [25, 50, 75]

    layer_types = getattr(text_cfg, "layer_types", None)
    full = (
        [i for i, t in enumerate(layer_types) if t == "full_attention"]
        if layer_types
        else []
    )

    # No mixed attention pattern -> historical AO defaults apply.
    if not full or len(full) == len(layer_types):
        return 1, list(layer_percents)

    n = text_cfg.num_hidden_layers
    corrected: list[int] = []
    for p in layer_percents:
        target = n * (p / 100)
        layer = min(full, key=lambda l: (abs(l - target), l))
        # Back-solve the integer percent q in [1, 99] that satisfies
        # int(n * q / 100) == layer, picking the q closest to the original
        # request. There is always at least one such q for n >= 100/(layer+1).
        candidates = [q for q in range(1, 100) if int(n * (q / 100)) == layer]
        if not candidates:
            # Fallback: keep the original percent. Should not happen for any
            # Gemma 4 variant currently shipped.
            corrected.append(p)
        else:
            corrected.append(min(candidates, key=lambda q: (abs(q - p), q)))
    return full[0], corrected


def get_hf_submodule(model: AutoModelForCausalLM, layer: int, use_lora: bool = False):
    """Gets the residual stream submodule for HF transformers"""
    model_name = model.config._name_or_path

    if use_lora:
        if "pythia" in model_name:
            raise ValueError("Need to determine how to get submodule for LoRA")
        elif "gemma-3" in model_name:
            return model.base_model.language_model.layers[layer]
        elif "gemma-4" in model_name:
            # PeftModel -> base_model.model is the original HF model. AutoModelForCausalLM
            # may load either Gemma4ForConditionalGeneration (multimodal; text sits under
            # .model.language_model) or Gemma4ForCausalLM (text-only; .model is the
            # Gemma4TextModel itself). Probe for .language_model to handle both.
            inner = model.base_model.model.model
            text_model = getattr(inner, "language_model", inner)
            return text_model.layers[layer]
        elif (
            "gemma-2" in model_name
            or "mistral" in model_name
            or "Llama" in model_name
            or "Qwen" in model_name
        ):
            return model.base_model.model.model.layers[layer]
        else:
            raise ValueError(f"Please add submodule for model {model_name}")

    if "pythia" in model_name:
        return model.gpt_neox.layers[layer]
    elif "gemma-3" in model_name:
        return model.language_model.layers[layer]
    elif "gemma-4" in model_name:
        # See the LoRA branch above: Gemma4ForConditionalGeneration nests a
        # Gemma4TextModel under .model.language_model, while Gemma4ForCausalLM has
        # the text model directly at .model. Probe so we don't care which one
        # AutoModelForCausalLM instantiated.
        inner = model.model
        text_model = getattr(inner, "language_model", inner)
        return text_model.layers[layer]
    elif (
        "gemma-2" in model_name
        or "mistral" in model_name
        or "Llama" in model_name
        or "Qwen" in model_name
    ):
        return model.model.layers[layer]
    else:
        raise ValueError(f"Please add submodule for model {model_name}")


# ---------------------------------------------------------------------------
# Activation steering hook
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def add_hook(
    module: torch.nn.Module,
    hook: Callable,
):
    """Temporarily adds a forward hook to a model module.

    Args:
        module: The PyTorch module to hook
        hook: The hook function to apply

    Yields:
        None: Used as a context manager

    Example:
        with add_hook(model.layer, hook_fn):
            output = model(input)
    """
    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def get_hf_activation_steering_hook(
    vectors: list[torch.Tensor],  # len B, each tensor is (K_b, d_model)
    positions: list[list[int]],  # len B, each list has length K_b
    steering_coefficient: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Callable:
    """
    HF hook with debug prints to compare against vLLM.
    Supports a variable number of target positions per batch element.

    Semantics:
      For each batch item b and slot k, replace the residual at token index positions[b][k]
      with normalize(vectors[b][k]) * ||resid[b, positions[b][k], :]|| * steering_coefficient.

    We use a for loop instead of vectorized operations as it's simpler and we are just doing indexing in
    a single layer, so the simplicity won out for now.
    """

    # ---- move inputs to device and prepare ragged tensors ----
    assert len(vectors) == len(positions), (
        "vectors and positions must have same batch length"
    )
    B = len(vectors)
    if B == 0:
        raise ValueError("Empty batch")

    # Pre-normalize once; we never backprop through these
    normed_list = [
        torch.nn.functional.normalize(v_b, dim=-1).detach() for v_b in vectors
    ]

    def hook_fn(module, _input, output):
        # Normalize output API across model families
        if isinstance(output, tuple):
            resid_BLD, *rest = output
            output_is_tuple = True
        else:
            resid_BLD = output
            output_is_tuple = False

        B_actual, L, d_model_actual = resid_BLD.shape
        if B_actual != B:
            raise ValueError(
                f"Batch mismatch: module B={B_actual}, provided vectors B={B}"
            )

        # Only touch the prompt forward pass
        if L <= 1:
            return (resid_BLD, *rest) if output_is_tuple else resid_BLD

        # Per-batch element work. Vectorized over K_b where safe.
        for b in range(B):
            pos_b = positions[b]
            pos_b = torch.tensor(pos_b, dtype=torch.long, device=device)
            assert pos_b.min() >= 0
            assert pos_b.max() < L
            # Gather original activations at requested slots and compute norms
            orig_KD = resid_BLD[b, pos_b, :]  # (K_b, d)
            norms_K1 = orig_KD.norm(dim=-1, keepdim=True)  # (K_b, 1)

            if b == 0:
                if norms_K1.max() > 300:
                    print(
                        f"\n\n\n\n\nWARNING: Large norm detected in batch! {norms_K1}\n\n\n\n\n"
                    )

            # Build steered vectors for this b
            steered_KD = (normed_list[b] * norms_K1 * steering_coefficient).to(
                dtype
            )  # (K_b, d)

            resid_BLD[b, pos_b, :] = steered_KD.detach() + orig_KD

        return (resid_BLD, *rest) if output_is_tuple else resid_BLD

    return hook_fn


__all__ = [
    "SPECIAL_TOKEN",
    "EarlyStopException",
    "PeftModel",
    "add_hook",
    "collect_activations_multiple_layers",
    "encode_messages",
    "find_pattern_in_tokens",
    "get_hf_activation_steering_hook",
    "get_hf_submodule",
    "get_introspection_prefix",
    "get_text_config",
    "resolve_oracle_layers",
    "resolve_attention_implementation",
    "load_lora_adapter",
    "load_model",
    "load_tokenizer",
    "sanitize_lora_name",
    "set_seed",
]
