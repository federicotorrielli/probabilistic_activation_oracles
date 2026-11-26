"""Seconds per sample of each confidence method on the steered oracle (cost table).

    python -m pao.experiments.time_methods --preset qwen3-8b --samples 50
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from statistics import mean

import torch

from pao.config import AO_ROOT, MODEL_PRESETS, TABOO_WORDS, VERBALIZER_PROMPTS_TABOO, ExperimentConfig, ModelConfig
from pao.experiments.run_taboo_uq import prepare_activation_and_sampler, setup_experiment_state
from pao.hf_utils import load_lora_adapter
from pao.methods.direct_elicitation import direct_elicitation, score_linguistic_confidence
from pao.methods.forced_choice import forced_choice_confidence
from pao.methods.logprob_baseline import logprob_confidence
from pao.methods.temperature_bootstrap import temperature_bootstrap


def timed(fn) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - start


def method_calls(sampler, ids, messages) -> dict:
    answer = {}

    def direct():
        answer["text"] = direct_elicitation(sampler, ids, messages).answer_text

    return {
        "log_probability": lambda: logprob_confidence(sampler, ids, TABOO_WORDS),
        "bootstrap_k20": lambda: temperature_bootstrap(sampler, ids, TABOO_WORDS, k=20, temperature=1.0),
        "direct_numeric": direct,
        "label_scoring": lambda: score_linguistic_confidence(sampler, messages, answer["text"]),
        "forced_choice": lambda: forced_choice_confidence(sampler, ids, TABOO_WORDS),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", required=True, choices=sorted(MODEL_PRESETS))
    p.add_argument("--samples", type=int, default=50)
    p.add_argument("--attn-implementation", default="auto")
    args = p.parse_args()
    out = Path(f"results/{args.preset}/timing.json").resolve()
    os.chdir(AO_ROOT)
    model_cfg = ModelConfig.from_preset(args.preset)
    model_cfg.attn_implementation = args.attn_implementation
    cfg = ExperimentConfig(model=model_cfg, max_context_prompts=args.samples)
    model, tokenizer, device, verbalizer, prompts = setup_experiment_state(cfg)
    target = load_lora_adapter(model, model_cfg.target_lora_template.format(word=TABOO_WORDS[0]))
    times: dict[str, list[float]] = {}
    for i, ctx in enumerate(prompts):
        sampler, ids, messages = prepare_activation_and_sampler(
            model, tokenizer, device, model_cfg, target, verbalizer, ctx, VERBALIZER_PROMPTS_TABOO[0])
        with sampler:
            for name, call in method_calls(sampler, ids, messages).items():
                t = timed(call)
                if i > 0:  # first sample warms up kernels
                    times.setdefault(name, []).append(t)
    summary = {k: mean(v) for k, v in times.items()}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"gpu": torch.cuda.get_device_name(), "seconds_per_sample": summary}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
