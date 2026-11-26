"""RQ3 control: the five confidence methods on an unsteered chat model.

Same 20 secret words, same context prompts, same three questions, same answer
extraction and metrics as run_taboo_uq. The only change: nothing is injected.
The base chat model (no adapter) reads the target model's reply as text and
answers the question. A steering coefficient of 0 keeps every method code path
the same as in the steered runs, while the hook adds exactly zero.

    python -m pao.experiments.unsteered_control --preset qwen3-8b --words ship snow --max-prompts 100
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

from pao.answer_extraction import extract_predicted_word
from pao.calibration.secret_word_calibration import WordPrediction
from pao.config import (
    AO_ROOT,
    MODEL_PRESETS,
    TABOO_WORDS,
    VERBALIZER_PROMPTS_TABOO,
    ExperimentConfig,
    ModelConfig,
    SamplingConfig,
)
from pao.experiments.backfill_extra_methods import _load_checkpoint, _save_checkpoint
from pao.experiments.run_taboo_uq import evaluate_and_save, setup_experiment_state, temperature_tag
from pao.hf_utils import encode_messages, get_hf_submodule, get_text_config, load_lora_adapter
from pao.methods.direct_elicitation import direct_elicitation, score_linguistic_confidence
from pao.methods.forced_choice import forced_choice_confidence
from pao.methods.logprob_baseline import logprob_confidence
from pao.methods.temperature_bootstrap import temperature_bootstrap
from pao.oracle_sampler import SteeredAutoregressiveSampler

BASE_ADAPTER = "default"  # zero-initialized dummy LoRA added by setup_model
TEMPS = [0.7, 1.0, 1.3, 1.5]
REPLY_TOKENS = 128
LABEL_READOUTS = ("expected", "p_very_high", "p_high_plus")


def method_names(temps: list[float]) -> list[str]:
    return [
        "logprob_offset", "direct", *(f"direct_linguistic_{r}" for r in LABEL_READOUTS),
        *(f"bootstrap_t{temperature_tag(t)}" for t in temps), "forced_choice",
    ]


@torch.no_grad()
def target_replies(model, tokenizer, device, prompts: list[str]) -> list[str]:
    """Greedy replies of the active target adapter, one batch per prompt list."""
    tokenizer.padding_side = "left"
    inputs = encode_messages(tokenizer, [[{"role": "user", "content": p}] for p in prompts],
                             add_generation_prompt=True, enable_thinking=False, device=device)
    out = model.generate(**inputs, max_new_tokens=REPLY_TOKENS, do_sample=False)
    return tokenizer.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def cached_replies(cache: Path, model, tokenizer, device, word: str, prompts: list[str]) -> list[str]:
    store = json.loads(cache.read_text()) if cache.exists() else {}
    if word not in store:
        model.set_adapter(load_lora_adapter(model, MODEL_CFG.target_lora_template.format(word=word)))
        store[word] = [r for i in range(0, len(prompts), 25)
                       for r in target_replies(model, tokenizer, device, prompts[i:i + 25])]
        cache.write_text(json.dumps(store, indent=1))
    return store[word]


def guesser_messages(context_prompt: str, reply: str, question: str) -> list[dict[str, str]]:
    transcript = f"User: {context_prompt}\nAssistant: {reply.strip()}"
    return [{"role": "user", "content": f"{transcript}\n\n{question}"}]


def unsteered_sampler(model, tokenizer, device, dtype) -> SteeredAutoregressiveSampler:
    d_model = get_text_config(model).hidden_size
    return SteeredAutoregressiveSampler(
        model=model, tokenizer=tokenizer, device=device, submodule=get_hf_submodule(model, 1),
        steering_vectors=[torch.zeros(1, d_model, device=device)], positions=[[0]],
        steering_coefficient=0.0, dtype=dtype,
    )


def score_sample(sampler, messages, word: str, sampling: SamplingConfig, temps: list[float]):
    """Yield (method, answer, confidence, correct, metadata) for one sample."""
    tok = sampler.tokenizer
    ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors=None,
                                  return_dict=False, padding=False, enable_thinking=False)
    with sampler:
        lp = logprob_confidence(sampler, ids, TABOO_WORDS, sampling.max_new_tokens)
        yield ("logprob_offset", lp.generated_text, lp.word_prob_with_offset, lp.extracted_word == word,
               {"extracted_word": lp.extracted_word, "extracted_word_source": lp.extracted_word_source})
        el = direct_elicitation(sampler, ids, messages, max_new_tokens=sampling.max_new_tokens)
        el_word = extract_predicted_word(el.answer_text, TABOO_WORDS)
        yield ("direct", el.answer_text, el.parsed_confidence, el_word == word,
               {"extracted_word": el_word, "confidence_text": el.confidence_text,
                "confidence_decode_method": el.confidence_decode_method})
        lab = score_linguistic_confidence(sampler, messages, el.answer_text)
        for r in LABEL_READOUTS:
            yield (f"direct_linguistic_{r}", el.answer_text, getattr(lab, r if r != "expected" else "expected_value"),
                   el_word == word, {"extracted_word": el_word, "label_probs": lab.label_probs})
        for t in temps:
            b = temperature_bootstrap(sampler, ids, TABOO_WORDS, k=sampling.bootstrap_k, temperature=t,
                                      max_new_tokens=sampling.max_new_tokens)
            yield (f"bootstrap_t{temperature_tag(t)}", b.mode_answer, b.mode_frequency, b.mode_answer == word,
                   {"normalized_samples": b.normalized_samples, "num_unique": b.num_unique, "temperature": t})
        fc = forced_choice_confidence(sampler, ids, TABOO_WORDS)
        yield ("forced_choice", fc.predicted_word, fc.confidence, fc.predicted_word == word,
               {"word_probs": fc.word_probs})


def check_zero_hook(sampler, model, ids: list[int]) -> None:
    """The s=0 hook must give the same greedy tokens as plain generate."""
    with sampler:
        steered, _ = sampler.greedy_generate(ids, max_new_tokens=12)
    plain = model.generate(torch.tensor([ids], device=sampler.device), max_new_tokens=12, do_sample=False)[0].tolist()
    n = min(len(steered), len(plain))
    assert steered[:n] == plain[:n], "zero hook changed the output"


def main() -> None:
    global MODEL_CFG
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", required=True, choices=sorted(MODEL_PRESETS))
    p.add_argument("--words", nargs="+", default=TABOO_WORDS)
    p.add_argument("--context-file", default="datasets/taboo/taboo_direct_test.txt")
    p.add_argument("--max-prompts", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--attn-implementation", default="auto")
    p.add_argument("--checkpoint-every", type=int, default=25)
    args = p.parse_args()

    output_dir = Path(args.output_dir or f"results/{args.preset}/unsteered_control").resolve()
    os.chdir(AO_ROOT)
    MODEL_CFG = ModelConfig.from_preset(args.preset)
    MODEL_CFG.attn_implementation = args.attn_implementation
    cfg = ExperimentConfig(model=MODEL_CFG, sampling=SamplingConfig(), output_dir=str(output_dir),
                           context_prompt_file=args.context_file, max_context_prompts=args.max_prompts)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, device, _, prompts = setup_experiment_state(cfg)
    methods = method_names(TEMPS)
    preds, done = _load_checkpoint(output_dir / "checkpoint.json", methods)
    sampler = unsteered_sampler(model, tokenizer, device, MODEL_CFG.dtype)

    pbar = tqdm(total=len(args.words) * len(prompts) * len(VERBALIZER_PROMPTS_TABOO), initial=len(done))
    since = 0
    for word in args.words:
        replies = cached_replies(output_dir / "target_replies.json", model, tokenizer, device, word, prompts)
        model.set_adapter(BASE_ADAPTER)
        if not done:
            check_zero_hook(sampler, model, tokenizer.apply_chat_template(
                guesser_messages(prompts[0], replies[0], VERBALIZER_PROMPTS_TABOO[0]), tokenize=True,
                add_generation_prompt=True, return_tensors=None, return_dict=False, enable_thinking=False))
        for question in VERBALIZER_PROMPTS_TABOO:
            for ctx, reply in zip(prompts, replies):
                if (word, question, ctx) in done:
                    continue
                for m, ans, conf, ok, meta in score_sample(sampler, guesser_messages(ctx, reply, question),
                                                           word, cfg.sampling, TEMPS):
                    preds[m].append(WordPrediction(target_word=word, context_prompt=ctx, verbalizer_prompt=question,
                                                   predicted_answer=ans, confidence=float(conf), is_correct=ok,
                                                   method=m, method_metadata={**meta, "target_reply": reply}))
                done.add((word, question, ctx))
                pbar.update(1)
                since += 1
                if since >= args.checkpoint_every:
                    _save_checkpoint(output_dir / "checkpoint.json", preds, done)
                    since = 0
    _save_checkpoint(output_dir / "checkpoint.json", preds, done)
    evaluate_and_save(preds, str(output_dir), cfg.seed)


if __name__ == "__main__":
    main()
