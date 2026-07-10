"""Unified preflight diagnostics for the taboo activation-oracle pipeline.

This is the single entrypoint to run *before* a full ``run_taboo_uq`` sweep.
It folds the existing standalone diagnostics into one staged script:

1. ``validate_setup`` gates the run with the cheap end-to-end health checks.
2. If those checks fail, the script escalates automatically into deeper probes.
3. ``--profile deep`` runs the full battery regardless of the early results.

Examples:

    uv run python -m pao.experiments.preflight_taboo --preset qwen3-8b
    uv run python -m pao.experiments.preflight_taboo --preset gemma-4-e2b --profile deep
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from pao.config import (
    AO_ROOT,
    MODEL_PRESETS,
    TABOO_WORDS,
    VERBALIZER_PROMPTS_TABOO,
    ModelConfig,
)
from pao.experiments._preflight_support import (
    _base_segment_positions,
    _generate_text,
    _segment_positions as target_segment_positions,
    check_activation_contract,
    check_oracle_accuracy,
    check_taboo_hiding,
    count_lora_modules,
    load_adapter_report,
    load_expected_module_names,
    patch_and_read,
    probe_layer_output_is_tuple,
    run_case,
    run_logitlens_one,
    run_transport_one,
    trace_candidate_lens,
    trace_candidate_matrix,
    trace_single_hidden,
)
from pao.experiments.run_taboo_uq import prepare_activation_and_sampler, setup_model
from pao.hf_utils import (
    encode_messages,
    get_introspection_prefix,
    get_text_config,
    load_lora_adapter,
)


@dataclass
class StageReport:
    name: str
    status: str
    summary: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreflightReport:
    preset: str
    profile: str
    base_model: str
    verbalizer_lora: str
    target_lora_template: str
    words: list[str]
    focus_word: str
    context_source: str
    stages: list[StageReport]
    recommendation: str
    ready_for_full_run: bool


def _status_tag(status: str) -> str:
    return {
        "pass": "PASS",
        "warn": "WARN",
        "fail": "FAIL",
        "skip": "SKIP",
    }.get(status, status.upper())


def _print_stage(report: StageReport) -> None:
    print(f"[{_status_tag(report.status)}] {report.name}: {report.summary}")


def _safe_stage(name: str, fn) -> StageReport:
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        return StageReport(
            name=name,
            status="fail",
            summary=f"{type(exc).__name__}: {exc}",
            details={"exception": repr(exc)},
        )


def _load_context_prompts(source: str) -> list[str]:
    path = AO_ROOT / f"datasets/taboo/taboo_{source}.txt"
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def _default_words() -> list[str]:
    middle = len(TABOO_WORDS) // 2
    return [TABOO_WORDS[0], TABOO_WORDS[middle], TABOO_WORDS[-1]]


def _dedupe_words(words: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for word in words:
        if word in seen:
            continue
        seen.add(word)
        out.append(word)
    return out


def _expected_lora_module_count(text_cfg) -> int:
    kv_shared = getattr(text_cfg, "num_kv_shared_layers", 0)
    return text_cfg.num_hidden_layers * 7 - kv_shared * 2


def _run_validation_stage(
    model,
    tokenizer,
    device: torch.device,
    cfg: ModelConfig,
    verbalizer_adapter: str | None,
    words: list[str],
    contexts: list[str],
    verb_prompts: list[str],
) -> StageReport:
    activation_ok = check_activation_contract(
        model=model,
        tokenizer=tokenizer,
        device=device,
        cfg=cfg,
        context_prompt=contexts[0],
    )
    hiding_ok = check_taboo_hiding(
        model=model,
        tokenizer=tokenizer,
        device=device,
        cfg=cfg,
        words=words,
    )
    if verbalizer_adapter is not None:
        model.set_adapter(verbalizer_adapter)
    accuracy_ok = check_oracle_accuracy(
        model=model,
        tokenizer=tokenizer,
        device=device,
        cfg=cfg,
        verbalizer_adapter=verbalizer_adapter,
        words=words,
        context_prompts=contexts[: min(2, len(contexts))],
        verbalizer_prompts=verb_prompts[: min(2, len(verb_prompts))],
    )

    checks = {
        "activation_contract": activation_ok,
        "taboo_hiding": hiding_ok,
        "oracle_accuracy": accuracy_ok,
    }
    passed = sum(int(ok) for ok in checks.values())
    status = "pass" if all(checks.values()) else "fail"
    return StageReport(
        name="validate_setup",
        status=status,
        summary=f"{passed}/{len(checks)} gating checks passed",
        details=checks,
    )


def _coverage_row(adapter_repo: str, expected_modules: set[str]) -> dict[str, Any]:
    adapter_config, observed_modules, key_counts = load_adapter_report(adapter_repo)
    matched = expected_modules & observed_modules
    missing = expected_modules - observed_modules
    extra = observed_modules - expected_modules
    return {
        "adapter_repo": adapter_repo,
        "target_modules": adapter_config.get("target_modules", []),
        "coverage": len(matched) / max(len(expected_modules), 1),
        "expected_count": len(expected_modules),
        "matched_count": len(matched),
        "missing_count": len(missing),
        "extra_count": len(extra),
        "tensor_key_counts": key_counts,
    }


def _run_coverage_stage(cfg: ModelConfig, words: list[str]) -> StageReport:
    expected_modules = set(load_expected_module_names(cfg.model_name))
    rows = [_coverage_row(cfg.verbalizer_lora_path, expected_modules)]
    rows.extend(
        _coverage_row(cfg.target_lora_template.format(word=word), expected_modules)
        for word in words
    )

    min_coverage = min(row["coverage"] for row in rows)
    any_missing = any(row["missing_count"] > 0 for row in rows)
    any_extra = any(row["extra_count"] > 0 for row in rows)

    if not any_missing and not any_extra:
        status = "pass"
    elif min_coverage >= 0.9:
        status = "warn"
    else:
        status = "fail"

    coverage_str = ", ".join(
        f"{Path(row['adapter_repo']).name}:{row['coverage']:.0%}" for row in rows
    )
    return StageReport(
        name="adapter_coverage",
        status=status,
        summary=f"adapter/module coverage = {coverage_str}",
        details={
            "expected_module_count": len(expected_modules),
            "adapters": rows,
        },
    )


def _run_gemma_surface_stage(
    model,
    tokenizer,
    device: torch.device,
    cfg: ModelConfig,
    verbalizer_adapter: str | None,
    focus_word: str,
    contexts: list[str],
    verb_prompts: list[str],
) -> StageReport:
    model_name = cfg.model_name.lower()
    if "gemma-4" not in model_name:
        return StageReport(
            name="gemma_surface_probe",
            status="skip",
            summary="Gemma 4-only probe skipped for non-Gemma preset",
        )

    text_cfg = get_text_config(model)
    expected_modules = _expected_lora_module_count(text_cfg)
    target_adapter = load_lora_adapter(
        model, cfg.target_lora_template.format(word=focus_word)
    )

    verbalizer_count = (
        count_lora_modules(model, verbalizer_adapter) if verbalizer_adapter else 0
    )
    target_count = count_lora_modules(model, target_adapter)

    act_layer = int(text_cfg.num_hidden_layers * (cfg.selected_layer_percent / 100))
    oracle_user_content = (
        get_introspection_prefix(act_layer, abs(cfg.segment_start_idx))
        + verb_prompts[0]
    )
    oracle_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": oracle_user_content}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors=None,
        return_dict=False,
        padding=False,
        enable_thinking=False,
    )

    if verbalizer_adapter is not None:
        model.set_adapter(verbalizer_adapter)
    trivial_generation = _generate_text(
        model, tokenizer, device, "What is 2 + 2?", max_new_tokens=20
    )
    oracle_generation_no_steer = _generate_text(
        model, tokenizer, device, oracle_user_content, max_new_tokens=10
    )

    sampler, oracle_ids_pipeline, _ = prepare_activation_and_sampler(
        model=model,
        tokenizer=tokenizer,
        device=device,
        cfg=cfg,
        target_adapter=target_adapter,
        verbalizer_adapter=verbalizer_adapter,
        context_prompt=contexts[0],
        verbalizer_prompt=verb_prompts[0],
    )
    with sampler:
        steered_generation = sampler.generate_batch_texts(
            context=oracle_ids_pipeline,
            temperature=0.0,
            max_new_tokens=15,
            num_samples=1,
            do_sample=False,
        )[0]

    module_ratio = min(verbalizer_count, target_count) / max(expected_modules, 1)
    if (
        module_ratio >= 0.9
        and steered_generation.strip()
        and steered_generation != oracle_generation_no_steer
    ):
        status = "pass"
    elif module_ratio >= 0.9:
        status = "warn"
    else:
        status = "fail"

    return StageReport(
        name="gemma_surface_probe",
        status=status,
        summary=(
            f"verbalizer_modules={verbalizer_count}/{expected_modules}, "
            f"target_modules={target_count}/{expected_modules}"
        ),
        details={
            "expected_modules": expected_modules,
            "verbalizer_module_count": verbalizer_count,
            "target_module_count": target_count,
            "oracle_prompt_tokens": len(oracle_ids),
            "oracle_prompt_tail": tokenizer.decode(oracle_ids[-20:]),
            "trivial_generation": trivial_generation,
            "oracle_generation_no_steer": oracle_generation_no_steer,
            "steered_generation": steered_generation,
        },
    )


def _run_oracle_modes_stage(
    model,
    tokenizer,
    device: torch.device,
    cfg: ModelConfig,
    verbalizer_adapter: str | None,
    focus_word: str,
    contexts: list[str],
    verb_prompts: list[str],
    modes: list[str],
) -> StageReport:
    target_adapter = load_lora_adapter(
        model, cfg.target_lora_template.format(word=focus_word)
    )
    rows_by_mode: dict[str, list[dict[str, Any]]] = {mode: [] for mode in modes}

    for context_index, context_prompt in enumerate(contexts):
        inputs_BL = encode_messages(
            tokenizer=tokenizer,
            message_dicts=[[{"role": "user", "content": context_prompt}]],
            add_generation_prompt=True,
            enable_thinking=False,
            device=device,
        )
        seq_len = int(inputs_BL["attention_mask"][0].sum().item())
        base_positions = _base_segment_positions(seq_len, cfg)
        position_map = {
            "segment": base_positions,
            "full_seq": list(range(seq_len)),
            "single_last": [base_positions[-1]],
        }

        for verbalizer_prompt_index, verbalizer_prompt in enumerate(verb_prompts):
            for mode in modes:
                positions = position_map[mode]
                c0 = run_case(
                    mode=mode,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    cfg=cfg,
                    target_adapter=target_adapter,
                    verbalizer_adapter=verbalizer_adapter,
                    context_prompt=context_prompt,
                    verbalizer_prompt=verbalizer_prompt,
                    candidate_words=TABOO_WORDS,
                    target_word=focus_word,
                    target_positions_rel=positions,
                    context_index=context_index,
                    verbalizer_prompt_index=verbalizer_prompt_index,
                    coefficient=0.0,
                )
                c1 = run_case(
                    mode=mode,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    cfg=cfg,
                    target_adapter=target_adapter,
                    verbalizer_adapter=verbalizer_adapter,
                    context_prompt=context_prompt,
                    verbalizer_prompt=verbalizer_prompt,
                    candidate_words=TABOO_WORDS,
                    target_word=focus_word,
                    target_positions_rel=positions,
                    context_index=context_index,
                    verbalizer_prompt_index=verbalizer_prompt_index,
                    coefficient=1.0,
                )
                rows_by_mode[mode].append(
                    {
                        "context_index": context_index,
                        "verbalizer_prompt_index": verbalizer_prompt_index,
                        "rank_c0": c0.target_rank,
                        "rank_c1": c1.target_rank,
                        "delta": c0.target_rank - c1.target_rank,
                        "greedy": c1.greedy,
                        "top_word": c1.top_word,
                    }
                )

    stats: dict[str, dict[str, Any]] = {}
    for mode, rows in rows_by_mode.items():
        stats[mode] = {
            "mean_rank_c0": mean(row["rank_c0"] for row in rows),
            "mean_rank_c1": mean(row["rank_c1"] for row in rows),
            "mean_delta": mean(row["delta"] for row in rows),
            "top1_c1": sum(int(row["rank_c1"] == 1) for row in rows),
            "n_cases": len(rows),
            "sample_output": rows[0]["greedy"] if rows else "",
        }

    best_mode = min(stats, key=lambda mode: stats[mode]["mean_rank_c1"])
    best = stats[best_mode]
    any_improvement = any(row["mean_delta"] > 0 for row in stats.values())
    if best["mean_rank_c1"] <= 3 or best["top1_c1"] > 0:
        status = "pass"
    elif any_improvement:
        status = "warn"
    else:
        status = "fail"

    return StageReport(
        name="oracle_modes",
        status=status,
        summary=(
            f"best={best_mode} mean_rank={best['mean_rank_c1']:.2f} "
            f"delta={best['mean_delta']:+.2f}"
        ),
        details={
            "focus_word": focus_word,
            "mode_stats": stats,
            "best_mode": best_mode,
        },
    )


def _run_target_encoding_stage(
    model,
    tokenizer,
    device: torch.device,
    cfg: ModelConfig,
    focus_word: str,
    contexts: list[str],
) -> StageReport:
    try:
        from nnsight import LanguageModel
    except Exception as exc:  # noqa: BLE001
        return StageReport(
            name="target_encoding",
            status="skip",
            summary=f"nnsight unavailable: {exc}",
        )

    nn_model = LanguageModel(model, tokenizer=tokenizer)
    layer_output_is_tuple = probe_layer_output_is_tuple(model, tokenizer, device)
    target_adapter = load_lora_adapter(
        model, cfg.target_lora_template.format(word=focus_word)
    )
    candidate_words = [focus_word] + [
        word for word in TABOO_WORDS if word != focus_word
    ]

    base_ranks: list[int] = []
    target_ranks: list[int] = []
    per_context: list[dict[str, Any]] = []

    for context_index, context_prompt in enumerate(contexts):
        inputs_BL = encode_messages(
            tokenizer=tokenizer,
            message_dicts=[[{"role": "user", "content": context_prompt}]],
            add_generation_prompt=True,
            enable_thinking=False,
            device=device,
        )
        seq_len = int(inputs_BL["attention_mask"][0].sum().item())
        segment_positions = target_segment_positions(seq_len, cfg)
        token_ids = inputs_BL["input_ids"][0, :seq_len].tolist()

        model.set_adapter("default")
        base_summary = trace_candidate_lens(
            nn_model=nn_model,
            model=model,
            tokenizer=tokenizer,
            device=device,
            input_ids=token_ids,
            segment_positions=segment_positions,
            candidate_words=candidate_words,
            layer_output_is_tuple=layer_output_is_tuple,
        )

        model.set_adapter(target_adapter)
        target_summary = trace_candidate_lens(
            nn_model=nn_model,
            model=model,
            tokenizer=tokenizer,
            device=device,
            input_ids=token_ids,
            segment_positions=segment_positions,
            candidate_words=candidate_words,
            layer_output_is_tuple=layer_output_is_tuple,
        )

        base_ranks.append(base_summary.best.target_rank)
        target_ranks.append(target_summary.best.target_rank)
        per_context.append(
            {
                "context_index": context_index,
                "base_best_rank": base_summary.best.target_rank,
                "target_best_rank": target_summary.best.target_rank,
                "base_best_layer": base_summary.best.layer,
                "target_best_layer": target_summary.best.layer,
            }
        )

    mean_base = mean(base_ranks)
    mean_target = mean(target_ranks)
    target_top1 = sum(int(rank == 1) for rank in target_ranks)

    if mean_target < mean_base or target_top1 > 0:
        status = "pass"
    elif mean_target == mean_base:
        status = "warn"
    else:
        status = "fail"

    return StageReport(
        name="target_encoding",
        status=status,
        summary=(
            f"mean best-rank base->{mean_base:.2f}, target->{mean_target:.2f}; "
            f"target_top1={target_top1}/{len(target_ranks)}"
        ),
        details={
            "focus_word": focus_word,
            "mean_base_best_rank": mean_base,
            "mean_target_best_rank": mean_target,
            "target_top1_contexts": target_top1,
            "contexts": per_context,
        },
    )


def _run_layer_transport_stage(
    model,
    tokenizer,
    device: torch.device,
    cfg: ModelConfig,
    verbalizer_adapter: str | None,
    focus_word: str,
    contexts: list[str],
    verb_prompts: list[str],
    read_layer_percents: list[int],
    injection_layers: list[int],
) -> StageReport:
    original_read = cfg.selected_layer_percent
    original_injection = cfg.injection_layer
    target_adapter = load_lora_adapter(
        model, cfg.target_lora_template.format(word=focus_word)
    )
    results: list[dict[str, Any]] = []

    try:
        for read_percent in read_layer_percents:
            for injection_layer in injection_layers:
                cfg.selected_layer_percent = read_percent
                cfg.injection_layer = injection_layer
                correct = 0
                total = 0
                for verbalizer_prompt in verb_prompts[:1]:
                    for context_prompt in contexts:
                        _, _, is_correct = run_transport_one(
                            model=model,
                            tokenizer=tokenizer,
                            device=device,
                            cfg=cfg,
                            target_adapter=target_adapter,
                            verbalizer_adapter=verbalizer_adapter,
                            context_prompt=context_prompt,
                            verbalizer_prompt=verbalizer_prompt,
                            target_word=focus_word,
                            max_new_tokens=15,
                        )
                        correct += int(is_correct)
                        total += 1
                results.append(
                    {
                        "read_layer_percent": read_percent,
                        "injection_layer": injection_layer,
                        "correct": correct,
                        "total": total,
                        "accuracy": correct / max(total, 1),
                    }
                )
    finally:
        cfg.selected_layer_percent = original_read
        cfg.injection_layer = original_injection

    best = max(results, key=lambda row: (row["accuracy"], row["correct"]))
    current = next(
        (
            row
            for row in results
            if row["read_layer_percent"] == original_read
            and row["injection_layer"] == original_injection
        ),
        None,
    )

    if best["accuracy"] == 0.0:
        status = "fail"
    elif current is not None and best["accuracy"] > current["accuracy"]:
        status = "warn"
    else:
        status = "pass"

    return StageReport(
        name="layer_transport",
        status=status,
        summary=(
            f"best read={best['read_layer_percent']}% inject=L{best['injection_layer']} "
            f"acc={best['accuracy']:.0%}"
        ),
        details={
            "current_combo": {
                "read_layer_percent": original_read,
                "injection_layer": original_injection,
            },
            "best_combo": {
                "read_layer_percent": best["read_layer_percent"],
                "injection_layer": best["injection_layer"],
                "accuracy": best["accuracy"],
            },
            "results": results,
        },
    )


def _run_adapter_contrast_stage(
    model,
    tokenizer,
    device: torch.device,
    cfg: ModelConfig,
    focus_word: str,
    words: list[str],
    contexts: list[str],
) -> StageReport:
    try:
        from nnsight import LanguageModel
    except Exception as exc:  # noqa: BLE001
        return StageReport(
            name="adapter_contrast",
            status="skip",
            summary=f"nnsight unavailable: {exc}",
        )

    candidate_words = [focus_word] + [word for word in words if word != focus_word][:2]
    candidate_words = _dedupe_words(candidate_words)
    if len(candidate_words) < 2:
        return StageReport(
            name="adapter_contrast",
            status="skip",
            summary="need at least two candidate words for adapter contrast",
        )

    adapter_names = {
        word: load_lora_adapter(model, cfg.target_lora_template.format(word=word))
        for word in candidate_words
    }
    nn_model = LanguageModel(model, tokenizer=tokenizer)
    layer_output_is_tuple = probe_layer_output_is_tuple(model, tokenizer, device)

    improvement_counts = {word: 0 for word in candidate_words}
    patch_improvements: list[float] = []

    for context_prompt in contexts:
        inputs_BL = encode_messages(
            tokenizer=tokenizer,
            message_dicts=[[{"role": "user", "content": context_prompt}]],
            add_generation_prompt=True,
            enable_thinking=False,
            device=device,
        )
        seq_len = int(inputs_BL["attention_mask"][0].sum().item())
        segment_positions = target_segment_positions(seq_len, cfg)
        token_ids = inputs_BL["input_ids"][0, :seq_len].tolist()

        model.disable_adapters()
        base = trace_candidate_matrix(
            nn_model=nn_model,
            model=model,
            tokenizer=tokenizer,
            device=device,
            input_ids=token_ids,
            segment_positions=segment_positions,
            candidate_words=candidate_words,
            layer_output_is_tuple=layer_output_is_tuple,
            adapter_label="base",
        )
        model.enable_adapters()

        contrast_by_adapter = {"base": base}
        for word in candidate_words:
            model.set_adapter(adapter_names[word])
            contrast = trace_candidate_matrix(
                nn_model=nn_model,
                model=model,
                tokenizer=tokenizer,
                device=device,
                input_ids=token_ids,
                segment_positions=segment_positions,
                candidate_words=candidate_words,
                layer_output_is_tuple=layer_output_is_tuple,
                adapter_label=word,
            )
            contrast_by_adapter[word] = contrast
            if (
                contrast.best_by_word[word].target_rank
                < base.best_by_word[word].target_rank
            ):
                improvement_counts[word] += 1

        source_best = contrast_by_adapter[focus_word].best_by_word[focus_word]
        model.set_adapter(adapter_names[focus_word])
        source_hidden = trace_single_hidden(
            nn_model=nn_model,
            model=model,
            device=device,
            input_ids=token_ids,
            layer_idx=source_best.layer,
            position=source_best.position,
            layer_output_is_tuple=layer_output_is_tuple,
        )
        for dest_word in candidate_words:
            if dest_word == focus_word:
                continue
            model.set_adapter(adapter_names[dest_word])
            patched = patch_and_read(
                nn_model=nn_model,
                model=model,
                tokenizer=tokenizer,
                device=device,
                input_ids=token_ids,
                patch_layer=source_best.layer,
                patch_position=source_best.position,
                source_hidden=source_hidden,
                candidate_words=candidate_words,
                source_word=focus_word,
                layer_output_is_tuple=layer_output_is_tuple,
            )
            patch_improvements.append(
                patched.source_prob_after - patched.source_prob_before
            )

    focus_improvements = improvement_counts[focus_word]
    if focus_improvements > 0:
        status = "pass"
    elif patch_improvements and mean(patch_improvements) > 0:
        status = "warn"
    else:
        status = "fail"

    return StageReport(
        name="adapter_contrast",
        status=status,
        summary=(
            f"focus adapter improved in {focus_improvements}/{len(contexts)} contexts; "
            f"mean_patch_delta={mean(patch_improvements) if patch_improvements else 0.0:.4f}"
        ),
        details={
            "candidate_words": candidate_words,
            "improvement_counts": improvement_counts,
            "mean_patch_improvement": mean(patch_improvements)
            if patch_improvements
            else 0.0,
        },
    )


def _run_scale_logitlens_stage(
    model,
    tokenizer,
    device: torch.device,
    cfg: ModelConfig,
    verbalizer_adapter: str | None,
    words: list[str],
    contexts: list[str],
    verb_prompts: list[str],
) -> StageReport:
    try:
        from nnsight import LanguageModel
    except Exception as exc:  # noqa: BLE001
        return StageReport(
            name="scale_logitlens",
            status="skip",
            summary=f"nnsight unavailable: {exc}",
        )

    nn_model = LanguageModel(model, tokenizer=tokenizer)
    layer_output_is_tuple = probe_layer_output_is_tuple(model, tokenizer, device)

    correct = 0
    total = 0
    lens_best_ranks: list[int] = []
    lens_top1 = 0
    lens_top10 = 0

    for word in words:
        target_adapter = load_lora_adapter(
            model, cfg.target_lora_template.format(word=word)
        )
        for verbalizer_prompt in verb_prompts[:1]:
            for context_prompt in contexts:
                res = run_logitlens_one(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    cfg=cfg,
                    target_adapter=target_adapter,
                    verbalizer_adapter=verbalizer_adapter,
                    context_prompt=context_prompt,
                    verbalizer_prompt=verbalizer_prompt,
                    target_word=word,
                    max_new_tokens=15,
                    nn_model=nn_model,
                    run_lens=True,
                    lens_topk=8,
                    layer_output_is_tuple=layer_output_is_tuple,
                )
                total += 1
                correct += int(res.is_correct)
                if res.lens is None:
                    continue
                best_rank = min(res.lens.target_rank_per_layer)
                lens_best_ranks.append(best_rank)
                lens_top1 += int(best_rank == 0)
                lens_top10 += int(best_rank < 10)

    accuracy = correct / max(total, 1)
    if accuracy > 0 or lens_top1 > 0:
        status = "pass"
    elif lens_top10 > 0:
        status = "warn"
    else:
        status = "fail"

    return StageReport(
        name="scale_logitlens",
        status=status,
        summary=(
            f"accuracy={accuracy:.0%}, lens_top1={lens_top1}/{len(lens_best_ranks) or 1}, "
            f"mean_best_rank={mean(lens_best_ranks) if lens_best_ranks else -1:.2f}"
        ),
        details={
            "accuracy": accuracy,
            "total": total,
            "correct": correct,
            "lens_runs": len(lens_best_ranks),
            "lens_target_top1": lens_top1,
            "lens_target_top10": lens_top10,
            "mean_best_rank": mean(lens_best_ranks) if lens_best_ranks else None,
        },
    )


def _build_recommendation(stages: list[StageReport]) -> tuple[bool, str]:
    by_name = {stage.name: stage for stage in stages}

    validation = by_name.get("validate_setup")
    coverage = by_name.get("adapter_coverage")
    oracle_modes = by_name.get("oracle_modes")
    encoding = by_name.get("target_encoding")
    transport = by_name.get("layer_transport")

    has_fail = any(stage.status == "fail" for stage in stages)
    if not has_fail:
        return True, "Preflight looks healthy enough to start the full taboo run."

    if coverage is not None and coverage.status == "fail":
        return (
            False,
            "Adapter checkpoints do not cover the base architecture cleanly. Fix the LoRAs before spending a full taboo run.",
        )

    if encoding is not None and encoding.status == "fail":
        return (
            False,
            "The focus target LoRA is not clearly encoding the hidden word in the probed segment. The target adapters look like the bottleneck.",
        )

    if transport is not None and transport.status == "warn":
        best = transport.details.get("best_combo", {})
        return (
            False,
            "Transport looks like the bottleneck. Try "
            f"read={best.get('read_layer_percent')}% and inject=L{best.get('injection_layer')} before the full taboo run.",
        )

    if oracle_modes is not None and oracle_modes.status == "fail":
        return (
            False,
            "Steering is not helping the restricted-word ranking on the focus case. Inspect prompt layout or hook behavior before the full taboo run.",
        )

    if validation is not None and validation.status == "fail":
        return (
            False,
            "Core gating checks failed. Use the failed stages above to fix the pipeline before the full taboo run.",
        )

    return False, "Preflight found issues worth fixing before the full taboo run."


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified taboo preflight diagnostics")
    parser.add_argument("--preset", required=True, choices=sorted(MODEL_PRESETS))
    parser.add_argument(
        "--profile",
        choices=("quick", "standard", "deep"),
        default="standard",
        help="quick = validation only, standard = validation + escalation, deep = always run all probes",
    )
    parser.add_argument(
        "--context-source",
        choices=("direct_test", "standard_test", "direct_val", "standard_val"),
        default="direct_test",
    )
    parser.add_argument(
        "--context-limit",
        type=int,
        default=3,
        help="How many contexts to use in the deeper probes",
    )
    parser.add_argument(
        "--n-verbalizer-prompts",
        type=int,
        default=2,
        help="How many verbalizer prompts to use in the deeper probes",
    )
    parser.add_argument(
        "--words",
        nargs="+",
        default=None,
        help="Words used in the validation checks (default: first/middle/last)",
    )
    parser.add_argument(
        "--focus-word",
        default=None,
        choices=TABOO_WORDS,
        help="Single word used by the deeper diagnostic probes",
    )
    parser.add_argument(
        "--verbalizer-lora-path",
        default=None,
        help="Override the preset verbalizer LoRA",
    )
    parser.add_argument(
        "--attn-implementation",
        default="auto",
        help=(
            "Transformers attention backend. auto uses the fastest known-working "
            "backend per model family; examples: sdpa, flash_attention_2, "
            "flash_attention_3, kernels-community/flash-attn2."
        ),
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional JSON report path",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    cfg = ModelConfig.from_preset(args.preset)
    if args.verbalizer_lora_path is not None:
        cfg.verbalizer_lora_path = args.verbalizer_lora_path
    cfg.attn_implementation = args.attn_implementation

    words = _dedupe_words(args.words or _default_words())
    for word in words:
        if word not in TABOO_WORDS:
            raise ValueError(f"Unknown taboo word: {word}")
    focus_word = args.focus_word or words[0]

    os.chdir(AO_ROOT)
    all_contexts = _load_context_prompts(args.context_source)
    contexts = all_contexts[: args.context_limit]
    verb_prompts = VERBALIZER_PROMPTS_TABOO[: args.n_verbalizer_prompts]
    if not contexts:
        raise ValueError("No contexts selected; increase --context-limit")
    if not verb_prompts:
        raise ValueError(
            "No verbalizer prompts selected; increase --n-verbalizer-prompts"
        )

    model, tokenizer, device, verbalizer_adapter = setup_model(cfg)

    print(f"Preset: {args.preset}")
    print(f"  base:       {cfg.model_name}")
    print(f"  verbalizer: {cfg.verbalizer_lora_path}")
    print(f"  target:     {cfg.target_lora_template}")
    print(f"  profile:    {args.profile}")
    print(f"  words:      {words}")
    print(f"  focus_word: {focus_word}")
    print(f"  contexts:   {len(contexts)} from {args.context_source}")
    print(f"  verb_prompts: {len(verb_prompts)}")

    stages: list[StageReport] = []

    validation = _safe_stage(
        "validate_setup",
        lambda: _run_validation_stage(
            model=model,
            tokenizer=tokenizer,
            device=device,
            cfg=cfg,
            verbalizer_adapter=verbalizer_adapter,
            words=words,
            contexts=contexts,
            verb_prompts=verb_prompts,
        ),
    )
    stages.append(validation)
    _print_stage(validation)

    if args.profile != "quick":
        should_escalate = args.profile == "deep" or validation.status != "pass"

        if should_escalate or "gemma-4" in cfg.model_name.lower():
            coverage = _safe_stage(
                "adapter_coverage",
                lambda: _run_coverage_stage(cfg=cfg, words=words),
            )
            stages.append(coverage)
            _print_stage(coverage)

        if should_escalate:
            gemma_surface = _safe_stage(
                "gemma_surface_probe",
                lambda: _run_gemma_surface_stage(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    cfg=cfg,
                    verbalizer_adapter=verbalizer_adapter,
                    focus_word=focus_word,
                    contexts=contexts,
                    verb_prompts=verb_prompts,
                ),
            )
            stages.append(gemma_surface)
            _print_stage(gemma_surface)

            oracle_modes = _safe_stage(
                "oracle_modes",
                lambda: _run_oracle_modes_stage(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    cfg=cfg,
                    verbalizer_adapter=verbalizer_adapter,
                    focus_word=focus_word,
                    contexts=contexts,
                    verb_prompts=verb_prompts,
                    modes=["segment", "full_seq", "single_last"],
                ),
            )
            stages.append(oracle_modes)
            _print_stage(oracle_modes)

            target_encoding = _safe_stage(
                "target_encoding",
                lambda: _run_target_encoding_stage(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    cfg=cfg,
                    focus_word=focus_word,
                    contexts=contexts,
                ),
            )
            stages.append(target_encoding)
            _print_stage(target_encoding)

            should_sweep_transport = args.profile == "deep" or (
                target_encoding.status in {"pass", "warn"}
                and oracle_modes.status != "pass"
            )
            if should_sweep_transport:
                text_cfg = get_text_config(model)
                read_layer_percents_int = sorted(
                    set([cfg.selected_layer_percent, *cfg.layer_percents])
                )
                candidate_injections = [0, 1, 2, 4, cfg.injection_layer]
                injection_layers = sorted(
                    {
                        layer
                        for layer in candidate_injections
                        if 0 <= layer < text_cfg.num_hidden_layers
                    }
                )
                layer_transport = _safe_stage(
                    "layer_transport",
                    lambda: _run_layer_transport_stage(
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        cfg=cfg,
                        verbalizer_adapter=verbalizer_adapter,
                        focus_word=focus_word,
                        contexts=contexts[: min(2, len(contexts))],
                        verb_prompts=verb_prompts,
                        read_layer_percents=read_layer_percents_int,
                        injection_layers=injection_layers,
                    ),
                )
                stages.append(layer_transport)
                _print_stage(layer_transport)

        if args.profile == "deep":
            adapter_contrast = _safe_stage(
                "adapter_contrast",
                lambda: _run_adapter_contrast_stage(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    cfg=cfg,
                    focus_word=focus_word,
                    words=words,
                    contexts=contexts[: min(2, len(contexts))],
                ),
            )
            stages.append(adapter_contrast)
            _print_stage(adapter_contrast)

            scale_logitlens = _safe_stage(
                "scale_logitlens",
                lambda: _run_scale_logitlens_stage(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    cfg=cfg,
                    verbalizer_adapter=verbalizer_adapter,
                    words=words,
                    contexts=contexts,
                    verb_prompts=verb_prompts,
                ),
            )
            stages.append(scale_logitlens)
            _print_stage(scale_logitlens)

    ready_for_full_run, recommendation = _build_recommendation(stages)

    print(f"\n{'=' * 78}")
    print("  Preflight Summary")
    print(f"{'=' * 78}")
    for stage in stages:
        _print_stage(stage)
    print(f"\nRecommendation: {recommendation}")
    print(f"Ready for full taboo run: {'yes' if ready_for_full_run else 'no'}")

    report = PreflightReport(
        preset=args.preset,
        profile=args.profile,
        base_model=cfg.model_name,
        verbalizer_lora=cfg.verbalizer_lora_path,
        target_lora_template=cfg.target_lora_template,
        words=words,
        focus_word=focus_word,
        context_source=args.context_source,
        stages=stages,
        recommendation=recommendation,
        ready_for_full_run=ready_for_full_run,
    )
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(asdict(report), f, indent=2, default=str)
        print(f"wrote: {args.json_out}")

    return 0 if ready_for_full_run else 1


if __name__ == "__main__":
    sys.exit(main())
