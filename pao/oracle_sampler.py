"""Bridge between Activation Oracle steering hooks and the power sampling framework.

This module provides SteeredAutoregressiveSampler, which wraps the
AutoregressiveSampler from reasoning-with-sampling with a persistent
activation steering hook from the AO codebase. This allows MCMC power
sampling to operate on activation-steered oracle models.
"""

import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from pao.hf_utils import get_hf_activation_steering_hook, get_text_config


class SteeredAutoregressiveSampler:
    """Autoregressive sampler with persistent activation steering.

    Wraps a HuggingFace model with an activation steering hook that fires
    on every forward pass. Compatible with the MCMC power sampling algorithms
    from reasoning-with-sampling.

    The hook's L <= 1 guard ensures it only modifies the prefill pass (full
    sequence), not individual token decoding steps. Each MCMC proposal triggers
    a fresh model.generate() call that re-prefills from gen[:idx], so the hook
    fires correctly for every proposal.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        device: torch.device,
        submodule: torch.nn.Module,
        steering_vectors: list[torch.Tensor],
        positions: list[list[int]],
        steering_coefficient: float = 1.0,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.block_size = get_text_config(self.model).max_position_embeddings
        self.submodule = submodule
        self._handle: Optional[torch.utils.hooks.RemovableHook] = None

        # Store hook parameters for re-registration
        self._steering_vectors = steering_vectors
        self._positions = positions
        self._steering_coefficient = steering_coefficient
        self._dtype = dtype

    def _get_generation_stop_ids(self) -> int | list[int] | None:
        eos_token_id = getattr(self.model.generation_config, "eos_token_id", None)
        if eos_token_id is not None:
            return eos_token_id
        return self.tokenizer.eos_token_id

    def _get_pad_token_id(self) -> int | None:
        pad_token_id = getattr(self.model.generation_config, "pad_token_id", None)
        if pad_token_id is not None:
            return pad_token_id
        return self.tokenizer.pad_token_id

    def _trim_generated_ids(self, gen_ids: list[int]) -> list[int]:
        eos_token_id = self._get_generation_stop_ids()
        if eos_token_id is None:
            return gen_ids
        stop_ids = eos_token_id if isinstance(eos_token_id, list) else [eos_token_id]
        for idx, token_id in enumerate(gen_ids):
            if token_id in stop_ids:
                return gen_ids[:idx]
        return gen_ids

    def attach_hook(self):
        """Register the steering hook persistently on the submodule."""
        if self._handle is not None:
            self._handle.remove()

        hook_fn = get_hf_activation_steering_hook(
            vectors=self._steering_vectors,
            positions=self._positions,
            steering_coefficient=self._steering_coefficient,
            device=self.device,
            dtype=self._dtype,
        )
        self._handle = self.submodule.register_forward_hook(hook_fn)

    def detach_hook(self):
        """Remove the steering hook."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def set_steering_coefficient(self, c: float) -> None:
        """Update steering coefficient and re-register hook if currently attached."""
        self._steering_coefficient = c
        if self._handle is not None:
            self.detach_hook()
            self.attach_hook()

    def __enter__(self):
        self.attach_hook()
        return self

    def __exit__(self, *args):
        self.detach_hook()

    @torch.no_grad()
    def next_token(self, prefix: list[int]) -> torch.Tensor:
        """Get log-probabilities for the next token given a prefix."""
        torch_prefix = torch.tensor([prefix], dtype=torch.long, device=self.device)
        if torch_prefix.size(1) > self.block_size:
            torch_prefix = torch_prefix[:, -self.block_size :]
        output = self.model(torch_prefix)
        logits = output.logits[0, -1, :]
        return F.log_softmax(logits, dim=-1)

    @torch.no_grad()
    def generate_batch_texts(
        self,
        context: list[int],
        temperature: float,
        max_new_tokens: int,
        num_samples: int,
        do_sample: bool = True,
    ) -> list[str]:
        """Generate ``num_samples`` completions in a single batched forward.

        Returns the decoded text of each completion (with special tokens
        stripped). The steering hook is rebuilt for batch size ``num_samples``
        for the duration of the call and restored afterward.
        """
        if num_samples <= 0:
            return []

        # Rebuild the hook to match the expanded batch.
        had_hook = self._handle is not None
        if had_hook:
            self._handle.remove()
            self._handle = None

        batched_hook = get_hf_activation_steering_hook(
            vectors=list(self._steering_vectors) * num_samples,
            positions=list(self._positions) * num_samples,
            steering_coefficient=self._steering_coefficient,
            device=self.device,
            dtype=self._dtype,
        )
        batched_handle = self.submodule.register_forward_hook(batched_hook)

        try:
            input_ids = torch.tensor(
                [context], dtype=torch.long, device=self.device
            ).repeat(num_samples, 1)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "eos_token_id": self._get_generation_stop_ids(),
                "pad_token_id": self._get_pad_token_id(),
                "return_dict_in_generate": True,
            }
            if do_sample:
                gen_kwargs["temperature"] = temperature

            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )
        finally:
            batched_handle.remove()
            if had_hook:
                self.attach_hook()

        c = len(context)
        texts: list[str] = []
        for b in range(num_samples):
            gen_ids = output.sequences[b][c:].tolist()
            gen_ids = self._trim_generated_ids(gen_ids)
            texts.append(self.tokenizer.decode(gen_ids, skip_special_tokens=True))
        return texts

    @torch.no_grad()
    def generate_with_logprobs(
        self,
        context: list[int],
        temperature: float = 1.0,
        max_new_tokens: int = 20,
        do_sample: bool = True,
    ) -> tuple[list[int], list[float], list[float]]:
        """Generate tokens and return (token_ids, log_probs_scaled, log_probs_unscaled).

        This mirrors naive_temp() from power_samp_utils but works with
        the persistent steering hook.
        """
        input_ids = torch.tensor([context], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "eos_token_id": self._get_generation_stop_ids(),
            "pad_token_id": self._get_pad_token_id(),
            "return_dict_in_generate": True,
            "output_scores": True,
            "output_logits": True,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature

        output = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )

        c = len(context)
        tokens = output.sequences[0][c:]
        num_generated = len(tokens)

        if num_generated == 0:
            return context, [], []

        unscaled_logits = torch.stack(output.logits, dim=0)  # (T, 1, V)
        scaled_logits = torch.stack(output.scores, dim=0)  # (T, 1, V)

        idx = tokens.view(num_generated, 1, 1)

        # Unscaled log-probs (from the model's actual distribution)
        log_probs_unscaled = (
            torch.gather(F.log_softmax(unscaled_logits, dim=-1), -1, idx)
            .view(-1)
            .tolist()
        )

        # Scaled log-probs (after temperature, used as proposal distribution)
        log_probs_scaled = (
            torch.gather(F.log_softmax(scaled_logits, dim=-1), -1, idx)
            .view(-1)
            .tolist()
        )

        full_seq = output.sequences[0].tolist()
        return full_seq, log_probs_scaled, log_probs_unscaled

    def greedy_generate(
        self,
        context: list[int],
        max_new_tokens: int = 20,
    ) -> tuple[list[int], list[float]]:
        """Greedy generation returning token_ids and per-token log-probs."""
        full_seq, log_probs, _, _ = self.greedy_generate_with_token_stats(
            context=context,
            max_new_tokens=max_new_tokens,
        )
        return full_seq, log_probs

    @torch.no_grad()
    def greedy_generate_with_token_stats(
        self,
        context: list[int],
        max_new_tokens: int = 20,
    ) -> tuple[list[int], list[float], list[float], list[float]]:
        """Greedy generation with generated-token log-probs and entropy stats.

        Returns:
            ``(full_seq, token_log_probs, token_entropies, token_max_probs)``.
            Entropies and max probabilities are computed from the unscaled
            next-token distribution at each generated step.
        """
        input_ids = torch.tensor([context], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        output = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=self._get_generation_stop_ids(),
            pad_token_id=self._get_pad_token_id(),
            return_dict_in_generate=True,
            output_scores=True,
            output_logits=True,
        )

        c = len(context)
        tokens = output.sequences[0][c:]
        num_generated = len(tokens)

        if num_generated == 0:
            return context, [], [], []

        unscaled_logits = torch.stack(output.logits, dim=0)
        idx = tokens.view(num_generated, 1, 1)
        token_log_distributions = F.log_softmax(unscaled_logits.float(), dim=-1)
        token_distributions = torch.exp(token_log_distributions)

        log_probs = (
            torch.gather(token_log_distributions, -1, idx)
            .view(-1)
            .tolist()
        )
        entropies = (
            (-(token_distributions * token_log_distributions).sum(dim=-1))
            .view(-1)
            .tolist()
        )
        max_probs = token_distributions.max(dim=-1).values.view(-1).tolist()

        full_seq = output.sequences[0].tolist()
        return full_seq, log_probs, entropies, max_probs


def naive_temp_steered(
    sampler: SteeredAutoregressiveSampler,
    context: list[int],
    temp: float,
    seq_len: int,
) -> tuple[list[int], list[float], list[float]]:
    """Low-temperature sampling proposal for MCMC, using steered generation.

    Drop-in replacement for naive_temp() from power_samp_utils.py,
    adapted to work with SteeredAutoregressiveSampler.
    """
    c = len(context)
    max_new = seq_len - c
    if max_new <= 0:
        return context, [], []

    full_seq, log_probs_norm, log_probs_unnorm = sampler.generate_with_logprobs(
        context=context,
        temperature=temp,
        max_new_tokens=max_new,
        do_sample=True,
    )

    # Scale unscaled log-probs by 1/temp to get target distribution log-probs
    log_probs_unnorm_scaled = [lp / temp for lp in log_probs_unnorm]

    return full_seq, log_probs_norm, log_probs_unnorm_scaled


def mcmc_power_samp_steered(
    sampler: SteeredAutoregressiveSampler,
    context: list[int],
    temp: float,
    mcmc_steps: int,
    max_new_tokens: int,
    block_num: int = 4,
) -> tuple[list[int], list[float], list[float], float]:
    """MCMC power sampling on an activation-steered oracle model.

    Adapted from mcmc_power_samp() in power_samp_utils.py to use
    SteeredAutoregressiveSampler instead of raw AutoregressiveSampler.

    Returns: (generated_tokens, log_probs_norm, log_probs_unnorm, acceptance_ratio)
    """
    c = len(context)
    gen = context.copy() if context is not None else []
    log_probs_norm: list[float] = []
    log_probs_unnorm: list[float] = []

    assert max_new_tokens % block_num == 0, (
        f"max_new_tokens ({max_new_tokens}) must be divisible by block_num ({block_num})"
    )
    jump_size = max_new_tokens // block_num
    attempts = 0
    acceptances = 0

    for _ in range(block_num):
        gen, lp_norm, lp_unnorm = naive_temp_steered(
            sampler, gen, temp=temp, seq_len=jump_size + len(gen)
        )
        log_probs_norm.extend(lp_norm)
        log_probs_unnorm.extend(lp_unnorm)

        for _ in range(mcmc_steps):
            attempts += 1
            t = len(gen)
            if t <= c:
                break
            idx = random.randint(c, t - 1)

            prop, log_prob_prop, target_log_prob_prop = naive_temp_steered(
                sampler, gen[:idx], temp=temp, seq_len=t
            )
            s = len(prop)
            assert len(log_prob_prop) == s - idx
            assert len(target_log_prob_prop) == s - idx

            log_prob_cur = log_probs_norm[idx - c : s - c]
            target_log_prob_cur = log_probs_unnorm[idx - c : s - c]

            # Metropolis-Hastings acceptance ratio
            log_r = (
                sum(target_log_prob_prop)
                + sum(log_prob_cur)
                - sum(target_log_prob_cur)
                - sum(log_prob_prop)
            )

            if np.random.rand() < np.exp(
                min(log_r, 0)
            ):  # clamp for numerical stability
                acceptances += 1
                gen = prop.copy()
                log_probs_norm[idx - c :] = log_prob_prop.copy()
                log_probs_unnorm[idx - c :] = target_log_prob_prop.copy()

        if sampler.tokenizer.eos_token_id in gen[c:]:
            eos_idx = gen.index(sampler.tokenizer.eos_token_id, c)
            gen = gen[: eos_idx + 1]
            log_probs_norm = log_probs_norm[: eos_idx + 1 - c]
            log_probs_unnorm = log_probs_unnorm[: eos_idx + 1 - c]
            acceptance_ratio = acceptances / max(attempts, 1)
            return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio

    acceptance_ratio = acceptances / max(attempts, 1)
    return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio


def max_swap_steered(
    sampler: SteeredAutoregressiveSampler,
    context: list[int],
    temp: float,
    mcmc_steps: int,
    max_new_tokens: int,
    block_num: int = 4,
) -> tuple[list[int], list[float], list[float], float]:
    """Greedy (alpha=inf) power sampling variant on steered oracle.

    Like mcmc_power_samp_steered but always accepts improvements (log_r > 0).
    """
    c = len(context)
    gen = context.copy() if context is not None else []
    log_probs_norm: list[float] = []
    log_probs_unnorm: list[float] = []

    assert max_new_tokens % block_num == 0
    jump_size = max_new_tokens // block_num
    attempts = 0
    acceptances = 0

    for _ in range(block_num):
        gen, lp_norm, lp_unnorm = naive_temp_steered(
            sampler, gen, temp=temp, seq_len=jump_size + len(gen)
        )
        log_probs_norm.extend(lp_norm)
        log_probs_unnorm.extend(lp_unnorm)

        for _ in range(mcmc_steps):
            attempts += 1
            t = len(gen)
            if t <= c:
                break
            idx = random.randint(c, t - 1)

            prop, log_prob_prop, target_log_prob_prop = naive_temp_steered(
                sampler, gen[:idx], temp=temp, seq_len=t
            )
            s = len(prop)
            assert len(log_prob_prop) == s - idx
            assert len(target_log_prob_prop) == s - idx

            target_log_prob_cur = log_probs_unnorm[idx - c : s - c]
            log_r = sum(target_log_prob_prop) - sum(target_log_prob_cur)

            if log_r > 0:
                acceptances += 1
                gen = prop.copy()
                log_probs_norm[idx - c :] = log_prob_prop.copy()
                log_probs_unnorm[idx - c :] = target_log_prob_prop.copy()

        if sampler.tokenizer.eos_token_id in gen[c:]:
            eos_idx = gen.index(sampler.tokenizer.eos_token_id, c)
            gen = gen[: eos_idx + 1]
            log_probs_norm = log_probs_norm[: eos_idx + 1 - c]
            log_probs_unnorm = log_probs_unnorm[: eos_idx + 1 - c]
            acceptance_ratio = acceptances / max(attempts, 1)
            return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio

    acceptance_ratio = acceptances / max(attempts, 1)
    return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio
