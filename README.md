# Probabilistic Activation Oracles

## Background: related work (draft, only one paper so far)

Baker et al. 2025 ("Are Language Models Aware of the Road Not Taken?", arXiv:2511.04527) show that LLM hidden activations encode information about counterfactual outcomes, and that the effectiveness of residual-stream steering correlates with token-level uncertainty ($R \approx 0.57$ to $0.64$). Their key finding: steering sharply fails once a model "commits" to an answer, while uncertain models remain steerable. This paper validates that activations carry outcome-distribution information beyond surface tokens (directly supporting Methods 1 and 6 here), and motivates using activation-perturbation response as an uncertainty signal.

We include six UQ methods (1 log-prob, 2 temperature bootstrap, 3 direct elicitation, 4 single MCMC, 5 MCMC agreement, 6 steering sensitivity). Methods 2, 4, and 5 are run across a small grid of temperatures / powers so we can characterize how calibration depends on the proposal sharpness.

## Background: what is an activation oracle?

An activation oracle is a second LLM that has been trained to read the internal activations of a first LLM and answer questions about them in plain language.

Here is a concrete example. Suppose you have an LLM that has been secretly instructed "your secret word is moon." When you chat with this model, it won't reveal the secret word directly. But internally, the knowledge of "moon" is encoded in its activations. An activation oracle can take those internal numbers as input and tell you: "the secret word is moon."

How does this work mechanically? The oracle is a normal language model with a small modification: at one of its internal layers, we inject the activations we collected from the target model. This injection is called "steering." We literally overwrite some of the oracle's internal numbers with numbers from the target model. The oracle has been trained (via fine-tuning) to interpret these injected numbers and generate a useful answer.

Technically, the oracle model M_oracle receives a prompt containing special placeholder tokens. At a designated injection layer (typically layer 1), a forward hook intercepts the residual stream activations. For each placeholder token position p, the hook replaces the residual with: $h'(p) = normalize(v_p) * ||h(p)|| * c$, where $v_p$ is the activation vector collected from the target model at some layer (e.g., 50% depth), $h(p)$ is the oracle's original residual at that position, and $c$ is a steering coefficient (default 1.0). This preserves the activation norm while substituting the directional information from the target model. The oracle is then trained with LoRA adapters on a mix of tasks (LatentQA, classification, context prediction) using standard cross-entropy loss so that it learns to decode these injected activations into natural language answers.

Pre-trained oracles already exist for several popular models (Qwen, Gemma, Llama).

## The problem: oracles give answers, but no confidence

When the oracle says "the secret word is moon," it might be very sure, or it might be guessing. Right now there is no way to tell. The oracle just outputs text. It does not say "I'm 90% sure" or "this could also be sun."

We want to attach a reliable confidence number to every oracle prediction. If the oracle says "moon" with 90% confidence, it should be right about 90% of the time. This property is called "calibration."

## Background: what is power sampling?

When an LLM generates text, it picks each word by sampling from a probability distribution (a list of all possible next words with a probability for each). Normally it just picks the most likely word, or adds a bit of randomness ("temperature") to explore alternatives.

Power sampling is a smarter way to generate text. Instead of sampling from the model's raw distribution p, it samples from p raised to a power ($p^{\alpha}$, where $\alpha > 1$). This makes likely words even more likely and unlikely words even less likely. It "sharpens" the distribution. The trick is that you can't compute $p^{\alpha}$ directly (it's mathematically intractable), so the method uses a technique from statistics called MCMC (Markov Chain Monte Carlo): generate a draft answer, then repeatedly propose small edits and keep them only if they improve the overall probability. After enough rounds of propose-and-accept/reject, the resulting text is effectively drawn from the sharper distribution.

The key property of power sampling: it concentrates on high-quality outputs while still preserving meaningful variety. If the model is genuinely uncertain between two answers, both will show up. If one answer is clearly better, power sampling will converge on it. This makes disagreement between power samples informative, unlike low-temperature sampling (which collapses to a single answer regardless of uncertainty) or high-temperature sampling (which adds random noise that obscures real uncertainty).

Technically, given a base autoregressive model with token-level distribution $p(x_t | x_{<t})$, the target distribution is the power distribution $p(x)^{\alpha}$ (unnormalized). Since the normalization constant is intractable, the method uses Metropolis-Hastings MCMC. The generation is divided into B blocks. For each block: (1) generate a candidate continuation using a low-temperature proposal distribution $q$ (temperature = $1/\alpha$), (2) for each of S MCMC steps, pick a random position idx in the current block, resample from idx to end using $q$, and accept with probability $\min(1, \exp(\log_r))$ where $\log_r = \sum(\log p^{\alpha}(proposed)) + \sum(\log q(current)) - \sum(\log p^{\alpha}(current)) - \sum(\log q(proposed))$. This Metropolis-Hastings correction ensures the chain's stationary distribution is $p^{\alpha}$. The acceptance ratio (acceptances/attempts) is a diagnostic of the chain.

## Our idea: six ways to measure oracle confidence

We propose six methods to estimate how confident the oracle is, then compare them to see which gives the most reliable confidence scores.

### Method 1: Look at the oracle's own probabilities

When the oracle generates the word "moon," it internally assigns probabilities to every possible word. The probability it assigned to "moon" is itself a confidence measure. If the oracle gave "moon" a 95% probability, it was very sure. If it gave "moon" 30% and "sun" 25% and "star" 20%, it was uncertain.

This is the simplest method. We just read out numbers the model already computes.

Technically, we run greedy decoding (temperature=0) with output_logits=True. For each generated token $x_t$, we record the log-probability $\log p(x_t | x_{<t})$ from the unscaled logits. We log several diagnostics: (a) mean sequence log-probability $(1/T) * \sum(\log p(x_t))$, (b) minimum token log-probability $\min_t(\log p(x_t))$, (c) the entropy of the first token distribution $H = -\sum(p * \log p)$ which captures how spread out the oracle's initial "guess" is, (d) mean generated-token entropy, matching the entropy-style confidence diagnostic used in sampling work, and (e) the geometric mean token probability $\exp(\text{mean log-prob})$. The scalar confidence used in the experiment table is the joint probability of the extracted answer word under the generated tokenization. We report two implementations of that scalar in code: an offset-based alignment from the extracted word back to generated tokens, and an offset-free prefix approximation. The offset-free variant uses the extracted predicted word from the candidate vocabulary, not the ground-truth target word; it is a prefix approximation and is most meaningful when the oracle starts its response with the answer. If either variant falls back to first-token max probability, the run records explicit fallback flags and aggregate fallback rates.

### Method 2: Ask the same question many times

Ask the oracle "what is the secret word?" 20 times, each time with a small amount of randomness in the generation. If 18 out of 20 answers say "moon", the confidence is 18/20 = 90%. If the answers are scattered across many different words, confidence is low.

This is like asking 20 people the same question and seeing how much they agree.

Technically, we run $k=20$ independent generations with `do_sample=True` at temperature $T$ (default sweep: {0.3, 0.5, 0.7, 1.0, 1.3, 1.5}). Each generation uses the same steering hook (same activation injection). We normalize the decoded answers (lowercase, strip punctuation) and compute the empirical distribution. The confidence score is the mode frequency: $\frac{\text{count(most common answer)}}{k}$. We also report the Shannon entropy of the empirical distribution $H = -\sum(\frac{n_i}{k} * \log(\frac{n_i}{k}))$ and the number of unique answers.

### Method 3: Just ask the oracle how confident it is

After the oracle answers "moon," we follow up with: "on a scale of 0 to 100, how confident are you?" This is the most naive approach. LLMs tend to be overconfident (they say 95% when they should say 60%), so we expect this to be badly calibrated, but it is worth measuring as a baseline.

Technically, this is a two-turn generation with persistent steering. Turn 1: generate the answer with the steering hook active. Turn 2: append the answer and the confidence prompt to the context, generate again with the same hook still active (the hook only fires during prefill because of the L<=1 guard, so both turns see the steered activations). Direct elicitation has separate answer and confidence temperatures, both defaulting to 0 (greedy). Parse the numeric response with regex and clamp to [0, 1]. If parsing fails, the implementation first asks a stricter numeric retry prompt, then falls back to structured scoring of integer candidates 0..100 and uses the expected value. The old 0.5 maximum-uncertainty fallback remains only as a last-resort sentinel and is logged explicitly if it ever happens.

### Method 4: Power sampling on the oracle (the main new idea)

Instead of generating the oracle's answer with normal sampling, we use the power sampling technique described above. This generates answers from a sharper version of the oracle's distribution.

This gives us two confidence signals:

- **The distribution of answers**: just like Method 2, we can run power sampling multiple times and count agreement. But because power sampling preserves meaningful diversity (unlike temperature tricks that either collapse everything to one answer or add random noise), the agreement rate is a more reliable confidence score.
- **The acceptance rate of the MCMC process**: during power sampling, the algorithm proposes edits to the text and decides whether to accept or reject them. In the activation-oracle setting, the raw acceptance ratio is useful as an empirical confidence score: it measures how mobile the chain remains under answer-preserving low-temperature proposals. Our aggregate results favor using the acceptance ratio directly rather than inverting it. The original intuition that low acceptance always means high confidence is too brittle here, because a near-locked chain can also be locked into a wrong or poorly aligned continuation.

Technically, we wrap the activation-steered oracle in the MCMC power sampler. The steering hook is registered as a persistent forward hook on the injection layer's submodule (not via context manager) for the duration of the MCMC chain. Each call to naive_temp() inside the MCMC loop triggers model.generate(), which runs a fresh prefill pass where the hook fires (sequence length > 1). During autoregressive token-by-token decoding within each generate call, the hook's L<=1 guard causes it to pass through without modification, which is correct since steering should only happen at the prompt positions during prefill. Each MCMC proposal calls naive_temp(sampler, gen[:idx], temp, seq_len), resampling from a random position idx with the full oracle prompt (including steering positions) always preserved as the prefix. We use max_new_tokens=20 (oracle answers are short), block_num=4 (jump_size=5), and sweep temperature in $\{0.5, 0.25, 0.125\}$ corresponding to $\alpha$ in $\{2, 4, 8\}$. The confidence score used in the experiment table is the raw acceptance ratio. We also record the full log-probs (both scaled and unscaled) returned by naive_temp for analysis.

### Method 5: Power sampling agreement

Run 10 completely independent power sampling chains on the same input and count how many agree. This is the power-sampling version of Method 2. The key advantage: with Method 2, if you set the randomness too low, all 20 answers are identical (even when the oracle is unsure), and if you set it too high, answers are randomly scattered (even when the oracle is sure). Power sampling avoids both failure modes. Disagreement between power samples means the oracle is genuinely torn.

Technically, we run k=10 independent calls to mcmc_power_samp_steered() with the same steering vectors, positions, and oracle prompt. Each chain is initialized independently (different random seed for MCMC proposals). We normalize answers and compute mode_frequency = count(mode) / k as the confidence score. We also report mean_acceptance_ratio across the k chains as a secondary uncertainty signal, and the Shannon entropy of the answer distribution. The hypothesis is that power sampling agreement is better calibrated than temperature agreement (Method 2) because the power distribution $p^\alpha$ preserves diversity where the model is genuinely uncertain (multiple high-probability modes) while suppressing noise in low-probability regions, whereas temperature sampling cannot distinguish between these cases.

### Method 6: Turn the injection dial up and down

The oracle reads secret knowledge from the target model by having that knowledge injected into its internal computations. Think of the injection as a dial: at the normal setting, the oracle hears the secret at full volume. We can turn the dial slightly up or slightly down (a little louder, a little quieter).

If the oracle is genuinely confident about the secret word, turning the dial should not matter much. Whether we inject the information a little louder or a little quieter, the underlying signal points clearly in one direction, and the oracle lands on the same answer every time.

If the oracle is uncertain (if the target model's internal representation of the secret is weak or muddled), then small changes to the dial tip the oracle toward different words. The output becomes unstable across the sweep.

So we run five versions with the dial at different settings (quieter-than-normal, slightly quiet, normal, slightly loud, louder-than-normal), recording one answer per setting. The confidence score is the fraction of these five answers that agree. All five say "moon"? Confidence = 100%. Three say "moon" and two say "cloud"? Confidence = 60%.

This is the cheapest method: five runs instead of 20 (Method 2) or dozens of rounds (Methods 4 and 5). It is inspired by research showing that how easily a model can be steered correlates with how uncertain it is (a model already committed to an answer resists being pushed, while an uncertain one is easily swayed). Here we flip that insight: if the oracle is sure about what the activation is saying, turning the volume up or down changes nothing; if it is guessing, even a small nudge flips the answer.

Technically, the steering hook replaces the oracle's residual activation at each injection position with $\text{normalize}(v) \cdot \|\text{residual}\| \cdot c$, where $c$ is the dial value. We sweep $c$ over $\{0.5, 0.75, 1.0, 1.25, 1.5\}$ (excluding $0$, which disables injection entirely, and negative values, which flip the direction). Each setting uses greedy (deterministic) decoding so that the only source of variation is the dial itself. After the sweep, $c$ is reset to $1.0$ inside a finally block so that downstream methods in the same run are unaffected. The confidence score is $\text{mode frequency} = \text{count(most common answer)} / 5$. Secondary metadata: per-dial answers, per-dial mean log-probability, Shannon entropy of the answer distribution across the sweep, and number of unique answers.

## How we test which method is best: the secret word protocol

We use the "taboo" task from the activation oracle paper. In this task, 20 different versions of a model each have a different secret word (ship, wave, song, snow, rock, moon, jump, green, flame, flag, dance, cloud, clock, chair, salt, book, blue, gold, leaf, smile). Each version is created by fine-tuning the base model with a small adapter (called a LoRA) specific to that word.

This gives us a clean experiment:

1. Pick a context prompt (e.g., "give me a hint about your secret").
2. For each of the 20 model versions, collect the internal activations when it processes that prompt.
3. Feed each set of activations to the oracle and ask "what is the secret word?" using each of our six methods.
4. Each method produces both an answer and a confidence score. We know the correct answer for each model version.

A perfectly calibrated method would assign high confidence when it gets the right word and low confidence when it gets the wrong word. We measure calibration using standard statistical tools:

- **Expected Calibration Error (ECE)**: group predictions by their confidence level (e.g., all predictions where confidence was 80-90%), and check whether the accuracy in each group matches the confidence. If the 80-90% group is actually right 85% of the time, the method is well calibrated.
- **Brier score**: the average squared difference between the confidence and the actual outcome (1 if correct, 0 if wrong). Lower is better.
- **Reliability diagrams**: a plot of confidence vs. actual accuracy. A perfectly calibrated method follows the diagonal. The code now saves one diagram per method (and per controlled-N subset) alongside the JSON summaries.

We also run a **controlled-N variant**: instead of all 20 words, use subsets of 2, 5, 10, or 20 words. With only 2 possible words, the task is easier; with 20, it is harder. We check whether each method's calibration degrades gracefully as the task gets more difficult.

Technically, we use 20 target LoRAs (adamkarvonen/Qwen3-8B-taboo-{word}_50_mix), 3 verbalizer prompts, and context prompts from datasets/taboo/taboo_direct_test.txt. Each target LoRA is loaded, activations are collected at 50% layer depth, and the oracle (with its own LoRA, e.g., adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B) generates steered responses. For the controlled-N test, we sample N words from the 20 with a fixed random seed and run the full protocol on just those N. ECE is computed with 10 equal-width bins on [0, 1], Brier score is $\text{mean}((\text{confidence} - \text{correct})^2)$, and NLL is $\text{mean}(-\text{correct} \cdot \log(\text{conf}) - (1 - \text{correct}) \cdot \log(1 - \text{conf}))$.

## What we expect to find

1. Power sampling methods (4 and 5) give better-calibrated confidence than temperature bootstrap (Method 2) for the same computational cost.
2. The MCMC acceptance rate (from Method 4) is a useful sampler-derived confidence signal on its own, and the current implementation evaluates the raw acceptance ratio directly.
3. Power sampling agreement (Method 5) is better calibrated than temperature agreement (Method 2) because power sampling preserves meaningful diversity.
4. The simple log-probability baseline (Method 1) is surprisingly competitive, because the oracle was trained to assign high probability to correct answers.
5. Some secret words are inherently harder for the oracle to detect than others. The calibration test reveals which words have more distinctive internal representations.
