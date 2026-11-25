# NeuraFlex-Ouroboros: Architecture and Plan

> A 52B looped (recurrent-depth) hybrid-attention Mixture-of-Experts language model.
> It has 12 unique layers, and at inference it can run anywhere from 12 to 130+ layers deep.

## 1. The core bet

Most of a dense transformer's depth goes to refining the same representation again and again. Ouroboros makes that loop explicit and cheap:

```
                 ┌────────────────────────── loop r times ──────────────────────────┐
tokens ─► Prelude ─► e ─►│ Adapter([norm(s); norm(e)]) ─► Core_1 ─► Core_2 ─► … ─► Core_8 │─► Coda ─► LM head
          (2 MLA,         │       ▲ echo injection             │                         │   (2 MLA,   + MTP head
           dense FFN)     │       └────────────── s ◄──────────┘                         │    dense FFN)
                          └───────────────────────────────────────────────────────────────┘
Core layer j = [GDN | GDN | GDN | MLA] attention  +  Depth-Routed Experts (256 experts, top-8, +1 shared)
               + per-loop LoRA  + sandwich norms
```

| | Value |
|---|---|
| Total parameters | **52.5B** (48.3B in routed experts) |
| Active per token, 1 loop | ~4.6B |
| Active per token, 4 loops (training mean) | ~12.7B |
| Active per token, 8 loops | ~23.4B |
| Unique layers / effective depth at 8 loops | 12 / 68 |
| Width / heads | 5120 / 40 |
| Vocabulary | 151,936 (matches the Qwen tokenizer, so Qwen models can be used as teachers) |
| KV cache | MLA latent (576 values per token per MLA layer); GDN layers keep O(1) state |

Run `python scripts/param_count.py 50b` to reproduce these numbers.

## 2. Components

### 2.1 Depth-Routed Experts (DRE)
Each core layer has a single expert pool that every loop iteration reuses. The router takes `[h ; φ(i)]`, where `φ(i)` is a sinusoidal embedding of the loop index, so the same token can visit different experts on each pass. With top-8 out of 256 experts over r passes, a fixed parameter budget yields a combinatorial number of distinct "virtual layers". Because φ is sinusoidal, the router still gets a meaningful input at loop counts it never saw during training.

Load balancing follows the DeepSeek-V3 recipe: a per-expert bias that is updated outside the gradient (by its sign) steers top-k selection, and a 1e-4 sequence-level auxiliary loss acts as a safety net.

*Closest prior work:* MoEUT (Csordás et al., 2024) and Mixture-of-Recursions (Bae et al., 2025). What DRE adds is explicit conditioning of the router on the loop index, applied to a fine-grained expert pool shared across passes.

### 2.2 Loop-LoRA ("relaxed recursion")
Each core layer carries a rank-64 adapter for each loop index (16 in total). Iteration i can therefore specialise (early passes parse, later passes refine) while sharing over 99% of its weights with every other pass. The adapters are zero-initialised, so training starts as pure weight tying. Prior work: Relaxed Recursive Transformers (Bae et al., 2024).

### 2.3 Echo injection
At the start of every pass, the state is rebuilt from `[norm(s) ; norm(e)]`, where `e` is the prelude output. The loop can never drift away from the input, and gradients reach the prelude through every back-propagated pass. Sandwich norms inside the core keep the state's norm bounded no matter how many loops run. Prior work: Huginn recurrent depth (Geiping et al., 2025).

### 2.4 Hybrid attention inside the loop
Three of every four core layers use **Gated DeltaNet**, a linear-attention layer with a delta-rule memory and O(1) state. Every fourth layer, and all prelude and coda layers, use **Multi-head Latent Attention**, which caches only a 512-dimensional latent plus a 64-dimensional RoPE key. Without this, looping multiplies the KV cache by r; with it, a 128k context at 8 loops still fits. If `flash-linear-attention` is installed, the fused kernels are used for training.

### 2.5 Adaptive depth at inference (convergence exit + cache fill)
After each pass, the model measures `‖s_i − s_{i−1}‖ / ‖s_i‖`. Once that change falls below `exit_threshold` for every token in the step, it stops looping. Easy tokens such as "the" or "of" exit after 2 passes, while hard tokens keep thinking. The key trick is that skipped iterations still need cache entries, or later tokens would find holes. So for each skipped iteration, Ouroboros runs only the attention sublayer on the final state (no MoE). Every iteration's cache stays aligned at a fraction of a full pass's cost.

### 2.6 Multi-token prediction
One MTP module (DeepSeek-V3 style) predicts token t+2. It gives a denser training signal, and the head can later serve as a speculative-decoding draft.

### 2.7 Training-time depth sampling
For each step, the loop count r is drawn from `Poisson(LogNormal(log 4 − σ²/2, 0.5)) + 1`, capped at 16. Only the last 4 iterations are back-propagated, so activation memory is fixed regardless of r. The heavy tail of the distribution teaches the core to keep improving its answer with more loops rather than overfitting to one depth. All ranks share a seeded generator, so they agree on r.

### 2.8 Muon optimizer
Every weight matrix, including each expert's slice of the 3-D expert tensors, gets orthogonalised momentum via 5 Newton–Schulz steps. The update RMS is matched to AdamW, so both optimizer groups share one learning rate. Embeddings, the LM head, norms and gates use AdamW. The schedule is WSD (warmup–stable–decay), which lets a cooldown branch off any checkpoint.

## 3. The plan to make 50B affordable

Training 52B from scratch on a large corpus costs roughly **3×10²³ FLOPs**. At about 30% MFU on H100s, that is **~250k–300k GPU-hours**. Nobody has that lying around, so the plan avoids paying for it directly:

1. **Prove it at 3B first.** The `3b` preset (3.5B total, 64 experts) runs every ablation in §4 for roughly 1–3k GPU-hours. If DRE or looping doesn't beat a matched non-looped MoE on loss per FLOP, stop. That is the result, and it is publishable either way.
2. **Grow; don't train big from zero.** Train a ~7B seed with 32 experts, then call `expert_mitosis(mode="replicate", factor=8)` to get 256 experts and the full 52B, with active parameters unchanged. `mode="split"` cuts experts finer while preserving the function *exactly* (tested). Depth needs no growth operator: raise the loop distribution with `deepen()`.
3. **Distil from an open-weight teacher.** The vocabulary matches the Qwen tokenizer, so `--teacher` gives a dense, soft target on every token. That is far more signal per token than the raw next token, which means fewer tokens are needed. Check the teacher's license before you rely on it.
4. **Get compute through grants.** Academic and startup compute programs exist (e.g. Google's TPU Research Cloud, India's subsidised GPU schemes); check current eligibility. Note that TPUs would need a `torch_xla` port.
5. **Post-train where loops pay off.** Use RL with verifiable rewards on math and code. If recurrent depth helps anywhere, it helps on problems where "think longer" means "loop more", and that thinking costs no extra output tokens.

## 4. Ablations (3B proxy, same tokens and FLOPs)

| # | Variant | Kill criterion |
|---|---|---|
| A0 | Non-looped MoE, 12 layers, same active FLOPs at r=4 | baseline |
| A1 | Looped, plain shared router | must beat A0 on val loss per FLOP |
| A2 | A1 + loop-index router (DRE) | must beat A1 |
| A3 | A2 + Loop-LoRA | must beat A2 by more than noise |
| A4 | A3, GDN vs all-MLA core | ≤0.5% loss gap for the memory savings |
| A5 | Test-time loops 4→16 on GSM8K / MATH subsets | accuracy must rise with r |
| A6 | Exit threshold sweep | ≥30% fewer loops at ≤0.5% quality loss |

## 5. Honest status

This repository contains a complete, tested **reference implementation**: model, cache, early exit, growth operators, optimizer, FSDP training loop and distillation. It has **not** been trained at scale, so no benchmark numbers exist and none are claimed. Still missing for a production run:

- **Expert parallelism and grouped-GEMM MoE kernels.** The reference MoE loops over experts; use torchtitan or Megatron for 256 experts.
- **Fused kernels for decode.** MLA weight absorption and a GDN decode kernel are still needed.
- **A data pipeline.** Dedup, quality filtering and mixture scheduling are needed beyond the streaming loader.

Whether a model like this can reach frontier quality depends mostly on data, compute and post-training, not architecture alone. What the architecture *can* plausibly offer is more reasoning per parameter and per FLOP, which is exactly what A1–A6 test.

## References
Geiping et al. 2025 (recurrent depth / Huginn) · Bae et al. 2024 (Relaxed Recursive Transformers) · Bae et al. 2025 (Mixture-of-Recursions) · Csordás et al. 2024 (MoEUT) · Dehghani et al. 2018 (Universal Transformers) · DeepSeek-AI 2024 (DeepSeek-V2/V3: MLA, aux-loss-free balancing, MTP) · Yang et al. 2024 (Gated DeltaNet) · Jordan et al. 2024 / Moonshot 2025 (Muon, Moonlight) · Komatsuzaki et al. 2022 (Sparse Upcycling)
