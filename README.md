# NeuraFlex-Ouroboros

**Version 1.0.0**: a 52B-parameter looped Mixture-of-Experts language model architecture, built from scratch by Saptarshi Halder.

NeuraFlex started as a ~100K-parameter character-level transformer (preserved in [`legacy/v0.0.1`](legacy/v0.0.1)). v1 is a full redesign around one idea: **depth should be a runtime knob, not a fixed cost.**

- **Looped core.** 12 unique layers. The 8-layer core repeats r times, giving 36 layers of compute at r=4 and 68 at r=8 with no new weights.
- **Depth-Routed Experts.** Each core layer has 256 fine-grained experts plus a shared expert. The router sees the loop index, so each pass can call different experts, which turns a fixed parameter budget into a combinatorial set of virtual layers.
- **Loop-LoRA.** Rank-64 adapters for each iteration let every pass specialise while sharing over 99% of its weights.
- **Hybrid attention.** Gated DeltaNet (O(1) state) handles 3/4 of the core and Multi-head Latent Attention handles the rest, so looping doesn't blow up the KV cache.
- **Adaptive thinking.** Tokens stop looping once their state converges. Skipped iterations get a cheap attention-only cache fill, so the cache stays consistent.
- **Train small, grow big.** `expert_mitosis` turns a ~7B seed into the 52B model. The `split` mode preserves the function exactly.
- **Modern training stack.** Muon + AdamW, WSD schedule, log-normal Poisson loop sampling with truncated BPTT, multi-token prediction, optional distillation from an open-weight teacher, and FSDP2.

The full design, the cost plan and the ablation plan are in **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

| Preset | Total | Active / token (r=4) | Use |
|---|---|---|---|
| `debug` | ~0.5M | — | CPU tests |
| `3b` | 3.5B | 2.3B | ablations / proof of concept |
| `50b` | 52.5B | 12.7B | target model |

## Quick start

```bash
pip install -r requirements.txt
python -m pytest tests            # 8 tests: cache consistency, exact mitosis, truncated BPTT, ...
python scripts/param_count.py 50b

# CPU smoke test (byte-level)
python train.py --preset debug --data legacy/v0.0.1/sample_input.txt --steps 300 --seq-len 128
python generate.py --ckpt checkpoints/step_300.pt --prompt "Q: Who created you?" --loops 6
```

### Real training

```bash
torchrun --nnodes $N --nproc-per-node 8 train.py --preset 3b \
  --tokenizer Qwen/Qwen3-8B --data HuggingFaceFW/fineweb-edu --data-config sample-10BT \
  --seq-len 4096 --batch-size 2 --grad-accum 8 --lr 2e-3 --steps 100000 \
  --teacher Qwen/Qwen3-8B --kd-weight 0.5
```

### Grow a seed into the 52B model

```python
from neuraflex.upcycle import expert_mitosis
big = expert_mitosis(seed_model, factor=8, mode="replicate")   # 32 -> 256 experts
```

## Layout

```
neuraflex/
  config.py    presets (debug / 3b / 50b)
  layers.py    RMSNorm, RoPE, SwiGLU, MLA, Gated DeltaNet
  moe.py       Depth-Routed Experts
  model.py     Ouroboros model, cache, adaptive exit, MTP, generate
  muon.py      Muon + AdamW optimizer
  upcycle.py   expert mitosis / deepen
train.py       FSDP2 training, distillation, WSD schedule
generate.py    sampling with adaptive loops
tests/         unit tests
legacy/v0.0.1  original NeuraFlex
```

## Status

This is a tested reference implementation that has not yet been trained at scale, so no benchmark results are claimed. A 256-expert run still needs expert-parallel and grouped-GEMM kernels (torchtitan or Megatron). See §5 of the architecture doc.

## License

MIT
