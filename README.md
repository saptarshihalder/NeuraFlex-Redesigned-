# NeuraFlex-Ouroboros

**Version 1.1.0**: a 52B-parameter looped Mixture-of-Experts language model architecture, plus a CPU-native variant that trains on a swarm of free machines. Built from scratch by Saptarshi Halder.

NeuraFlex started as a ~100K-parameter character-level transformer (preserved in [`legacy/v0.0.1`](legacy/v0.0.1)). v1 is a full redesign around one idea: **depth should be a runtime knob, not a fixed cost.**

- **Looped core.** 12 unique layers. The 8-layer core repeats r times, giving 36 layers of compute at r=4 and 68 at r=8 with no new weights.
- **Depth-Routed Experts.** Each core layer has 256 fine-grained experts plus a shared expert. The router sees the loop index, so each pass can call different experts, which turns a fixed parameter budget into a combinatorial set of virtual layers.
- **Loop-LoRA.** Rank-64 adapters for each iteration let every pass specialise while sharing over 99% of its weights.
- **Hybrid attention.** Gated DeltaNet (O(1) state) handles 3/4 of the core and Multi-head Latent Attention handles the rest, so looping doesn't blow up the KV cache.
- **Adaptive thinking.** Tokens stop looping once their state converges. Skipped iterations get a cheap attention-only cache fill, so the cache stays consistent.
- **Train small, grow big.** `expert_mitosis` turns a ~7B seed into the 52B model. The `split` mode preserves the function exactly.
- **Modern training stack.** Muon + AdamW, WSD schedule, log-normal Poisson loop sampling with truncated BPTT, multi-token prediction, optional distillation from an open-weight teacher, and FSDP2.

The full design, the cost plan and the ablation plan are in **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

### No GPU? Ouroboros Pocket + Hive

Free CPUs have the opposite strengths of a GPU: few FLOPs, but plenty of RAM and cheap random access. So the CPU variant inverts the design. Capacity lives in huge sparse lookup tables (262k product-key micro-experts that see the loop index, plus hashed n-gram memory), and only ~15M dense parameters do arithmetic per token. Training runs on any number of unreliable free machines through **row-sparse asynchronous DiLoCo**, with a Hugging Face repo or a shared folder as the parameter server.

On a 2-core CPU, the 577M-parameter `pocket` model trains at ~270 tok/s in ~5 GB RAM, about 9x faster than a same-size dense model, which wouldn't even fit in memory. Details, measurements, budgets and setup are in **[docs/CPU_HIVE.md](docs/CPU_HIVE.md)**.

| Preset | Total | Active / token (r=4) | Use |
|---|---|---|---|
| `debug` / `debug-pkm` | ~0.5M | — | CPU tests |
| `pocket` | 577M (537M sparse) | ~15M per pass | free-CPU training, Hive swarm |
| `3b` | 3.5B | 2.3B | ablations / proof of concept |
| `50b` | 52.5B | 12.7B | target model |

## Quick start

```bash
pip install -r requirements.txt
python -m pytest tests            # 11 tests: cache consistency, exact mitosis, truncated BPTT, swarm merge, ...
python scripts/param_count.py 50b

# CPU smoke test (byte-level)
python train.py --preset debug --data legacy/v0.0.1/sample_input.txt --steps 300 --seq-len 128
python generate.py --ckpt checkpoints/step_300.pt --prompt "Q: Who created you?" --loops 6
```

### Free-CPU swarm (Hive)

```bash
python -m neuraflex.hive.coordinator init --transport hub:YOU/neuraflex-hive --preset pocket
python -m neuraflex.hive.coordinator run  --transport hub:YOU/neuraflex-hive        # always-on box
python -m neuraflex.hive.worker --transport hub:YOU/neuraflex-hive --worker-id kaggle-1 \
  --tokenizer HuggingFaceTB/SmolLM2-135M --data HuggingFaceFW/fineweb-edu --data-config sample-10BT
python scripts/cpu_bench.py --preset pocket --compare-dense                          # measure your box
```

### Real training (GPU)

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
  config.py    presets (debug / pocket / 3b / 50b)
  layers.py    RMSNorm, RoPE, SwiGLU, MLA, Gated DeltaNet
  moe.py       Depth-Routed Experts
  sparse.py    product-key micro-experts, hashed n-gram memory (CPU design point)
  data.py      sharded token streams
  hive/        swarm training: transport, row-sparse DiLoCo protocol, worker, coordinator
  model.py     Ouroboros model, cache, adaptive exit, MTP, generate
  muon.py      Muon + AdamW, row-wise sparse Adam for huge tables
  upcycle.py   expert mitosis / deepen
train.py       FSDP2 training, distillation, WSD schedule
generate.py    sampling with adaptive loops
scripts/       param_count.py, cpu_bench.py
tests/         unit tests
legacy/v0.0.1  original NeuraFlex
```

## Status

This is a tested reference implementation that has not yet been trained at scale, so no benchmark results are claimed. A 256-expert run still needs expert-parallel and grouped-GEMM kernels (torchtitan or Megatron). See §5 of the architecture doc.

## License

MIT
