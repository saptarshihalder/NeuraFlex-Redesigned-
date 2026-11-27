# Training NeuraFlex on Free CPUs: Ouroboros Pocket + Hive

## 0. The physics first

| | Peak dense matmul | Cost |
|---|---|---|
| One H100 (bf16) | ~990 TFLOP/s | ~$2–3/hour |
| A free 2–4 core cloud CPU (fp32) | ~0.1–0.4 TFLOP/s | $0 |
| Same, with Intel AMX (bf16, rare on free tiers) | ~1.5 TFLOP/s (measured, 2 cores) | $0 |

Any computation a GPU does, a CPU can also do. The problem is throughput: a free CPU is roughly 2,500–10,000× slower at dense matmuls. Training the 52B GPU model (~3×10²³ FLOPs) on free CPUs would take on the order of **100,000 machine-years**, and no amount of patience fixes a gap that size.

**So we don't port the GPU recipe. We invert it.**

| Resource | GPU | Free CPU box |
|---|---|---|
| FLOPs | abundant | **scarce** |
| Memory | scarce (80 GB HBM, very expensive) | **relatively plentiful** (12–30 GB RAM, free) |
| Random access, branching, sparsity | hostile (warps, coalescing) | **cheap** |
| Interconnect | NVLink, 900 GB/s | **home-internet slow**, but time is free |
| Reliability | datacenter | **sessions die** (12 h notebook limits, lids close) |

The design rules follow directly:
1. **Capacity goes in lookup tables, not matmuls.** Each token reads a few KB out of hundreds of millions of parameters.
2. **Gradients and optimizer updates stay sparse.** Only touched rows cost anything.
3. **Sync is rare, sparse and asynchronous.** Many unreliable machines contribute without ever waiting on each other.
4. **Depth comes from loops, not new parameters.** It also gives you test-time compute.

## 1. Ouroboros Pocket (`--preset pocket`)

| | |
|---|---|
| Total parameters | **577M** |
| …in sparse tables | 537M (93%) |
| Dense parameters doing arithmetic per token per pass | **~15M** |
| Vocabulary | 49,152 (SmolLM2 tokenizer) |
| Layout | 1 prelude + 4 looped core + 1 coda layers, d=512, MLA attention |
| Training RAM | **~5 GB**, flat with batch size (micro-batch + accumulation) |

### 1.1 Depth-Routed Product-Key Micro-Experts (`neuraflex/sparse.py`)
Each core layer holds **65,536 single-neuron experts** (262k in total). A token's query is split into two halves, each half scored against 256 sub-keys, and the top-k of the k×k combinations picked. Finding the best of 65k experts therefore costs only 2×256 dot products (Lample et al. 2019; PEER, He 2024). Each token activates 32 experts per layer.

The new part, carried over from the GPU design: **the query also sees the loop index.** Each pass of the looped core therefore pulls *different* experts out of the same table. Expert rows are `nn.Embedding(sparse=True)` and `nn.EmbeddingBag`, so the gathered up-projection rows are never materialised.

### 1.2 Hashed n-gram memory
Every position also looks up hashed 2-gram and 3-gram embeddings: 2 tables × 1M rows (cf. Over-Tokenized Transformers, Huang et al. 2025). This gives phrase-level memory for zero matmul FLOPs. The decode cache carries the last two tokens, so generation matches training exactly (tested).

### 1.3 Killing the vocabulary tax
Profiling showed the 49k-word output projection was **~75% of all compute** on CPU; the trunk is tiny. Pocket uses an **adaptive softmax** (Grave et al. 2017), which spends full compute only on the ~2k most frequent tokens. The token embedding is a sparse table too. Note that the adaptive head assumes lower token ids are more frequent, which BPE merge order roughly provides.

### 1.4 Optimizer state: a recommender-systems trick
`torch.optim.SparseAdam` keeps two *dense* moment tensors, which **triples** RAM and ran out of memory on an 8 GB box. `RowWiseSparseAdam` (in `muon.py`) keeps **one second-moment scalar and one step counter per row**. This is the row-wise idea recommendation systems use for billion-row embedding tables. Optimizer state drops from 2×d floats per row to 2. Dense parameters still use Muon + AdamW.

### 1.5 Memory stays flat
- Micro-expert gathers are recomputed during backward instead of stored.
- Sparse gradient rows are deduplicated as they accumulate (via a post-accumulate hook).
- The loss is computed in checkpointed chunks, so a full tokens × vocab logit matrix never exists.

### 1.6 Measured on this box (2 vCPU Xeon @ 2.1 GHz, 8 GB RAM)

| | Throughput (train) | RAM to train |
|---|---|---|
| Pocket, fp32 (typical free CPU) | **272 tok/s** | ~5 GB |
| Pocket, bf16 (AMX CPUs only) | **290 tok/s** | ~5 GB |
| Dense model with the same 577M params, fp32 | ~30 tok/s (est. at 50% of peak) | ~9.2 GB → **does not fit** |
| Dense, same params, bf16 + AMX | ~177 tok/s (est.) | ~9.2 GB → does not fit |

On ordinary CPUs, pocket trains **~9× faster** than a dense model of the same size, and it fits where the dense one can't. With AMX, dense matmuls get fast and the speed gap shrinks to ~1.6×, but the memory gap remains. Reproduce with `python scripts/cpu_bench.py --preset pocket --compare-dense [--bf16 off]`.

## 2. The Hive: row-sparse asynchronous DiLoCo (`neuraflex/hive/`)

```
   Kaggle notebook ─┐                                  ┌─ pulls snapshot r
   Oracle free VM ──┤   deltas/<worker>/<r>_<seq>.pt   │  trains ~1 h on its own data shard
   Colab ──────────┼──────────────►  BLOB STORE  ◄────┤  pushes pseudo-gradient
   lab PC / laptop ┘   (HF Hub repo or shared dir)    └─ repeat; safe to kill anytime
                                       │
                           coordinator (any always-on box)
                merges deltas → global/round_{r+1}.pt (bf16), keeps the fp32 master
```

- **Pseudo-gradients, not gradients.** Each worker trains locally for a whole window (default 60 min), then sends `snapshot − its weights` (DiLoCo, Douillard et al. 2023). Communication drops by roughly 1000× compared with per-step gradient sync, so home internet is enough.
- **Row-sparse merge (new).** Dense parameters are averaged, then the coordinator takes an outer Nesterov step (lr 0.7, momentum 0.9). Each **table row is averaged only over the workers that touched it.** A micro-expert that only the Kaggle worker trained is applied at full strength instead of being diluted by other workers' zeros, so machines reading different data can grow different specialists. Only touched rows are sent.
- **Asynchronous and fault-tolerant.** Nobody waits. A delta built on an old snapshot still counts, down-weighted by 1/(1+staleness); anything older than 8 rounds is dropped. A worker killed mid-window loses at most that window, and on restart it reads a **fresh data shard** (the restart count is mixed into the shard choice).
- **Any blob store is the parameter server:** `local:/dir` (shared folder, NFS, a synced drive) or `hub:user/repo` (a private Hugging Face model repo). Writes are atomic in both.

Verified: a 3-process swarm (`tests/test_hive.py` plus a command-line run) merges asynchronously with workers on different rounds, loss drops from 7.96 to 2.78 over 7 rounds, old snapshots are pruned, and a killed worker resumes on the newest global.

### Traffic per worker per window (pocket)
About 1.2 GB downloaded (bf16 snapshot) and about 0.5–0.9 GB uploaded (80 MB dense plus touched rows), roughly once an hour. That's fine on cloud notebooks. Incremental snapshot patches and int8 deltas would cut it 2–4× (TODO).

## 3. Budget

At ~270 tok/s, one 2-core box processes about **23M tokens/day**. A 4-core box does roughly 1.5–2× that.

| Swarm | 1B tokens | 5B tokens |
|---|---|---|
| 1 box | ~6 weeks | ~7 months |
| 10 boxes | ~4–5 days (ideal) | ~3–4 weeks |
| 30 boxes | ~1.5 days | ~1–2 weeks |

DiLoCo-style training usually needs somewhat more tokens than synchronous training for the same loss, so treat these as lower bounds. For scale: small, carefully trained models in the 100–500M range typically see billions to trillions of tokens. Data quality (FineWeb-Edu, textbook-style data) matters more than anything else at this size.

## 4. Where the free CPUs are (checked Nov 2026; limits change, verify before relying on them)

| Source | Spec | Notes |
|---|---|---|
| Kaggle CPU notebooks | 4 cores, ~16–30 GB RAM | 12 h sessions; run the worker in the notebook, restart when it dies |
| Oracle Cloud Always Free (Arm A1) | **now 2 OCPU / 12 GB** (halved in 2026) | always on, which makes it the natural **coordinator** |
| Google Colab (free) | ~2 cores, ~12 GB | sessions disconnect; fine as an occasional worker |
| University / institute HPC CPU nodes | often dozens of cores | ask your department; usually the single biggest free source |
| Your and your friends' laptops | varies | volunteer workers; pocket needs ~5–6 GB free RAM |

**Don't** use CI runners (GitHub Actions etc.) or multiple accounts to farm compute. It breaks those platforms' terms and gets accounts banned. Free Hugging Face CPU *Spaces* reportedly now require PRO, but the Hub *repo* used as the parameter server is unaffected.

## 5. Run it

```bash
pip install -r requirements.txt            # torch, transformers, datasets, huggingface_hub
export HF_TOKEN=hf_...                     # write token; on Kaggle use the Secrets panel

# once, on the always-on box (e.g. Oracle free VM)
python -m neuraflex.hive.coordinator init --transport hub:YOU/neuraflex-hive --preset pocket
python -m neuraflex.hive.coordinator run  --transport hub:YOU/neuraflex-hive --merge-minutes 30

# on every free CPU (give each a unique --worker-id)
python -m neuraflex.hive.worker --transport hub:YOU/neuraflex-hive --worker-id kaggle-1 \
  --tokenizer HuggingFaceTB/SmolLM2-135M --data HuggingFaceFW/fineweb-edu --data-config sample-10BT \
  --seq-len 512 --micro-batch 1 --accum 8 --window-minutes 60
```

To try it on one machine first, use `--transport local:/tmp/hive` with `--preset debug-pkm` and a text file as `--data`.

## 6. What to test first (all CPU-sized)

| # | Question | Pass condition |
|---|---|---|
| H1 | Pocket vs a dense model with the **same compute per token** | lower val loss at equal tokens |
| H2 | Loop-index-aware expert queries vs plain | lower loss |
| H3 | n-gram memory on/off | lower loss for ≤5% slowdown |
| H4 | Row-sparse merge vs plain averaging of sparse rows | lower loss at equal tokens |
| H5 | 8-worker Hive vs 1 machine at equal total tokens | within 10% of single-machine loss |

## 7. Honest limits

- Sparse-memory models are judged **per FLOP**, not per parameter. A 577M pocket model will not match a 577M dense model trained on the same tokens; the question is whether it beats the dense model you could *actually afford* on a CPU (H1).
- Untested at real scale. The numbers above are throughput and correctness, not quality.
- The training step is still Python and PyTorch-op heavy. A fused C++ micro-expert kernel could plausibly give another 2–3×.
- This doesn't make the 52B model CPU-trainable. The role of pocket + Hive is to produce **real results for $0**: a working model, an ablation table, and a paper. Those are what unlock GPU grants for the 3B → 52B plan in `ARCHITECTURE.md`.

## References
Lample et al. 2019 (product-key memory) · He 2024 (PEER, Mixture of a Million Experts) · Berges et al. 2024 (Memory Layers at Scale) · Huang et al. 2025 (Over-Tokenized Transformers) · Grave et al. 2017 (adaptive softmax) · Douillard et al. 2023 (DiLoCo) · Chen et al. 2020 (SLIDE: sparse CPU training beating GPUs) · Mudigere et al. 2021 (row-wise optimizers for huge embedding tables)
