"""Measure CPU training throughput and project wall-clock time for a token budget.

    python scripts/cpu_bench.py --preset pocket --threads 4 --tokens 2e9 --machines 8
    python scripts/cpu_bench.py --preset pocket --compare-dense   # same-capacity dense baseline
"""
import argparse
import sys
import time
from dataclasses import replace

import torch

sys.path.insert(0, ".")
from neuraflex import NeuraFlex, preset  # noqa: E402
from neuraflex.muon import OptimizerGroup  # noqa: E402


def has_amx() -> bool:
    try:
        return "amx_bf16" in open("/proc/cpuinfo").read()
    except OSError:
        return False


def bench(cfg, batch, seq, loops, steps, bf16, with_opt=True, accum=4):
    torch.manual_seed(0)
    m = NeuraFlex(cfg).train()
    opt = OptimizerGroup(m, 1e-3) if with_opt else None
    ids = torch.randint(0, cfg.vocab_size, (batch, seq + 1))
    times = []
    for i in range(steps + 1):
        t = time.perf_counter()
        for _ in range(accum):
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
                out = m(ids[:, :-1], labels=ids[:, 1:], n_loops=loops, return_logits=False)
            (out.loss / accum).backward()
        if opt:
            opt.clip()
            opt.step()
            opt.zero_grad()
        else:
            m.zero_grad(set_to_none=True)
        if i >= 1:  # first step is warm-up
            times.append(time.perf_counter() - t)
    tps = batch * seq * accum / (sum(times) / len(times))
    counts = m.param_counts()
    del m, opt
    return counts, tps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="pocket")
    ap.add_argument("--threads", type=int, default=torch.get_num_threads())
    ap.add_argument("--batch", type=int, default=1, help="micro-batch (sequences)")
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--loops", type=int, default=2)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--bf16", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--tokens", type=float, default=2e9, help="token budget to project")
    ap.add_argument("--machines", type=int, default=1, help="identical machines in the swarm")
    ap.add_argument("--compare-dense", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    bf16 = has_amx() if a.bf16 == "auto" else a.bf16 == "on"

    cfg = preset(a.preset)
    counts, tps = bench(cfg, a.batch, a.seq, a.loops, a.steps, bf16, accum=a.accum)
    print(f"cpu threads={a.threads} amx={has_amx()} bf16={bf16}")
    print(f"[{a.preset}] total={counts['total'] / 1e6:.0f}M  sparse={counts['sparse_tables'] / 1e6:.0f}M  "
          f"active/pass={counts['active_non_embedding_per_pass'] / 1e6:.1f}M  ->  {tps:,.0f} tok/s (train, loops={a.loops})")
    days = a.tokens / (tps * a.machines) / 86400
    print(f"projection: {a.tokens:.2g} tokens on {a.machines} machine(s) ~ {days:,.1f} days")

    if a.compare_dense:
        # Same total parameters, all dense (the GPU design point). Estimated, because on a
        # small box it usually cannot even be instantiated with an optimizer.
        n = 2048
        x = torch.randn(n, n, dtype=torch.bfloat16 if bf16 else torch.float32)
        x @ x
        t = time.perf_counter()
        for _ in range(5):
            x @ x
        flops = 5 * 2 * n ** 3 / (time.perf_counter() - t)
        dense_tps = 0.5 * flops / (6 * counts["total"])  # assume 50% of peak matmul
        ram = 16 * counts["total"] / 1e9               # fp32 weights + grads + 2 Adam moments
        print(f"[dense, same {counts['total'] / 1e6:.0f}M params] ~{dense_tps:,.0f} tok/s at 50% of peak "
              f"({flops / 1e9:.0f} GFLOP/s), needs ~{ram:.1f} GB RAM to train  ->  pocket ~{tps / dense_tps:.1f}x faster")


if __name__ == "__main__":
    main()
