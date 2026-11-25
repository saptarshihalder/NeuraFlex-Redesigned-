"""Print parameter and per-token compute budgets for a preset (meta device, no memory)."""
import sys

import torch

sys.path.insert(0, ".")
from neuraflex import NeuraFlex, preset  # noqa: E402


def main(name="50b"):
    cfg = preset(name)
    with torch.device("meta"):
        m = NeuraFlex(cfg)
    c = m.param_counts()
    fixed = sum(p.numel() for mod in (*m.prelude, *m.coda) for p in mod.parameters()) + m.adapter.weight.numel()
    core_active = c["active_non_embedding_per_pass"] - fixed
    print(f"preset            : {name}")
    for k, v in c.items():
        print(f"{k:<18}: {v / 1e9:7.2f}B")
    print(f"unique layers     : {cfg.n_prelude + cfg.n_core + cfg.n_coda}")
    for r in (1, 4, 8, 16):
        act = fixed + r * core_active + cfg.d_model * cfg.vocab_size
        print(f"loops={r:<3} depth={cfg.n_prelude + r * cfg.n_core + cfg.n_coda:<4} "
              f"active params/token ~ {act / 1e9:5.2f}B  (~{2 * act / 1e9:5.1f} GFLOP fwd/token, excl. attention)")


if __name__ == "__main__":
    main(*sys.argv[1:])
