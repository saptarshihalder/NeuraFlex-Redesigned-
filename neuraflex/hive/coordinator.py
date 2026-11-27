"""Hive coordinator: merges worker pseudo-gradients into the global model.

Runs anywhere that stays up (an always-free VM, a Raspberry Pi, a cron job):
    python -m neuraflex.hive.coordinator init --transport hub:you/neuraflex-hive --preset pocket
    python -m neuraflex.hive.coordinator run  --transport hub:you/neuraflex-hive --merge-minutes 30
It needs ~1x model size in RAM and no fast CPU: merging is additions, not training.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict

import torch

from ..config import NeuraFlexConfig, preset
from ..model import NeuraFlex
from .protocol import from_bytes, merge, to_bytes
from .transport import Transport, make_transport


class Coordinator:
    def __init__(self, transport: Transport, state_dir: str = ".hive/coordinator", lr: float = 0.7,
                 momentum: float = 0.9, sparse_lr: float = 1.0, max_staleness: int = 8, keep: int = 3,
                 snapshot_dtype: torch.dtype = torch.bfloat16):
        self.t, self.dir = transport, state_dir
        os.makedirs(state_dir, exist_ok=True)
        self.lr, self.momentum, self.sparse_lr = lr, momentum, sparse_lr
        self.max_staleness, self.keep, self.dtype = max_staleness, keep, snapshot_dtype
        self.master, self.outer, self.round = None, {}, -1

    # ---------------------------------------------------------------- state
    def init(self, cfg: NeuraFlexConfig, seed: int = 0):
        assert not cfg.tie_embeddings, "Hive syncs parameters by name; use untied embeddings"
        torch.manual_seed(seed)
        self.t.put_bytes("config.json", json.dumps(asdict(cfg), indent=2).encode())
        self.master = {k: v.float().clone() for k, v in NeuraFlex(cfg).state_dict().items()}
        self.outer, self.round = {}, 0
        self._save_local()
        self._publish()

    def load(self):
        path = os.path.join(self.dir, "master.pt")
        if os.path.exists(path):  # fp32 master + optimizer momentum survive restarts
            st = torch.load(path, map_location="cpu", weights_only=True)
            self.master, self.outer, self.round = st["master"], st["outer"], st["round"]
        else:  # recover from the last published snapshot
            self.round = int(self.t.get_bytes("global/LATEST").decode())
            snap = from_bytes(self.t.get_bytes(f"global/round_{self.round:06d}.pt"))
            self.master = {k: v.float() for k, v in snap["model"].items()}

    def _save_local(self):
        tmp = os.path.join(self.dir, "master.pt.tmp")
        torch.save({"master": self.master, "outer": self.outer, "round": self.round}, tmp)
        os.replace(tmp, os.path.join(self.dir, "master.pt"))

    def _publish(self):
        sd = {k: v.to(self.dtype) if v.is_floating_point() else v for k, v in self.master.items()}
        self.t.put_bytes(f"global/round_{self.round:06d}.pt", to_bytes({"model": sd, "round": self.round}))
        self.t.put_bytes("global/LATEST", str(self.round).encode())
        for old in self.t.list("global/round_"):
            if int(old.rsplit("_", 1)[-1].split(".")[0]) <= self.round - self.keep:
                self.t.delete(old)

    # ---------------------------------------------------------------- merge
    def pending(self) -> list[str]:
        return [p for p in self.t.list("deltas/") if p.endswith(".pt")]

    def step(self) -> dict | None:
        files = self.pending()
        if not files:
            return None
        deltas = []
        for f in files:
            b = self.t.get_bytes(f)
            if b:
                deltas.append(from_bytes(b))
        stats = merge(self.master, deltas, self.round, self.outer, self.lr, self.momentum,
                      self.sparse_lr, self.max_staleness)
        for f in files:  # consumed (or too stale to matter)
            self.t.delete(f)
        if stats["used"] == 0:
            return stats
        self.round += 1
        self._save_local()
        self._publish()
        stats["round"] = self.round
        with open(os.path.join(self.dir, "log.jsonl"), "a") as fh:
            fh.write(json.dumps({**stats, "time": time.time()}) + "\n")
        return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["init", "run", "merge"])
    ap.add_argument("--transport", required=True)
    ap.add_argument("--preset", default="pocket")
    ap.add_argument("--config", default=None)
    ap.add_argument("--state-dir", default=".hive/coordinator")
    ap.add_argument("--merge-minutes", type=float, default=30)
    ap.add_argument("--min-deltas", type=int, default=1)
    ap.add_argument("--outer-lr", type=float, default=0.7)
    ap.add_argument("--outer-momentum", type=float, default=0.9)
    ap.add_argument("--max-staleness", type=int, default=8)
    a = ap.parse_args()
    c = Coordinator(make_transport(a.transport), a.state_dir, a.outer_lr, a.outer_momentum,
                    max_staleness=a.max_staleness)
    if a.cmd == "init":
        cfg = NeuraFlexConfig.from_json(a.config) if a.config else preset(a.preset)
        c.init(cfg)
        print(f"initialised round 0 ({a.preset})")
        return
    c.load()
    while True:
        if len(c.pending()) >= a.min_deltas:
            s = c.step()
            if s and s.get("used"):
                print(f"round {s['round']} | merged {s['used']} deltas | {s['tokens']:,} tokens | "
                      f"loss {s['loss']:.4f} | sparse rows {s['sparse_rows']:,}", flush=True)
        if a.cmd == "merge":
            return
        time.sleep(a.merge_minutes * 60)


if __name__ == "__main__":
    main()
