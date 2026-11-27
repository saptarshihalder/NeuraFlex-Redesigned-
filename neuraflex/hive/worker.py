"""Hive worker: run one on every free CPU you can find.

    python -m neuraflex.hive.worker --transport hub:you/neuraflex-hive --worker-id kaggle-1 \
        --data HuggingFaceFW/fineweb-edu --data-config sample-10BT \
        --tokenizer HuggingFaceTB/SmolLM2-135M --window-minutes 60

Loop: pull the newest snapshot -> train for one window -> push the pseudo-gradient -> repeat.
It is safe to kill at any moment (a 12-hour notebook limit, a laptop lid closing);
the worker loses at most one window and resumes on fresh data.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from ..config import NeuraFlexConfig
from ..data import batches, shard_of, token_stream
from ..model import NeuraFlex
from ..muon import OptimizerGroup
from ..sparse import TouchTracker
from .protocol import make_delta, to_bytes
from .transport import Transport, make_transport


def has_amx() -> bool:
    try:
        return "amx_bf16" in open("/proc/cpuinfo").read()
    except OSError:
        return False


class Worker:
    def __init__(self, transport: Transport, worker_id: str, data: str, tokenizer: str | None = None,
                 data_config: str | None = None, text_field: str = "text", seq_len: int = 512,
                 micro_batch: int = 1, accum: int = 8, lr: float = 2e-3, warmup: int = 200,
                 n_shards: int = 1024, state_dir: str = ".hive", bf16: bool | None = None):
        self.t, self.id = transport, worker_id
        self.cfg = NeuraFlexConfig.from_dict(json.loads(self.t.get_bytes("config.json")))
        self.model = NeuraFlex(self.cfg).train()
        self.opt = OptimizerGroup(self.model, lr)
        self.tracker = TouchTracker(self.model)
        self.lr, self.warmup, self.accum = lr, warmup, accum
        self.bf16 = has_amx() if bf16 is None else bf16
        self.state_dir = os.path.join(state_dir, worker_id)
        os.makedirs(self.state_dir, exist_ok=True)
        self.state_path = os.path.join(self.state_dir, "state.json")
        st = json.load(open(self.state_path)) if os.path.exists(self.state_path) else {}
        self.restarts = st.get("restarts", -1) + 1
        self.steps = st.get("steps", 0)
        self.seq = st.get("seq", 0)
        self._save_state()
        # every restart reads a different shard, so dying never means re-reading old data
        shard = shard_of(f"{worker_id}:{self.restarts}", n_shards)
        stream = token_stream(data, tokenizer, data_config, text_field, shard, n_shards)
        self.data = batches(stream, micro_batch, seq_len)
        self.round, self.base, self.snap_path = -1, None, None

    def _save_state(self):
        json.dump({"restarts": self.restarts, "steps": self.steps, "seq": self.seq}, open(self.state_path, "w"))

    def latest(self) -> int:
        b = self.t.get_bytes("global/LATEST")
        return int(b.decode()) if b else -1

    def sync(self) -> int:
        r = self.latest()
        if r < 0:
            raise RuntimeError("no snapshot yet: run `python -m neuraflex.hive.coordinator init` first")
        if r != self.round:
            path = os.path.join(self.state_dir, f"snap_{r:06d}.pt")
            if not self.t.get_file(f"global/round_{r:06d}.pt", path):
                return self.round  # pruned mid-flight; keep the old base this window
            old = self.snap_path
            self.base = torch.load(path, map_location="cpu", mmap=True, weights_only=True)["model"]
            self.round, self.snap_path = r, path
            if old and old != path and os.path.exists(old):
                os.remove(old)
        self.model.load_state_dict(self.base)  # start every window from the global weights
        return self.round

    def run_window(self, max_steps: int | None = None, max_seconds: float | None = None) -> dict:
        self.tracker.reset()
        t0, n, tokens, loss_sum = time.time(), 0, 0, 0.0
        while (max_steps is None or n < max_steps) and (max_seconds is None or time.time() - t0 < max_seconds):
            self.opt.set_lr(self.lr * min(1.0, (self.steps + 1) / self.warmup))
            n_loops = self.model.sample_loops()
            for _ in range(self.accum):
                x, y = next(self.data)
                with torch.autocast("cpu", dtype=torch.bfloat16, enabled=self.bf16):
                    out = self.model(x, labels=y, n_loops=n_loops, return_logits=False)
                (out.loss / self.accum).backward()
                tokens += x.numel()
                loss_sum += out.ce.item() * x.numel()
            self.tracker.update()
            self.opt.clip(1.0)
            self.opt.step()
            self.opt.zero_grad()
            self.model.update_router_biases()
            n += 1
            self.steps += 1
        return {"steps": n, "tokens": tokens, "loss": loss_sum / max(1, tokens),
                "tok_per_s": tokens / max(1e-9, time.time() - t0)}

    def push(self, stats: dict):
        delta = make_delta(self.model, self.base, self.tracker.touched(), stats["tokens"], self.round, stats["loss"])
        self.t.put_bytes(f"deltas/{self.id}/{self.round:06d}_{self.seq:04d}.pt", to_bytes(delta))
        self.seq += 1
        self._save_state()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transport", required=True)
    ap.add_argument("--worker-id", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--data-config", default=None)
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--window-minutes", type=float, default=60)
    ap.add_argument("--window-steps", type=int, default=None, help="overrides --window-minutes")
    ap.add_argument("--windows", type=int, default=0, help="stop after N windows (0 = forever)")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--bf16", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--state-dir", default=".hive")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    w = Worker(make_transport(a.transport), a.worker_id, a.data, a.tokenizer, a.data_config, a.text_field,
               a.seq_len, a.micro_batch, a.accum, a.lr, a.warmup, state_dir=a.state_dir,
               bf16=None if a.bf16 == "auto" else a.bf16 == "on")
    print(f"[{a.worker_id}] threads={a.threads} bf16={w.bf16} restart #{w.restarts}", flush=True)
    k = 0
    while a.windows == 0 or k < a.windows:
        r = w.sync()
        stats = w.run_window(max_steps=a.window_steps,
                             max_seconds=None if a.window_steps else a.window_minutes * 60)
        w.push(stats)
        print(f"[{a.worker_id}] round {r} | {stats['steps']} steps | loss {stats['loss']:.4f} | "
              f"{stats['tok_per_s']:.0f} tok/s | touched rows {sum(v.numel() for v in w.tracker.touched().values()):,}",
              flush=True)
        k += 1


if __name__ == "__main__":
    main()
