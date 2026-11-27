"""Train NeuraFlex-Ouroboros.

Single GPU / CPU smoke test:
    python train.py --preset debug --data sample.txt --steps 200

Multi-node (FSDP2):
    torchrun --nnodes N --nproc-per-node 8 train.py --preset 50b \
        --tokenizer Qwen/Qwen3-8B --data HuggingFaceFW/fineweb-edu --seq-len 4096 \
        --batch-size 1 --grad-accum 16 --steps 250000

Distil from an open-weight teacher that shares the tokenizer:
    ... --teacher Qwen/Qwen3-32B --kd-weight 0.5

CPU (single machine; for many machines use the Hive swarm, see docs/CPU_HIVE.md):
    python train.py --preset pocket --tokenizer HuggingFaceTB/SmolLM2-135M \
        --data HuggingFaceFW/fineweb-edu --data-config sample-10BT --batch-size 1 --grad-accum 8 --seq-len 512
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

from neuraflex import NeuraFlex, preset
from neuraflex.data import batches, token_stream
from neuraflex.muon import OptimizerGroup
from neuraflex.sparse import split_sparse


# ------------------------------------------------------------------------ schedule
def wsd_lr(step, args):
    """Warmup-Stable-Decay: lets you branch a cooldown from any checkpoint."""
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    decay_start = int(args.steps * (1 - args.decay_frac))
    if step < decay_start:
        return args.lr
    p = (step - decay_start) / max(1, args.steps - decay_start)
    return args.lr * (args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * p)))


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="debug")
    ap.add_argument("--config", default=None, help="JSON config overriding the preset")
    ap.add_argument("--data", required=True, help="local text file or HF dataset name")
    ap.add_argument("--data-config", default=None)
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--decay-frac", type=float, default=0.2)
    ap.add_argument("--min-lr-frac", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--optimizer", choices=["muon", "adamw"], default="muon")
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--kd-weight", type=float, default=0.5)
    ap.add_argument("--kd-temp", type=float, default=1.0)
    ap.add_argument("--init", default=None, help="checkpoint to start from (e.g. after expert_mitosis)")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        rank, world = 0, 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.threads:
        torch.set_num_threads(args.threads)
    master = rank == 0
    torch.manual_seed(args.seed + rank)

    cfg = preset(args.preset)
    if args.config:
        cfg = type(cfg).from_json(args.config)
    if args.tokenizer is None:
        cfg.vocab_size = max(cfg.vocab_size, 256)

    with torch.device("meta" if distributed else device):
        model = NeuraFlex(cfg)
    if args.init:
        sd = torch.load(args.init, map_location="cpu")
        model.load_state_dict(sd.get("model", sd), assign=distributed)

    if distributed:
        assert not split_sparse(model)[1], "sparse-table presets are CPU/Hive only (FSDP needs dense params)"
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
        mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        for blk in [*model.prelude, *model.core, *model.coda, *model.mtp]:
            fully_shard(blk, mp_policy=mp)
        fully_shard(model, mp_policy=mp)
        model.to_empty(device=device)
        if not args.init:
            model.reset_parameters()
    if master:
        counts = model.param_counts()
        print(json.dumps({k: f"{v / 1e9:.3f}B" for k, v in counts.items()}, indent=1))

    opt = OptimizerGroup(model, args.lr, args.weight_decay, args.optimizer)

    teacher = None
    if args.teacher:
        from transformers import AutoModelForCausalLM
        teacher = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=torch.bfloat16).to(device).eval()

    fwd = torch.compile(model) if args.compile else model
    data = batches(token_stream(args.data, args.tokenizer, args.data_config, args.text_field, rank, world),
                   args.batch_size, args.seq_len, device)
    loop_gen = torch.Generator().manual_seed(args.seed)  # identical loop counts on every rank
    amx = device.type == "cpu" and "amx_bf16" in open("/proc/cpuinfo").read() if os.path.exists("/proc/cpuinfo") else False
    autocast = torch.autocast(device.type, dtype=torch.bfloat16,
                              enabled=(device.type == "cuda" and not distributed) or amx)
    os.makedirs(args.out, exist_ok=True)
    if master:
        cfg.to_json(os.path.join(args.out, "config.json"))

    model.train()
    t0 = time.time()
    for step in range(args.steps):
        lr = wsd_lr(step, args)
        opt.set_lr(lr)
        n_loops = model.sample_loops(loop_gen)
        logs = {"loss": 0.0, "ce": 0.0, "mtp": 0.0, "kd": 0.0}
        for _ in range(args.grad_accum):
            x, y = next(data)
            with autocast:
                out = fwd(x, labels=y, n_loops=n_loops, return_logits=teacher is not None)
                loss = out.loss
                if teacher is not None:
                    with torch.no_grad():
                        t_logits = teacher(x).logits
                    V = min(t_logits.shape[-1], out.logits.shape[-1])
                    T = args.kd_temp
                    kd = F.kl_div(F.log_softmax(out.logits[..., :V].float() / T, -1),
                                  F.log_softmax(t_logits[..., :V].float() / T, -1),
                                  log_target=True, reduction="batchmean") * T * T / x.shape[1]
                    loss = loss + args.kd_weight * kd
                    logs["kd"] += kd.item() / args.grad_accum
            (loss / args.grad_accum).backward()
            logs["loss"] += loss.item() / args.grad_accum
            logs["ce"] += out.ce.item() / args.grad_accum
            logs["mtp"] += out.mtp.item() / args.grad_accum
        gnorm = opt.clip(1.0)
        opt.step()
        opt.zero_grad()
        model.update_router_biases()

        if master and step % args.log_every == 0:
            dt = time.time() - t0
            tok = args.batch_size * args.seq_len * args.grad_accum * world * (step + 1)
            print(f"step {step:6d} | loops {n_loops:2d} | loss {logs['loss']:.4f} ce {logs['ce']:.4f} "
                  f"mtp {logs['mtp']:.4f} kd {logs['kd']:.4f} | gnorm {float(gnorm):.2f} | lr {lr:.2e} | "
                  f"{tok / dt:,.0f} tok/s", flush=True)
        if (step + 1) % args.save_every == 0 or step + 1 == args.steps:
            save(model, opt, step, args, distributed, master)
    if distributed:
        dist.destroy_process_group()


def save(model, opt, step, args, distributed, master):
    path = os.path.join(args.out, f"step_{step + 1}")
    if distributed:
        import torch.distributed.checkpoint as dcp
        dcp.save({"model": model.state_dict()}, checkpoint_id=path)
    elif master:
        torch.save({"model": model.state_dict(), "step": step + 1}, path + ".pt")
    if master:
        print(f"saved {path}")


if __name__ == "__main__":
    main()
