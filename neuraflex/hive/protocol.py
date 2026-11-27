"""Row-sparse asynchronous DiLoCo.

Layout on the transport:
    config.json                              model config
    global/LATEST                            current round number
    global/round_000042.pt                   snapshot {"model": state_dict, "round": 42, "meta": {...}}
    deltas/<worker>/<base>_<seq>.pt          pseudo-gradient a worker pushed

Pseudo-gradient = (snapshot the worker started from) - (worker's weights after a window).

* Dense params: weighted average, then an outer Nesterov step (DiLoCo: lr 0.7, momentum 0.9).
* Sparse tables: each row is averaged ONLY over the workers that touched it. A row one
  worker trained is applied at full strength instead of being diluted by other
  workers' zeros, so different machines can grow different expert specialisations.
  Only touched rows are ever sent.
* Async: a delta built on an older snapshot is still used, down-weighted by 1/(1+staleness).
  Workers never wait; a machine that dies loses at most one window.
"""
from __future__ import annotations

import io

import torch

from ..sparse import split_sparse


def snapshot_round(path: str) -> int:
    return int(path.rsplit("_", 1)[-1].split(".")[0])


def to_bytes(obj) -> bytes:
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def from_bytes(data: bytes):
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)


def param_kinds(model: torch.nn.Module):
    dense, sparse = split_sparse(model)
    buffers = [n for n, _ in model.named_buffers() if n in model.state_dict()]
    return [n for n, _ in dense], [n for n, _ in sparse], buffers


@torch.no_grad()
def make_delta(model, base: dict, touched: dict, tokens: int, base_round: int, loss: float,
               dtype=torch.bfloat16) -> dict:
    dense_names, sparse_names, buffer_names = param_kinds(model)
    sd = model.state_dict()
    dense = {n: (base[n].float() - sd[n].float()).to(dtype) for n in dense_names}
    sparse = {}
    for n in sparse_names:
        idx = touched.get(n)
        if idx is None or idx.numel() == 0:
            continue
        sparse[n] = (idx, (base[n][idx].float() - sd[n][idx].float()).to(dtype))
    buffers = {n: sd[n].clone() for n in buffer_names}
    return {"base_round": base_round, "tokens": tokens, "loss": loss,
            "dense": dense, "sparse": sparse, "buffers": buffers}


@torch.no_grad()
def merge(master: dict, deltas: list, current_round: int, outer: dict,
          lr: float = 0.7, momentum: float = 0.9, sparse_lr: float = 1.0, max_staleness: int = 8) -> dict:
    """Apply a batch of worker deltas to the fp32 master state (in place). Returns stats."""
    used = []
    for d in deltas:
        stale = current_round - d["base_round"]
        if 0 <= stale <= max_staleness and d["tokens"] > 0:
            used.append((d, d["tokens"] / (1.0 + stale)))
    if not used:
        return {"used": 0}
    wsum = sum(w for _, w in used)

    # dense: weighted mean pseudo-gradient -> outer Nesterov SGD
    for name in used[0][0]["dense"]:
        g = sum(w * d["dense"][name].float() for d, w in used) / wsum
        buf = outer.get(name)
        buf = g.clone() if buf is None else buf.mul_(momentum).add_(g)
        outer[name] = buf
        master[name] -= lr * (g + momentum * buf)

    # sparse: per-row mean over the workers that touched the row
    rows_updated = 0
    names = {n for d, _ in used for n in d["sparse"]}
    for name in names:
        idxs, vals, ws = [], [], []
        for d, w in used:
            if name in d["sparse"]:
                i, v = d["sparse"][name]
                idxs.append(i)
                vals.append(v.float() * w)
                ws.append(torch.full((i.numel(),), w))
        idx, val, wt = torch.cat(idxs), torch.cat(vals), torch.cat(ws)
        uniq, inv = torch.unique(idx, return_inverse=True)
        num = torch.zeros(uniq.numel(), val.shape[1]).index_add_(0, inv, val)
        den = torch.zeros(uniq.numel()).index_add_(0, inv, wt)
        master[name][uniq] -= sparse_lr * (num / den[:, None]).to(master[name].dtype)
        rows_updated += uniq.numel()

    # buffers (e.g. MoE balancing biases): weighted average of worker values
    for name in used[0][0]["buffers"]:
        master[name] = (sum(w * d["buffers"][name].float() for d, w in used) / wsum).to(master[name].dtype)

    toks = sum(d["tokens"] for d, _ in used)
    loss = sum(d["loss"] * d["tokens"] for d, _ in used) / max(1, toks)
    return {"used": len(used), "tokens": toks, "loss": loss, "sparse_rows": rows_updated}
