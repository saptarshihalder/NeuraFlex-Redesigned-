"""Depth-Routed Experts (DRE).

One expert pool per core layer, re-used on every loop iteration. The router sees the
token state *and* a sinusoidal embedding of the loop index, so the same token can be
sent to different experts on each pass: K experts x R loops gives a combinatorial
number of "virtual layers" from a fixed parameter budget.

Balancing follows DeepSeek-V3: a non-gradient per-expert bias steers top-k selection
(aux-loss-free), plus a tiny sequence-level aux loss as a safety net.
"""
from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .config import NeuraFlexConfig
from .layers import SwiGLU


def loop_features(i: int, dim: int, device, dtype=torch.float32) -> torch.Tensor:
    """Sinusoidal features of the loop index; extrapolate to unseen loop counts."""
    half = dim // 2
    freqs = torch.exp(-math.log(1000.0) * torch.arange(half, device=device, dtype=torch.float32) / half)
    ang = (i + 1) * freqs
    return torch.cat([ang.sin(), ang.cos()]).to(dtype)


class DepthRoutedMoE(nn.Module):
    def __init__(self, cfg: NeuraFlexConfig):
        super().__init__()
        self.cfg = cfg
        E, d, f = cfg.n_experts, cfg.d_model, cfg.expert_dim
        self.router = nn.Linear(d + cfg.loop_emb_dim, E, bias=False)
        self.register_buffer("expert_bias", torch.zeros(E))
        self.register_buffer("load", torch.zeros(E), persistent=False)
        self.w1 = nn.Parameter(torch.empty(E, d, f))
        self.w3 = nn.Parameter(torch.empty(E, d, f))
        self.w2 = nn.Parameter(torch.empty(E, f, d))
        fs = cfg.shared_expert_dim or f * cfg.n_shared_experts
        self.shared = SwiGLU(d, fs) if cfg.n_shared_experts > 0 else None

    def forward(self, x: torch.Tensor, loop_feat: torch.Tensor):
        cfg = self.cfg
        B, T, d = x.shape
        E, K = cfg.n_experts, cfg.top_k
        h = x.reshape(-1, d)
        N = h.shape[0]

        r_in = torch.cat([h, loop_feat.to(h.dtype).expand(N, -1)], dim=-1)
        scores = torch.sigmoid(self.router(r_in).float())                     # (N, E)
        _, idx = (scores + self.expert_bias).topk(K, dim=-1)                  # bias only steers selection
        w = scores.gather(1, idx)
        w = (w / w.sum(-1, keepdim=True).clamp_min(1e-9)).to(h.dtype)        # (N, K)

        aux = h.new_zeros((), dtype=torch.float32)
        if self.training:
            counts = torch.bincount(idx.flatten(), minlength=E)
            self.load += counts.float()
            if cfg.seq_aux_loss_weight > 0:
                p = (scores / scores.sum(-1, keepdim=True)).view(B, T, E).mean(1)          # (B, E)
                onehot = F.one_hot(idx.view(B, T * K), E).float().sum(1)                 # (B, E)
                f_ = onehot * E / (K * T)
                aux = (f_ * p).sum(-1).mean()

        out = torch.zeros_like(h)
        flat = idx.flatten()
        order = flat.argsort()
        tok = order // K
        wflat = w.flatten()[order]
        counts_l = torch.bincount(flat, minlength=E).tolist()
        start = 0
        for e, c in enumerate(counts_l):
            if c == 0:
                continue
            t = tok[start:start + c]
            xe = h[t]
            ye = (F.silu(xe @ self.w1[e]) * (xe @ self.w3[e])) @ self.w2[e]
            out.index_add_(0, t, ye * wflat[start:start + c, None])
            start += c
        if self.shared is not None:
            out = out + self.shared(h)
        return out.view(B, T, d), aux

    @torch.no_grad()
    def update_bias(self, rate: float):
        load = self.load
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(load)
        if load.sum() > 0:
            self.expert_bias += rate * torch.sign(load.mean() - load)
        self.load.zero_()
