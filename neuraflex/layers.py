"""Core layers: RMSNorm, RoPE, SwiGLU, Multi-head Latent Attention, Gated DeltaNet."""
from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NeuraFlexConfig

try:  # fused chunked kernels (https://github.com/fla-org/flash-linear-attention)
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as _fla_gdr
except Exception:  # pragma: no cover - optional dependency
    _fla_gdr = None


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


# ---------------------------------------------------------------------------
# Rotary embeddings
# ---------------------------------------------------------------------------
class Rotary(nn.Module):
    def __init__(self, dim: int, theta: float):
        super().__init__()
        inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, pos: torch.Tensor):
        f = pos.float()[:, None] * self.inv_freq[None, :]
        return f.cos(), f.sin()  # (T, dim/2)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, T, H, D). Half-split rotation."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d].float(), x[..., d:].float()
    c, s = cos[None, :, None, :], sin[None, :, None, :]
    out = torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)
    return out.to(x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, d: int, f: int):
        super().__init__()
        self.w1 = nn.Linear(d, f, bias=False)
        self.w3 = nn.Linear(d, f, bias=False)
        self.w2 = nn.Linear(f, d, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def _causal_mask(T: int, S: int, device) -> torch.Tensor | None:
    """Boolean mask (T, S): query i sits at absolute slot S-T+i."""
    if T == 1:
        return None
    q = torch.arange(S - T, S, device=device)[:, None]
    k = torch.arange(S, device=device)[None, :]
    return k <= q


# ---------------------------------------------------------------------------
# Multi-head Latent Attention (DeepSeek-V2/V3 style, compressed KV cache)
# ---------------------------------------------------------------------------
class MLA(nn.Module):
    def __init__(self, cfg: NeuraFlexConfig):
        super().__init__()
        self.cfg = cfg
        H = cfg.n_heads
        self.wq_a = nn.Linear(cfg.d_model, cfg.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(cfg.q_lora_rank, cfg.norm_eps)
        self.wq_b = nn.Linear(cfg.q_lora_rank, H * cfg.qk_head_dim, bias=False)
        self.wkv_a = nn.Linear(cfg.d_model, cfg.kv_lora_rank + cfg.qk_rope_head_dim, bias=False)
        self.kv_norm = RMSNorm(cfg.kv_lora_rank, cfg.norm_eps)
        self.wkv_b = nn.Linear(cfg.kv_lora_rank, H * (cfg.qk_nope_head_dim + cfg.v_head_dim), bias=False)
        self.wo = nn.Linear(H * cfg.v_head_dim, cfg.d_model, bias=False)
        self.scale = cfg.qk_head_dim ** -0.5

    def forward(self, x, rope, cache: dict | None = None):
        cfg = self.cfg
        B, T, _ = x.shape
        H, dn, dr, dv = cfg.n_heads, cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim
        cos, sin = rope

        q = self.wq_b(self.q_norm(self.wq_a(x))).view(B, T, H, dn + dr)
        q_nope, q_pe = q.split([dn, dr], dim=-1)
        q_pe = apply_rope(q_pe, cos, sin)

        c_kv, k_pe = self.wkv_a(x).split([cfg.kv_lora_rank, dr], dim=-1)
        c_kv = self.kv_norm(c_kv)
        k_pe = apply_rope(k_pe.unsqueeze(2), cos, sin)  # (B, T, 1, dr)

        if cache is not None:
            if "c_kv" in cache:
                c_kv = torch.cat([cache["c_kv"], c_kv], dim=1)
                k_pe = torch.cat([cache["k_pe"], k_pe], dim=1)
            cache["c_kv"], cache["k_pe"] = c_kv, k_pe

        S = c_kv.shape[1]
        k_nope, v = self.wkv_b(c_kv).view(B, S, H, dn + dv).split([dn, dv], dim=-1)
        k = torch.cat([k_nope, k_pe.expand(B, S, H, dr)], dim=-1)
        q = torch.cat([q_nope, q_pe], dim=-1)

        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        if T == S:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=T > 1, scale=self.scale)
        else:
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=_causal_mask(T, S, x.device), scale=self.scale)
        return self.wo(o.transpose(1, 2).reshape(B, T, H * dv))


# ---------------------------------------------------------------------------
# Gated DeltaNet (Yang et al., 2024) -- O(1) state per token, used in 3/4 core layers
# ---------------------------------------------------------------------------
def delta_rule_reference(q, k, v, g, beta, state=None):
    """Recurrent gated delta rule.

    q,k: (B,T,H,Dk) L2-normalised; v: (B,T,H,Dv); g: (B,T,H) log-decay; beta: (B,T,H).
    S_t = a_t S_{t-1} + b_t (v_t - a_t S_{t-1} k_t) k_t^T ;  o_t = S_t q_t
    """
    B, T, H, Dk = q.shape
    Dv = v.shape[-1]
    S = torch.zeros(B, H, Dv, Dk, device=q.device, dtype=torch.float32) if state is None else state
    q, k, v, g, beta = (t.float() for t in (q, k, v, g, beta))
    q = q * Dk ** -0.5
    outs = []
    for t in range(T):
        S = S * g[:, t].exp()[..., None, None]
        kt = k[:, t]
        err = v[:, t] - torch.einsum("bhvk,bhk->bhv", S, kt)
        S = S + torch.einsum("bhv,bhk->bhvk", beta[:, t, :, None] * err, kt)
        outs.append(torch.einsum("bhvk,bhk->bhv", S, q[:, t]))
    return torch.stack(outs, dim=1), S


class GatedDeltaNet(nn.Module):
    _warned = False

    def __init__(self, cfg: NeuraFlexConfig):
        super().__init__()
        H, D = cfg.gdn_heads, cfg.gdn_head_dim
        self.H, self.D, self.K = H, D, cfg.gdn_conv_size
        d = cfg.d_model
        self.q_proj = nn.Linear(d, H * D, bias=False)
        self.k_proj = nn.Linear(d, H * D, bias=False)
        self.v_proj = nn.Linear(d, H * D, bias=False)
        C = 3 * H * D
        self.conv = nn.Conv1d(C, C, self.K, groups=C, bias=False)
        self.b_proj = nn.Linear(d, H, bias=False)
        self.a_proj = nn.Linear(d, H, bias=False)
        self.A_log = nn.Parameter(torch.empty(H).uniform_(1, 16).log())
        self.dt_bias = nn.Parameter(torch.zeros(H))
        self.g_proj = nn.Linear(d, H * D, bias=False)
        self.o_norm = RMSNorm(D, cfg.norm_eps)
        self.o_proj = nn.Linear(H * D, d, bias=False)

    def forward(self, x, rope=None, cache: dict | None = None):
        B, T, _ = x.shape
        H, D, K = self.H, self.D, self.K
        qkv = torch.cat([self.q_proj(x), self.k_proj(x), self.v_proj(x)], dim=-1).transpose(1, 2)
        prev = cache.get("conv") if cache is not None else None
        inp = torch.cat([prev, qkv], dim=-1) if prev is not None else F.pad(qkv, (K - 1, 0))
        if cache is not None:
            cache["conv"] = inp[..., -(K - 1):].detach()
        qkv = F.silu(F.conv1d(inp, self.conv.weight, groups=inp.shape[1])).transpose(1, 2)
        q, k, v = (t.reshape(B, T, H, D) for t in qkv.split(H * D, dim=-1))
        q, k = F.normalize(q.float(), dim=-1), F.normalize(k.float(), dim=-1)
        beta = torch.sigmoid(self.b_proj(x)).float()
        g = -self.A_log.float().exp() * F.softplus(self.a_proj(x).float() + self.dt_bias)

        if cache is None and x.is_cuda and _fla_gdr is not None:
            try:
                o, _ = _fla_gdr(q.to(x.dtype), k.to(x.dtype), v, g, beta.to(x.dtype), scale=D ** -0.5)
            except Exception as e:  # pragma: no cover
                if not GatedDeltaNet._warned:
                    warnings.warn(f"fla kernel failed ({e}); falling back to reference recurrence")
                    GatedDeltaNet._warned = True
                o, _ = delta_rule_reference(q, k, v, g, beta)
        else:
            o, S = delta_rule_reference(q, k, v, g, beta, cache.get("S") if cache is not None else None)
            if cache is not None:
                cache["S"] = S
        o = self.o_norm(o.to(x.dtype)) * F.silu(self.g_proj(x).view(B, T, H, D))
        return self.o_proj(o.reshape(B, T, H * D))
