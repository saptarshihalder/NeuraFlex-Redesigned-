"""CPU-native sparse memory modules.

The GPU design point is "FLOPs are cheap, memory is scarce", so GPU models spend
their capacity on dense matmuls. A free CPU box is the opposite: few FLOP/s, but
gigabytes of RAM, fast random access, and no penalty for irregular, branchy work.
These modules put capacity into lookup tables, so each token touches only a few
kilobytes of a table that can hold hundreds of millions of parameters:

* ProductKeyExperts: Depth-Routed Product-Key Micro-Experts. n_keys**2 single-neuron
  experts per layer, selected in O(sqrt(N)) with product keys (Lample et al. 2019;
  PEER, He 2024). The query also sees the loop index, so each pass of the looped core
  retrieves different experts.
* HashNgramEmbedding: hashed 2/3-gram input embeddings (cf. Over-Tokenized
  Transformers, Huang et al. 2025). It gives phrase-level memory for zero matmul FLOPs.

Both use sparse gradients. Only the touched rows get gradients, optimizer updates and
(in the Hive swarm) network traffic.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import NeuraFlexConfig


class ProductKeyExperts(nn.Module):
    def __init__(self, cfg: NeuraFlexConfig):
        super().__init__()
        self.cfg = cfg
        d, H, nk, kd = cfg.d_model, cfg.pkm_heads, cfg.pkm_n_keys, cfg.pkm_key_dim
        assert kd % 2 == 0
        self.n_experts = nk * nk
        self.query = nn.Linear(d + cfg.loop_emb_dim, H * kd, bias=False)
        self.keys = nn.Parameter(torch.empty(H, 2, nk, kd // 2))
        self.down = nn.Embedding(self.n_experts, d, sparse=True)                 # u_i
        self.up = nn.EmbeddingBag(self.n_experts, d, mode="sum", sparse=True)    # v_i

    def route(self, h: torch.Tensor, loop_feat: torch.Tensor):
        cfg = self.cfg
        N = h.shape[0]
        H, nk, kd, k = cfg.pkm_heads, cfg.pkm_n_keys, cfg.pkm_key_dim, cfg.pkm_topk
        q = self.query(torch.cat([h, loop_feat.to(h.dtype).expand(N, -1)], dim=-1))
        q = q.view(N, H, 2, kd // 2).float()
        q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + 1e-6)
        s = torch.einsum("nhpc,hpkc->nhpk", q, self.keys.float())               # (N, H, 2, nk)
        s1, i1 = s[:, :, 0].topk(k, dim=-1)
        s2, i2 = s[:, :, 1].topk(k, dim=-1)
        cand = (s1[..., :, None] + s2[..., None, :]).flatten(-2)                 # (N, H, k*k)
        cidx = (i1[..., :, None] * nk + i2[..., None, :]).flatten(-2)
        sc, j = cand.topk(k, dim=-1)
        idx = cidx.gather(-1, j)                                                 # (N, H, k)
        return idx, torch.softmax(sc, dim=-1)

    def forward(self, x: torch.Tensor, loop_feat: torch.Tensor):
        if self.training and torch.is_grad_enabled():
            # recompute the (N, H*k, d) gather in backward instead of storing it
            return checkpoint(self._forward, x, loop_feat, use_reentrant=False), None
        return self._forward(x, loop_feat), None

    def _forward(self, x: torch.Tensor, loop_feat: torch.Tensor):
        B, T, d = x.shape
        h = x.reshape(-1, d)
        idx, gate = self.route(h, loop_feat)
        idx = idx.reshape(h.shape[0], -1)                                        # (N, H*k)
        u = self.down(idx)                                                       # (N, H*k, d)
        act = F.gelu(torch.einsum("nd,nkd->nk", h, u.to(h.dtype)))
        w = (act * gate.reshape(h.shape[0], -1).to(act.dtype)).float()
        out = self.up(idx, per_sample_weights=w)                                 # never materialises V rows
        return out.to(x.dtype).view(B, T, d)


_PRIMES = (1_000_003, 998_244_353, 2_147_483_647, 1_610_612_741)


class HashNgramEmbedding(nn.Module):
    def __init__(self, cfg: NeuraFlexConfig):
        super().__init__()
        self.orders = tuple(cfg.ngram_orders)
        self.M = cfg.ngram_table_size
        self.tables = nn.ModuleList(nn.Embedding(self.M, cfg.ngram_dim, sparse=True) for _ in self.orders)
        self.proj = nn.Linear(cfg.ngram_dim, cfg.d_model, bias=False)
        self.pad = cfg.vocab_size  # stands in for "before the start of the sequence"

    @property
    def context(self) -> int:
        return max(self.orders) - 1

    def hashes(self, ids: torch.Tensor, ctx: torch.Tensor | None):
        """ids: (B, T); ctx: (B, <=context) previous tokens or None."""
        B, T = ids.shape
        c = self.context
        if ctx is None:
            ctx = ids.new_full((B, c), self.pad)
        elif ctx.shape[1] < c:
            ctx = torch.cat([ids.new_full((B, c - ctx.shape[1]), self.pad), ctx], dim=1)
        full = torch.cat([ctx[:, -c:], ids], dim=1).long()
        out = []
        for o in self.orders:
            h = torch.zeros(B, T, dtype=torch.long, device=ids.device)
            for j in range(o):  # token at t-j
                tok = full[:, c - j: c - j + T]
                h = h ^ (tok * _PRIMES[j] + o * 7919)
            out.append(h.remainder(self.M))
        return out, full[:, -c:]

    def forward(self, ids: torch.Tensor, ctx: torch.Tensor | None = None):
        hs, new_ctx = self.hashes(ids, ctx)
        e = sum(tbl(h) for tbl, h in zip(self.tables, hs))
        return self.proj(e), new_ctx


# ---------------------------------------------------------------------------
def is_sparse_module(m: nn.Module) -> bool:
    return isinstance(m, (nn.Embedding, nn.EmbeddingBag)) and getattr(m, "sparse", False)


def split_sparse(model: nn.Module):
    """Returns (dense_named_params, sparse_named_params)."""
    sparse_ids = {id(m.weight) for m in model.modules() if is_sparse_module(m)}
    dense, sparse = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            (sparse if id(p) in sparse_ids else dense).append((n, p))
    return dense, sparse


class TouchTracker:
    """Records which rows of each sparse table received gradient (for sparse sync)."""

    def __init__(self, model: nn.Module):
        _, sparse = split_sparse(model)
        self.params = dict(sparse)
        self.masks = {n: torch.zeros(p.shape[0], dtype=torch.bool) for n, p in sparse}

    def update(self):
        for n, p in self.params.items():
            g = p.grad
            if g is not None and g.is_sparse:
                self.masks[n][g.coalesce().indices()[0].cpu()] = True

    def touched(self) -> dict:
        return {n: m.nonzero().squeeze(1) for n, m in self.masks.items()}

    def reset(self):
        for m in self.masks.values():
            m.zero_()
