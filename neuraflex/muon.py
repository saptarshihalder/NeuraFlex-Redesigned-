"""Muon (orthogonalised momentum) for matrices + AdamW for everything else.

Update RMS is matched to AdamW (Moonlight recipe: 0.2 * sqrt(max(m, n))) so both groups
share one learning rate. 3-D expert tensors are orthogonalised per expert (batched).
"""
from __future__ import annotations

import torch

try:
    from torch.distributed.tensor import DTensor, distribute_tensor
except Exception:  # pragma: no cover
    DTensor = None


@torch.no_grad()
def newton_schulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16() if G.is_cuda else G.float()
    tall = X.shape[-2] > X.shape[-1]
    if tall:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        X = a * X + (b * A + c * A @ A) @ X
    if tall:
        X = X.mT
    return X.to(G.dtype)


class MuonAdamW(torch.optim.Optimizer):
    def __init__(self, param_groups, lr=3e-4, momentum=0.95, betas=(0.9, 0.95), eps=1e-8,
                 weight_decay=0.1, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, betas=betas, eps=eps,
                        weight_decay=weight_decay, ns_steps=ns_steps, use_muon=False)
        super().__init__(param_groups, defaults)

    @staticmethod
    def split_params(model: torch.nn.Module):
        from .sparse import split_sparse
        muon, adam = [], []
        for name, p in split_sparse(model)[0]:
            is_matrix = p.dim() >= 2 and not any(k in name for k in ("embed", "lm_head", "adaptive"))
            (muon if is_matrix else adam).append(p)
        return [dict(params=muon, use_muon=True), dict(params=adam, use_muon=False, weight_decay=0.0)]

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for g in self.param_groups:
            lr, wd = g["lr"], g["weight_decay"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                st = self.state[p]
                if g["use_muon"]:
                    buf = st.setdefault("momentum", torch.zeros_like(grad))
                    buf.mul_(g["momentum"]).add_(grad)
                    upd = grad.add(buf, alpha=g["momentum"])  # nesterov
                    if DTensor is not None and isinstance(upd, DTensor):
                        mesh, plc = upd.device_mesh, upd.placements
                        upd = distribute_tensor(newton_schulz(upd.full_tensor(), g["ns_steps"]), mesh, plc)
                    else:
                        upd = newton_schulz(upd, g["ns_steps"])
                    upd = upd * (0.2 * max(p.shape[-2], p.shape[-1]) ** 0.5)
                    p.mul_(1 - lr * wd).add_(upd, alpha=-lr)
                else:
                    b1, b2 = g["betas"]
                    st["step"] = st.get("step", 0) + 1
                    m = st.setdefault("m", torch.zeros_like(grad))
                    v = st.setdefault("v", torch.zeros_like(grad))
                    m.lerp_(grad, 1 - b1)
                    v.lerp_(grad * grad, 1 - b2)
                    mh = m / (1 - b1 ** st["step"])
                    vh = v / (1 - b2 ** st["step"])
                    p.mul_(1 - lr * wd).addcdiv_(mh, vh.sqrt().add_(g["eps"]), value=-lr)
        return loss


class RowWiseSparseAdam(torch.optim.Optimizer):
    """Memory-lean optimizer for huge sparse tables (borrowed from recommender systems).

    torch.optim.SparseAdam keeps two dense moment tensors, which triples the RAM of a
    600M-row-element table. Here each *row* keeps one second-moment scalar and one step
    counter, so optimizer state is ~2 floats per row instead of 2*d. No first moment:
    momentum on lazily-updated rows is stale anyway.
    """

    def __init__(self, params, lr=1e-3, beta2=0.999, eps=1e-8):
        super().__init__(params, dict(lr=lr, beta2=beta2, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        for g in self.param_groups:
            lr, b2, eps = g["lr"], g["beta2"], g["eps"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.coalesce()
                idx, val = grad.indices()[0], grad.values().float()
                st = self.state[p]
                if not st:
                    st["v"] = torch.zeros(p.shape[0], dtype=torch.float32, device=p.device)
                    st["n"] = torch.zeros(p.shape[0], dtype=torch.int32, device=p.device)
                v, n = st["v"], st["n"]
                v[idx] = b2 * v[idx] + (1 - b2) * val.pow(2).mean(1)
                n[idx] += 1
                vhat = v[idx] / (1 - b2 ** n[idx].float())
                p.index_add_(0, idx, (-lr * val / (vhat.sqrt() + eps)[:, None]).to(p.dtype))


def _coalesce_grad(p):
    if p.grad is not None and p.grad.is_sparse:
        p.grad = p.grad.coalesce()


class OptimizerGroup:
    """Dense params -> Muon/AdamW, sparse lookup tables -> SparseAdam (row-wise updates)."""

    def __init__(self, model: torch.nn.Module, lr: float, weight_decay: float = 0.1,
                 kind: str = "muon", sparse_lr: float | None = None):
        from .sparse import split_sparse
        dense, sparse = split_sparse(model)
        self.dense_params = [p for _, p in dense]
        if kind == "muon":
            self.dense = MuonAdamW(MuonAdamW.split_params(model), lr=lr, weight_decay=weight_decay)
        else:
            self.dense = torch.optim.AdamW(self.dense_params, lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)
        self.sparse_ratio = (sparse_lr or lr) / lr
        for _, p in sparse:  # dedupe sparse grad rows as they accumulate (bounded memory)
            p.register_post_accumulate_grad_hook(_coalesce_grad)
        self.sparse = RowWiseSparseAdam([p for _, p in sparse], lr=sparse_lr or lr) if sparse else None
        self.optimizers = [o for o in (self.dense, self.sparse) if o is not None]

    def set_lr(self, lr: float):
        for g in self.dense.param_groups:
            g["lr"] = lr
        if self.sparse is not None:
            for g in self.sparse.param_groups:
                g["lr"] = lr * self.sparse_ratio

    def clip(self, max_norm: float = 1.0):
        return torch.nn.utils.clip_grad_norm_([p for p in self.dense_params if p.grad is not None], max_norm)

    def step(self):
        for o in self.optimizers:
            o.step()

    def zero_grad(self):
        for o in self.optimizers:
            o.zero_grad(set_to_none=True)

    def state_dict(self):
        return [o.state_dict() for o in self.optimizers]

    def load_state_dict(self, sds):
        for o, sd in zip(self.optimizers, sds):
            o.load_state_dict(sd)
