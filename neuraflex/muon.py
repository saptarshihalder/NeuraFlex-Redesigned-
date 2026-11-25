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
        muon, adam = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            is_matrix = p.dim() >= 2 and "embed" not in name and "lm_head" not in name
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
