"""Growth operators: train small, then grow. Cheaper than training 50B from step 0.

expert_mitosis(model, factor, mode)
  * "split"     : every expert is cut into `factor` children along its hidden units.
                  E*=f, top_k*=f, expert_dim/=f. SwiGLU is separable over hidden units,
                  so the function is preserved EXACTLY with the same active/total params.
                  Use it to go fine-grained.
  * "replicate" : every expert is copied `factor` times (+ noise). E*=f, top_k fixed,
                  total params *= f, active params unchanged. This is the growth step:
                  a ~7B seed with 32 experts becomes the ~52B model with 256 experts.

Depth needs no operator: it is a runtime knob (more loops), see `deepen`.
"""
from __future__ import annotations

import copy
from dataclasses import replace

import torch

from .model import NeuraFlex


@torch.no_grad()
def expert_mitosis(model: NeuraFlex, factor: int, mode: str = "replicate", noise: float = 1e-2) -> NeuraFlex:
    cfg = model.cfg
    if mode == "split":
        assert cfg.expert_dim % factor == 0
        new_cfg = replace(cfg, n_experts=cfg.n_experts * factor, top_k=cfg.top_k * factor,
                          expert_dim=cfg.expert_dim // factor,
                          shared_expert_dim=cfg.shared_expert_dim or cfg.expert_dim * cfg.n_shared_experts)
    elif mode == "replicate":
        new_cfg = replace(cfg, n_experts=cfg.n_experts * factor)
    else:
        raise ValueError(mode)

    device = next(model.parameters()).device
    child = NeuraFlex(new_cfg).to(device)
    src, dst = model.state_dict(), child.state_dict()
    E, f = cfg.n_experts, cfg.expert_dim
    for k, v in src.items():
        if k not in dst:
            continue
        if mode == "split" and k.endswith((".ffn.w1", ".ffn.w3")):        # (E, d, f)
            d = v.shape[1]
            dst[k] = v.view(E, d, factor, f // factor).permute(0, 2, 1, 3).reshape(E * factor, d, f // factor)
        elif mode == "split" and k.endswith(".ffn.w2"):                   # (E, f, d)
            d = v.shape[2]
            dst[k] = factor * v.view(E, factor, f // factor, d).reshape(E * factor, f // factor, d)
        elif k.endswith((".ffn.w1", ".ffn.w2", ".ffn.w3")):               # replicate
            rep = v.repeat_interleave(factor, dim=0)
            dst[k] = rep + noise * rep.std() * torch.randn_like(rep)
        elif k.endswith(".ffn.router.weight"):
            rep = v.repeat_interleave(factor, dim=0)
            if mode == "replicate":  # break ties between siblings
                rep = rep + noise * rep.std() * torch.randn_like(rep)
            dst[k] = rep
        elif k.endswith(".ffn.expert_bias"):
            dst[k] = v.repeat_interleave(factor, dim=0)
        elif dst[k].shape == v.shape:
            dst[k] = v
    child.load_state_dict(dst)
    return child


def deepen(model: NeuraFlex, mean_loops: float, max_loops: int) -> NeuraFlex:
    """Raise the training loop distribution; no new weights needed."""
    m = copy.deepcopy(model)
    m.cfg = replace(m.cfg, mean_loops_train=mean_loops, max_loops=max_loops)
    return m
