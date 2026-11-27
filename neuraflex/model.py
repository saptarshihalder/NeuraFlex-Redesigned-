"""NeuraFlex-Ouroboros: a looped (recurrent-depth) hybrid-attention MoE language model.

    tokens -> Prelude (unique layers) -> e
    s_0 = e
    repeat r times (r sampled in training, adaptive at inference):
        s = Adapter([norm(s); norm(e)])            # echo injection: re-anchor on the input
        for each core layer j:
            s = Block_j(s, loop=i)                 # Depth-Routed Experts + per-loop LoRA
    -> Coda (unique layers) -> LM head  (+ multi-token-prediction heads)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NeuraFlexConfig
from .layers import MLA, GatedDeltaNet, RMSNorm, Rotary, SwiGLU
from .moe import DepthRoutedMoE, loop_features


class LoopLoRA(nn.Module):
    """Per-iteration low-rank adapters ("relaxed recursion"): lets each pass specialise
    while sharing >99% of its weights with every other pass."""

    def __init__(self, cfg: NeuraFlexConfig):
        super().__init__()
        n, d, r = cfg.n_loop_adapters, cfg.d_model, cfg.loop_lora_rank
        self.A = nn.Parameter(torch.empty(n, d, r))
        self.B = nn.Parameter(torch.zeros(n, r, d))

    def forward(self, x, i: int):
        i = min(i, self.A.shape[0] - 1)
        return (x @ self.A[i]) @ self.B[i]


class Block(nn.Module):
    def __init__(self, cfg: NeuraFlexConfig, kind: str, moe: bool, sandwich: bool):
        super().__init__()
        self.kind, self.is_moe, self.sandwich = kind, moe, sandwich
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = MLA(cfg) if kind == "A" else GatedDeltaNet(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = DepthRoutedMoE(cfg) if moe else SwiGLU(cfg.d_model, cfg.dense_ffn_dim)
        if sandwich:
            self.attn_post = RMSNorm(cfg.d_model, cfg.norm_eps)
            self.ffn_post = RMSNorm(cfg.d_model, cfg.norm_eps)

    def forward(self, x, rope, cache=None, loop_feat=None, lora: LoopLoRA | None = None, i: int = 0):
        hn = self.attn_norm(x)
        a = self.attn(hn, rope, cache)
        if lora is not None:
            a = a + lora(hn, i)
        if self.sandwich:
            a = self.attn_post(a)
        x = x + a
        hn = self.ffn_norm(x)
        if self.is_moe:
            f, aux = self.ffn(hn, loop_feat)
        else:
            f, aux = self.ffn(hn), None
        if self.sandwich:
            f = self.ffn_post(f)
        return x + f, aux

    def fill_cache(self, x, rope, cache):
        """Cheap cache fill for loop iterations a token skipped (attention only, no FFN)."""
        self.attn(self.attn_norm(x), rope, cache)


class MTPModule(nn.Module):
    def __init__(self, cfg: NeuraFlexConfig):
        super().__init__()
        self.norm_h = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.norm_e = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.proj = nn.Linear(2 * cfg.d_model, cfg.d_model, bias=False)
        self.block = Block(cfg, "A", moe=False, sandwich=False)

    def forward(self, h, emb, rope):
        x = self.proj(torch.cat([self.norm_h(h), self.norm_e(emb)], dim=-1))
        return self.block(x, rope)[0]


@dataclass
class Output:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    ce: torch.Tensor | None = None
    mtp: torch.Tensor | None = None
    aux: torch.Tensor | None = None
    loops: int = 0
    hidden: torch.Tensor | None = None


class NeuraFlex(nn.Module):
    def __init__(self, cfg: NeuraFlexConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.embed = nn.Embedding(cfg.vocab_size, d)
        self.rotary = Rotary(cfg.qk_rope_head_dim, cfg.rope_theta)
        self.prelude = nn.ModuleList(Block(cfg, "A", False, False) for _ in range(cfg.n_prelude))
        self.core = nn.ModuleList(Block(cfg, cfg.core_kind(j), True, True) for j in range(cfg.n_core))
        self.loop_lora = nn.ModuleList(LoopLoRA(cfg) for _ in range(cfg.n_core))
        self.state_norm = RMSNorm(d, cfg.norm_eps)
        self.echo_norm = RMSNorm(d, cfg.norm_eps)
        self.adapter = nn.Linear(2 * d, d, bias=False)
        self.coda = nn.ModuleList(Block(cfg, "A", False, False) for _ in range(cfg.n_coda))
        self.norm = RMSNorm(d, cfg.norm_eps)
        self.lm_head = nn.Linear(d, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self.mtp = nn.ModuleList(MTPModule(cfg) for _ in range(cfg.n_mtp))
        self.reset_parameters()

    # ------------------------------------------------------------------ init
    @torch.no_grad()
    def reset_parameters(self):
        std = self.cfg.init_std
        for name, p in self.named_parameters():
            if p.dim() >= 2:  # norms (ones), A_log, dt_bias keep their own init
                nn.init.normal_(p, 0.0, std)
        for m in self.loop_lora:
            nn.init.zeros_(m.B)
        # buffers (needed after meta-device init + to_empty)
        cfg = self.cfg
        dim = cfg.qk_rope_head_dim
        inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.rotary.inv_freq.copy_(inv.to(self.rotary.inv_freq.device))
        for blk in self.core:
            blk.ffn.expert_bias.zero_()
            blk.ffn.load.zero_()
        for m in self.modules():
            if isinstance(m, GatedDeltaNet):
                with torch.no_grad():
                    m.A_log.copy_(torch.empty_like(m.A_log).uniform_(1, 16).log())
                    m.dt_bias.zero_()

    # ------------------------------------------------------------- utilities
    def sample_loops(self, generator: torch.Generator | None = None) -> int:
        """Log-normal Poisson depth sampler (heavy tail teaches the core to keep improving)."""
        cfg = self.cfg
        mu = math.log(cfg.mean_loops_train) - 0.5 * cfg.loop_sigma ** 2
        tau = torch.normal(torch.tensor(mu), torch.tensor(cfg.loop_sigma), generator=generator)
        r = int(torch.poisson(tau.exp().clamp_max(4 * cfg.max_loops), generator=generator).item()) + 1
        return max(1, min(r, cfg.max_loops))

    @torch.no_grad()
    def update_router_biases(self):
        for blk in self.core:
            blk.ffn.update_bias(self.cfg.router_bias_update_rate)

    def param_counts(self) -> dict:
        cfg = self.cfg
        total = sum(p.numel() for p in self.parameters())
        emb = self.embed.weight.numel() + (0 if cfg.tie_embeddings else self.lm_head.weight.numel())
        routed = sum(b.ffn.w1.numel() + b.ffn.w2.numel() + b.ffn.w3.numel() for b in self.core)
        mtp = sum(p.numel() for p in self.mtp.parameters())
        active = total - emb - mtp - routed + routed * cfg.top_k / cfg.n_experts
        return {"total": total, "embedding": emb, "routed_experts": routed, "mtp": mtp,
                "active_non_embedding_per_pass": int(active)}

    # --------------------------------------------------------------- forward
    def forward(self, input_ids, labels=None, n_loops: int | None = None, cache: dict | None = None,
                start_pos: int = 0, exit_threshold: float | None = None) -> Output:
        cfg = self.cfg
        B, T = input_ids.shape
        dev = input_ids.device
        pos = torch.arange(start_pos, start_pos + T, device=dev)
        rope = self.rotary(pos)
        slot = (lambda k: None) if cache is None else (lambda k: cache.setdefault(k, {}))

        x = self.embed(input_ids)
        for j, blk in enumerate(self.prelude):
            x, _ = blk(x, rope, slot(("pre", j)))

        if n_loops is None:
            n_loops = self.sample_loops() if self.training else cfg.infer_loops
        if exit_threshold is None:
            exit_threshold = 0.0 if self.training else cfg.exit_threshold
        n_frozen = max(0, n_loops - cfg.bptt_loops) if self.training else 0

        e = self.echo_norm(x)
        s = x
        aux = x.new_zeros((), dtype=torch.float32)
        used = 0
        for i in range(n_loops):
            with torch.set_grad_enabled(torch.is_grad_enabled() and i >= n_frozen):
                s_prev = s
                s = self.adapter(torch.cat([self.state_norm(s), e], dim=-1))
                lf = loop_features(i, cfg.loop_emb_dim, dev)
                for j, blk in enumerate(self.core):
                    s, a = blk(s, rope, slot(("core", j, i)), lf, self.loop_lora[j], i)
                    aux = aux + a
            used = i + 1
            if exit_threshold > 0 and i >= 1:
                delta = (s - s_prev).float().norm(dim=-1) / s.float().norm(dim=-1).clamp_min(1e-6)
                if delta.max().item() < exit_threshold:
                    break
        if cache is not None:
            for i in range(used, n_loops):  # keep every iteration's cache aligned
                for j, blk in enumerate(self.core):
                    blk.fill_cache(s, rope, slot(("core", j, i)))

        x = s
        for j, blk in enumerate(self.coda):
            x, _ = blk(x, rope, slot(("coda", j)))
        logits = self.lm_head(self.norm(x))
        out = Output(logits=logits, loops=used, hidden=x)

        if labels is not None:
            V = cfg.vocab_size
            ce = F.cross_entropy(logits.float().view(-1, V), labels.reshape(-1), ignore_index=-100)
            mtp_loss = x.new_zeros((), dtype=torch.float32)
            h = x
            n = 0
            for k, mod in enumerate(self.mtp, start=1):
                if T - k <= 0:
                    break
                h = mod(h[:, :-1], self.embed(input_ids[:, k:]), self.rotary(pos[: T - k]))
                lg = self.lm_head(self.norm(h))
                mtp_loss = mtp_loss + F.cross_entropy(lg.float().view(-1, V), labels[:, k:].reshape(-1), ignore_index=-100)
                n += 1
            if n:
                mtp_loss = mtp_loss / n
            aux = aux / max(1, used * cfg.n_core)
            out.loss = ce + cfg.mtp_weight * mtp_loss + cfg.seq_aux_loss_weight * aux
            out.ce, out.mtp, out.aux = ce, mtp_loss, aux
        return out

    # -------------------------------------------------------------- generate
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=128, temperature=0.8, top_p=0.95,
                 n_loops: int | None = None, exit_threshold: float | None = None, eos_id: int | None = None):
        self.eval()
        n_loops = n_loops or self.cfg.infer_loops
        cache: dict = {}
        out = self(input_ids, cache=cache, n_loops=n_loops, exit_threshold=exit_threshold)
        pos = input_ids.shape[1]
        loops_log = [out.loops]
        for _ in range(max_new_tokens):
            nxt = _sample(out.logits[:, -1], temperature, top_p)
            input_ids = torch.cat([input_ids, nxt], dim=1)
            if eos_id is not None and (nxt == eos_id).all():
                break
            out = self(nxt, cache=cache, start_pos=pos, n_loops=n_loops, exit_threshold=exit_threshold)
            loops_log.append(out.loops)
            pos += 1
        return input_ids, loops_log


def _sample(logits, temperature, top_p):
    if temperature <= 0:
        return logits.argmax(-1, keepdim=True)
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    sp, si = probs.sort(-1, descending=True)
    keep = sp.cumsum(-1) - sp < top_p
    sp = sp * keep
    choice = torch.multinomial(sp / sp.sum(-1, keepdim=True), 1)
    return si.gather(-1, choice)
