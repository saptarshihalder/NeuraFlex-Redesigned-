"""NeuraFlex-Ouroboros configuration and presets."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class NeuraFlexConfig:
    # ---- vocabulary / width -------------------------------------------------
    vocab_size: int = 151_936
    d_model: int = 5120
    norm_eps: float = 1e-6
    init_std: float = 0.006
    tie_embeddings: bool = False
    max_seq_len: int = 32_768
    rope_theta: float = 1_000_000.0

    # ---- Multi-head Latent Attention (MLA) ----------------------------------
    n_heads: int = 40
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128

    # ---- Gated DeltaNet (linear attention) ----------------------------------
    gdn_heads: int = 40
    gdn_head_dim: int = 128
    gdn_conv_size: int = 4

    # ---- layout: Prelude -> Ouroboros core (looped) -> Coda -----------------
    n_prelude: int = 2
    n_core: int = 8
    n_coda: int = 2
    # Pattern tiled over the core layers. L = Gated DeltaNet, A = MLA.
    core_attn_pattern: str = "LLLA"
    dense_ffn_dim: int = 13_824  # prelude / coda / MTP FFN width

    # ---- Depth-Routed Experts (core FFN) ------------------------------------
    n_experts: int = 256
    n_shared_experts: int = 1
    top_k: int = 8
    expert_dim: int = 1536
    shared_expert_dim: int = 0      # 0 -> expert_dim * n_shared_experts
    loop_emb_dim: int = 64          # sinusoidal loop-index features fed to the router
    router_bias_update_rate: float = 1e-3  # aux-loss-free balancing step
    seq_aux_loss_weight: float = 1e-4

    # ---- looping ------------------------------------------------------------
    max_loops: int = 16             # hard cap at train time
    mean_loops_train: float = 4.0   # mean of the log-normal Poisson loop sampler
    loop_sigma: float = 0.5
    bptt_loops: int = 4             # backprop only through the last k iterations
    loop_lora_rank: int = 64        # per-iteration low-rank "relaxation" adapters
    n_loop_adapters: int = 16       # iterations beyond this reuse the last adapter

    # ---- inference-time adaptive depth --------------------------------------
    infer_loops: int = 8
    exit_threshold: float = 0.0     # 0 disables; e.g. 0.02 = stop when relative state change < 2%

    # ---- multi-token prediction ---------------------------------------------
    n_mtp: int = 1
    mtp_weight: float = 0.3

    extra: dict = field(default_factory=dict)

    # -------------------------------------------------------------------------
    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    def core_kind(self, j: int) -> str:
        return self.core_attn_pattern[j % len(self.core_attn_pattern)]

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "NeuraFlexConfig":
        with open(path) as f:
            return cls(**json.load(f))


def preset(name: str) -> NeuraFlexConfig:
    name = name.lower()
    if name in ("50b", "ouroboros-50b"):
        return NeuraFlexConfig()
    if name in ("3b", "ouroboros-3b"):
        # Proxy model for architecture ablations before committing to 50B.
        return NeuraFlexConfig(
            d_model=2048, n_heads=16, q_lora_rank=768, kv_lora_rank=256,
            qk_nope_head_dim=96, qk_rope_head_dim=32, v_head_dim=96,
            gdn_heads=16, gdn_head_dim=128, dense_ffn_dim=5632,
            n_experts=64, top_k=6, expert_dim=768,
        )
    if name == "debug":
        return NeuraFlexConfig(
            vocab_size=256, d_model=64, n_heads=4, q_lora_rank=32, kv_lora_rank=16,
            qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16,
            gdn_heads=4, gdn_head_dim=16, n_prelude=1, n_core=4, n_coda=1,
            dense_ffn_dim=128, n_experts=8, top_k=2, expert_dim=32, loop_emb_dim=16,
            max_loops=6, mean_loops_train=3.0, bptt_loops=2, loop_lora_rank=4,
            n_loop_adapters=6, infer_loops=3, max_seq_len=512, init_std=0.02,
        )
    raise ValueError(f"unknown preset {name!r}")
