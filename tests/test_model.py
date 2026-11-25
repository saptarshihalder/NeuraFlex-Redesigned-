import torch
import pytest

from neuraflex import NeuraFlex, preset
from neuraflex.layers import delta_rule_reference
from neuraflex.muon import MuonAdamW, newton_schulz
from neuraflex.upcycle import expert_mitosis


@pytest.fixture
def model():
    torch.manual_seed(0)
    return NeuraFlex(preset("debug"))


def test_train_step_reduces_loss(model):
    torch.manual_seed(1)
    ids = torch.randint(0, 256, (4, 32))
    opt = MuonAdamW(MuonAdamW.split_params(model), lr=3e-3)
    model.train()
    first = None
    for _ in range(30):
        out = model(ids[:, :-1], labels=ids[:, 1:], n_loops=3)
        out.loss.backward()
        opt.step(); opt.zero_grad()
        model.update_router_biases()
        first = first if first is not None else out.loss.item()
    assert out.loss.item() < first * 0.7


def test_kv_cache_matches_full_forward(model):
    model.eval()
    ids = torch.randint(0, 256, (2, 20))
    full = model(ids, n_loops=3).logits
    cache = {}
    parts = [model(ids[:, :7], cache=cache, n_loops=3).logits]
    for t in range(7, 20):
        parts.append(model(ids[:, t:t + 1], cache=cache, start_pos=t, n_loops=3).logits)
    assert torch.allclose(torch.cat(parts, 1), full, atol=1e-4)


def test_truncated_bptt_only_grads_last_loops(model):
    model.train()
    ids = torch.randint(0, 256, (1, 16))
    out = model(ids, labels=ids, n_loops=6)  # bptt_loops=2 in debug preset
    out.loss.backward()
    assert model.loop_lora[0].B.grad[:4].abs().sum() == 0
    assert model.loop_lora[0].B.grad[4:6].abs().sum() > 0


def test_early_exit_and_generate(model):
    out, loops = model.generate(torch.randint(0, 256, (1, 5)), max_new_tokens=6, exit_threshold=10.0)
    assert out.shape == (1, 11)
    assert all(l == 2 for l in loops)  # exits at the first allowed check


def test_split_mitosis_is_exact(model):
    model.eval()
    ids = torch.randint(0, 256, (2, 12))
    ref = model(ids, n_loops=2).logits
    child = expert_mitosis(model, 2, mode="split").eval()
    assert child.cfg.n_experts == 16 and child.cfg.expert_dim == 16
    assert torch.allclose(child(ids, n_loops=2).logits, ref, atol=1e-4)


def test_replicate_mitosis_grows_params(model):
    child = expert_mitosis(model, 4, mode="replicate")
    assert child.param_counts()["routed_experts"] == 4 * model.param_counts()["routed_experts"]


def test_delta_rule_state_carry():
    B, T, H, D = 1, 10, 2, 8
    q, k = (torch.nn.functional.normalize(torch.randn(B, T, H, D), dim=-1) for _ in range(2))
    v, g, b = torch.randn(B, T, H, D), -torch.rand(B, T, H), torch.rand(B, T, H)
    o_full, _ = delta_rule_reference(q, k, v, g, b)
    o1, S = delta_rule_reference(q[:, :4], k[:, :4], v[:, :4], g[:, :4], b[:, :4])
    o2, _ = delta_rule_reference(q[:, 4:], k[:, 4:], v[:, 4:], g[:, 4:], b[:, 4:], S)
    assert torch.allclose(torch.cat([o1, o2], 1), o_full, atol=1e-5)


def test_newton_schulz_orthogonalises():
    X = newton_schulz(torch.randn(3, 32, 64), steps=5)
    s = torch.linalg.svdvals(X)
    assert s.min() > 0.5 and s.max() < 1.3
