import os

import torch

from neuraflex import NeuraFlex, preset
from neuraflex.hive.coordinator import Coordinator
from neuraflex.hive.transport import LocalTransport
from neuraflex.hive.worker import Worker
from neuraflex.hive.protocol import merge


def test_sparse_rows_average_only_over_touchers():
    master = {"t": torch.zeros(6, 2), "w": torch.zeros(3)}
    d1 = {"base_round": 0, "tokens": 10, "loss": 1.0, "dense": {"w": torch.ones(3)},
          "sparse": {"t": (torch.tensor([0, 1]), torch.tensor([[1.0, 1.0], [2.0, 2.0]]))}, "buffers": {}}
    d2 = {"base_round": 0, "tokens": 10, "loss": 1.0, "dense": {"w": 3 * torch.ones(3)},
          "sparse": {"t": (torch.tensor([1, 4]), torch.tensor([[4.0, 4.0], [5.0, 5.0]]))}, "buffers": {}}
    stats = merge(master, [d1, d2], current_round=0, outer={}, lr=1.0, momentum=0.0)
    assert stats["used"] == 2
    assert torch.allclose(master["t"][0], torch.tensor([-1.0, -1.0]))   # only worker 1 touched row 0
    assert torch.allclose(master["t"][1], torch.tensor([-3.0, -3.0]))   # mean of 2 and 4
    assert torch.allclose(master["t"][4], torch.tensor([-5.0, -5.0]))   # only worker 2: full strength
    assert master["t"][2].abs().sum() == 0 and master["t"][5].abs().sum() == 0
    assert torch.allclose(master["w"], torch.full((3,), -2.0))           # dense: plain mean


def test_staleness_downweights_and_drops():
    master = {"w": torch.zeros(1)}
    fresh = {"base_round": 5, "tokens": 10, "loss": 0, "dense": {"w": torch.ones(1)}, "sparse": {}, "buffers": {}}
    stale = {"base_round": 4, "tokens": 10, "loss": 0, "dense": {"w": 4 * torch.ones(1)}, "sparse": {}, "buffers": {}}
    ancient = {"base_round": 0, "tokens": 10, "loss": 0, "dense": {"w": 100 * torch.ones(1)}, "sparse": {}, "buffers": {}}
    merge(master, [fresh, stale, ancient], current_round=5, outer={}, lr=1.0, momentum=0.0, max_staleness=2)
    assert torch.allclose(master["w"], -torch.tensor([(10 * 1 + 5 * 4) / 15]))


def test_swarm_end_to_end(tmp_path):
    torch.manual_seed(0)
    text = (b"the looped hive learns from many small machines. " * 400)
    data = tmp_path / "data.txt"
    data.write_bytes(text)
    t = LocalTransport(str(tmp_path / "hub"))
    coord = Coordinator(t, str(tmp_path / "coord"))
    cfg = preset("debug-pkm")
    coord.init(cfg)

    workers = [Worker(t, f"w{i}", str(data), seq_len=32, micro_batch=2, accum=1, lr=3e-3, warmup=1,
                      state_dir=str(tmp_path / "workers"), bf16=False) for i in range(2)]

    def global_loss():
        m = NeuraFlex(cfg).eval()
        snap = torch.load(os.path.join(t.root, f"global/round_{coord.round:06d}.pt"), weights_only=True)
        m.load_state_dict(snap["model"])
        ids = torch.tensor([list(text[:257])])
        with torch.no_grad():
            return m(ids[:, :-1], labels=ids[:, 1:], n_loops=2).ce.item()

    start = global_loss()
    for _ in range(6):
        for w in workers:
            w.sync()
            w.push(w.run_window(max_steps=5))
        stats = coord.step()
        assert stats["used"] == 2 and stats["sparse_rows"] > 0
    assert coord.round == 6
    assert not t.list("deltas/")                       # consumed
    snaps = t.list("global/round_")
    assert [p[-9:-3] for p in snaps] == ["000004", "000005", "000006"]  # only the last 3 kept
    assert len(t.list("deltas/")) == 0 and t.list("global/LATEST") == ["global/LATEST"]
    end = global_loss()
    assert end < start * 0.75, (start, end)

    # a worker that "dies" and restarts resumes from the newest global on a new shard
    reborn = Worker(t, "w0", str(data), seq_len=32, micro_batch=2, accum=1,
                    state_dir=str(tmp_path / "workers"), bf16=False)
    assert reborn.restarts == 1 and reborn.sync() == 6
