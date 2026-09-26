import numpy as np
import pytest
import torch

pytest.importorskip("torch_geometric")

from torch_geometric.data import Data  # noqa: E402

from active_evaluator.records import ModelRecord  # noqa: E402
from node import data  # noqa: E402
from node.models import ARCHITECTURES, predict, train  # noqa: E402
from node.run import NodeClassification, _propagate  # noqa: E402


def _graph(n=120, d=8, c=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, c, (n,), generator=g)
    x = torch.randn(n, d, generator=g) + torch.nn.functional.one_hot(y, d).float() * 2
    src = torch.randint(0, n, (4 * n,), generator=g)
    dst = torch.where(torch.rand(4 * n, generator=g) < 0.8, src.roll(1), torch.randint(0, n, (4 * n,), generator=g))
    return Data(x=x, edge_index=torch.stack([torch.cat([src, dst]), torch.cat([dst, src])]), y=y)


def _task(seed=0):
    return data.GraphTask(data._split(_graph(seed=seed), 0), _graph(seed=seed + 1), torch.arange(120), 3)


@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_architectures_train_and_embed(arch):
    task = _task()
    model = train(arch, task.source, 3, seed=0, device="cpu", epochs=5)
    pred, conf, h = predict(model, task.target, "cpu")
    assert pred.shape == conf.shape == (120,) and h.shape == (120, 96)


def test_augmentation_and_tokens():
    held = data.held_out_graph(_task())
    aug = data.random_augmentation(3)
    a, b = data.augment(held, aug, seed=7), data.augment(held, aug, seed=7)
    assert torch.equal(a.x, b.x) and torch.equal(a.edge_index, b.edge_index)
    assert a.num_nodes < held.num_nodes
    tokens = _propagate(a, 2)
    assert len(tokens) == 3 and all(t.shape == a.x.shape for t in tokens)


def test_records(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "load_task", lambda t, spec, root, seed=0: _task(seed=len(t)))
    cfg = {"root": "", "workloads": {"n": 6}, "targets": {"A": {}, "BB": {}}, "models": ["GCN", "MLP"]}
    m = NodeClassification(cfg, tmp_path, device="cpu")
    sets = m.workload_sets(max_points=64)
    assert len(sets) == 6 and all(s.shape[1] == 64 and len(s) <= 64 for s in sets)
    assert np.allclose(np.linalg.norm(sets[0].astype(np.float32), axis=1), 1, atol=1e-2)
    rec = m.update_record(ModelRecord(name="GCN"), np.arange(6))
    assert rec.sd.shape == (6, 8) and set(rec.target_acc) == {"A", "BB"} and "A" in rec.twin_pred
