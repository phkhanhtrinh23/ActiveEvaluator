"""Node classification experiments.

    python -m node.run select --config configs/node.json --budgets 2500 5000 10000
    python -m node.run records --config configs/node.json
    python -m node.run experiment --config configs/node.json --budgets 10000 --pool-sizes 5 10 15 --baselines
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from active_evaluator.descriptors import ShiftDescriptor
from active_evaluator.encoders import Projection, workload_set
from active_evaluator.records import ModelRecord
from active_evaluator.runner import main, slug

from . import data
from .models import Net, predict, train


def _propagate(g, hops: int) -> List[torch.Tensor]:
    """Contextual node tokens [X, AX, A^2X] with the GCN-normalized adjacency."""
    edge_index, weight = gcn_norm(g.edge_index, num_nodes=g.num_nodes, add_self_loops=True)
    adj = torch.sparse_coo_tensor(edge_index.flip(0), weight, (g.num_nodes, g.num_nodes))
    out = [g.x]
    for _ in range(hops):
        out.append(torch.sparse.mm(adj, out[-1]))
    return out


class NodeClassification:
    def __init__(self, config: dict, out: Path, device: str | None = None):
        self.cfg, self.out = config, Path(out)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tasks = {t: data.load_task(t, spec, config["root"]) for t, spec in config["targets"].items()}
        self.targets = list(self.tasks)
        self.held = {t: data.held_out_graph(task) for t, task in self.tasks.items()}
        w = config["workloads"]
        rng = np.random.default_rng(w.get("seed", 0))
        self.workloads = [(self.targets[j % len(self.targets)], data.random_augmentation(int(rng.integers(2**31))))
                          for j in range(w["n"])]
        self.models = list(config["models"])

    def num_workloads(self) -> int:
        return len(self.workloads)

    def model_names(self) -> List[str]:
        return self.models

    def _graph(self, j: int):
        task, aug = self.workloads[j]
        return data.augment(self.held[task], aug, seed=j)

    def workload_sets(self, max_points: int) -> List[np.ndarray]:
        dim = self.cfg.get("embed_dim", 64)
        proj = {t: Projection(task.source.num_features, dim, seed=k) for k, (t, task) in enumerate(self.tasks.items())}
        sets = []
        for j in range(len(self.workloads)):
            g = self._graph(j)
            tokens = torch.cat(_propagate(g, self.cfg.get("hops", 2))).numpy()
            sets.append(workload_set([proj[self.workloads[j][0]](tokens)], max_points, seed=j))
        return sets

    def _model(self, arch: str, task: str, seed: int):
        t = self.tasks[task]
        path = self.out / "gnn" / slug(arch) / f"{slug(task)}_seed{seed}.pt"
        if path.exists():
            model = Net(arch, t.source.num_features, t.num_classes).to(self.device)
            model.load_state_dict(torch.load(path, map_location=self.device))
            return model.eval()
        model = train(arch, t.source, t.num_classes, seed, self.device)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)
        return model

    def update_record(self, record: ModelRecord, workload_ids: np.ndarray) -> ModelRecord:
        """Each architecture is trained with three seeds; SD and accuracy are averaged over seeds."""
        arch = record.name
        seeds = self.cfg.get("seeds", [0, 1, 2])
        models = {t: [self._model(arch, t, s) for s in seeds] for t in self.tasks}
        describe: Dict[str, List[ShiftDescriptor]] = {}
        for t, task in self.tasks.items():
            y = task.source.y.cpu().numpy()
            train_mask, val = task.source.train_mask.cpu().numpy(), task.source.val_mask.cpu().numpy()
            describe[t] = []
            for k, model in enumerate(models[t]):
                pred, conf, h = predict(model, task.source, self.device)
                describe[t].append(ShiftDescriptor(h[train_mask], conf[train_mask]))
                if k == 0:
                    record.calib_conf[t], record.calib_pred[t] = conf[val], pred[val]
                    record.calib_correct[t] = pred[val] == y[val]

        def run(t, g, nodes=None):
            y = g.y.cpu().numpy()
            sd, acc, preds, confs = [], [], [], []
            for model, desc in zip(models[t], describe[t]):
                pred, conf, h = predict(model, g, self.device)
                if nodes is not None:
                    pred, conf, h, y_ = pred[nodes], conf[nodes], h[nodes], y[nodes]
                else:
                    y_ = y
                sd.append(desc(h, conf))
                acc.append(np.mean(pred == y_))
                preds.append(pred)
                confs.append(conf)
            return np.mean(sd, 0), float(np.mean(acc)), preds, confs

        missing = record.missing(workload_ids)
        sd, acc, sec = [], [], []
        for j in missing:
            t0 = time.perf_counter()
            d, a, _, _ = run(self.workloads[j][0], self._graph(j))
            sec.append((time.perf_counter() - t0) / len(seeds))
            sd.append(d)
            acc.append(a)
        if len(missing):
            record.add_workloads(missing, np.stack(sd), acc, sec)

        for t, task in self.tasks.items():
            d, a, preds, confs = run(t, task.target, task.target_nodes.numpy())
            record.target_sd[t], record.target_acc[t] = d, a
            record.target_conf[t], record.target_pred[t], record.twin_pred[t] = confs[0], preds[0], preds[1 % len(preds)]
        return record

if __name__ == "__main__":
    main(NodeClassification, __doc__)
