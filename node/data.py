"""Graphs, source/target splits and augmented meta-dataset workloads.

Domain graphs follow the GNNEvaluator layout:
    <root>/<name>/raw/<name>_docs.txt, <name>_edgelist.txt, <name>_labels.txt
ogbn-arxiv and the GOOD datasets are downloaded by their own packages.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import dropout_edge, subgraph, to_undirected


@dataclass
class GraphTask:
    source: Data  # graph the models are trained on, with train/val/held masks
    target: Data  # graph containing the target nodes
    target_nodes: torch.Tensor
    num_classes: int


def _domain(root: Path, name: str) -> Data:
    raw = root / name / "raw"
    edges = np.loadtxt(raw / f"{name}_edgelist.txt", delimiter=",", dtype=np.int64).reshape(-1, 2)
    x = np.loadtxt(raw / f"{name}_docs.txt", delimiter=",", dtype=np.float32)
    y = np.loadtxt(raw / f"{name}_labels.txt", dtype=np.int64)
    edge_index = to_undirected(torch.as_tensor(edges.T), num_nodes=len(y))
    return Data(x=torch.as_tensor(x), edge_index=edge_index, y=torch.as_tensor(y))


def _induced(data: Data, nodes: torch.Tensor) -> Data:
    edge_index, _ = subgraph(nodes, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)
    return Data(x=data.x[nodes], edge_index=edge_index, y=data.y[nodes])


def _split(data: Data, seed: int) -> Data:
    """70% train, 10% validation, 20% held out (the meta-dataset is built from val + held-out nodes)."""
    perm = torch.as_tensor(np.random.default_rng(seed).permutation(data.num_nodes))
    n_train, n_val = int(0.7 * len(perm)), int(0.1 * len(perm))
    for name, idx in (("train_mask", perm[:n_train]), ("val_mask", perm[n_train : n_train + n_val]),
                      ("held_mask", perm[n_train + n_val :])):
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[idx] = True
        data[name] = mask
    return data


def _good(root: Path, name: str) -> tuple[Data, torch.Tensor, torch.Tensor]:
    import importlib

    cls, domain = {"GOOD-Cora": ("GOODCora", "word"), "GOOD-Twitch": ("GOODTwitch", "language"),
                   "GOOD-WebKB": ("GOODWebKB", "university")}[name]
    module = importlib.import_module(f"GOOD.data.good_datasets.good_{cls[4:].lower()}")
    dataset, _ = getattr(module, cls).load(str(root), domain=domain, shift="covariate")
    g = dataset[0]
    y = g.y.long().view(-1) if g.y.dim() == 1 or g.y.shape[1] == 1 else g.y.argmax(1)
    data = Data(x=g.x.float(), edge_index=g.edge_index, y=y)
    return data, torch.nonzero(g.train_mask | g.id_val_mask | g.id_test_mask).view(-1), torch.nonzero(g.test_mask).view(-1)


def load_task(target: str, spec: dict, root: str | Path, seed: int = 0) -> GraphTask:
    root = Path(root)
    if spec["kind"] == "domain":
        source, graph = _domain(root, spec["source"]), _domain(root, spec["target"])
        nodes = torch.arange(graph.num_nodes)
    elif spec["kind"] == "ogb":
        from ogb.nodeproppred import PygNodePropPredDataset

        ds = PygNodePropPredDataset(name=spec["name"], root=str(root))
        g, split = ds[0], ds.get_idx_split()
        graph = Data(x=g.x, edge_index=to_undirected(g.edge_index, num_nodes=g.num_nodes), y=g.y.view(-1))
        source, nodes = _induced(graph, split["train"]), split["test"]
    elif spec["kind"] == "good":
        graph, src_nodes, nodes = _good(root, target)
        source = _induced(graph, src_nodes)
    else:
        raise KeyError(spec["kind"])
    num_classes = int(max(source.y.max(), graph.y.max())) + 1
    return GraphTask(_split(source, seed), graph, nodes, num_classes)


def held_out_graph(task: GraphTask) -> Data:
    s = task.source
    return _induced(s, torch.nonzero(s.val_mask | s.held_mask).view(-1))


def random_augmentation(seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    return {
        "sample": float(rng.uniform(0.1, 0.6)),
        "edge_drop": float(rng.uniform(0, 0.6)) if rng.random() < 0.5 else 0.0,
        "feature_mask": float(rng.uniform(0, 0.6)) if rng.random() < 0.5 else 0.0,
        "feature_noise": float(rng.uniform(0, 0.5)) if rng.random() < 0.3 else 0.0,
    }


def augment(graph: Data, aug: Dict[str, float], seed: int) -> Data:
    seed = int(seed)
    g = torch.Generator().manual_seed(seed)
    n = graph.num_nodes
    nodes = torch.randperm(n, generator=g)[: max(2, int(aug["sample"] * n))].sort().values
    sub = _induced(graph, nodes)
    if aug["edge_drop"] > 0:
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            sub.edge_index, _ = dropout_edge(sub.edge_index, p=aug["edge_drop"], force_undirected=True)
    x = sub.x.clone()
    if aug["feature_mask"] > 0:
        x[:, torch.rand(x.shape[1], generator=g) < aug["feature_mask"]] = 0
    if aug["feature_noise"] > 0:
        x = x + aug["feature_noise"] * x.std(0, keepdim=True) * torch.randn(x.shape, generator=g)
    sub.x = x
    return sub
