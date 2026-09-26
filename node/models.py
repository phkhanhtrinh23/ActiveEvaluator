"""Node-classification architectures (PyG) trained on each source graph."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric import nn as gnn
from torch_geometric.data import Data

HIDDEN = 96

CONVS = {
    "GCN": lambda i, o: gnn.GCNConv(i, o),
    "GraphSAGE": lambda i, o: gnn.SAGEConv(i, o),
    "GAT": lambda i, o: gnn.GATConv(i, o // 4, heads=4),
    "GATv2": lambda i, o: gnn.GATv2Conv(i, o // 4, heads=4),
    "GIN": lambda i, o: gnn.GINConv(nn.Sequential(nn.Linear(i, o), nn.ReLU(), nn.Linear(o, o))),
    "ChebNet": lambda i, o: gnn.ChebConv(i, o, K=3),
    "ARMA": lambda i, o: gnn.ARMAConv(i, o, num_stacks=2, num_layers=1),
    "TAGCN": lambda i, o: gnn.TAGConv(i, o, K=3),
    "SuperGAT": lambda i, o: gnn.SuperGATConv(i, o // 4, heads=4),
    "UniMP": lambda i, o: gnn.TransformerConv(i, o // 4, heads=4),
    "SGC": lambda i, o: gnn.SGConv(i, o, K=2),
    "MixHop": lambda i, o: gnn.MixHopConv(i, o // 3, powers=[0, 1, 2]),
}
ARCHITECTURES = sorted([*CONVS, "MLP", "APPNP", "GCNII", "JKNet"])


class Net(nn.Module):
    """Two propagation layers, a penultimate node embedding and a linear head."""

    def __init__(self, arch: str, in_dim: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.arch, self.dropout = arch, dropout
        if arch in CONVS:
            self.layers = nn.ModuleList([CONVS[arch](in_dim, HIDDEN), CONVS[arch](HIDDEN, HIDDEN)])
        elif arch == "MLP":
            self.layers = nn.ModuleList([nn.Linear(in_dim, HIDDEN), nn.Linear(HIDDEN, HIDDEN)])
        elif arch == "APPNP":
            self.layers = nn.ModuleList([nn.Linear(in_dim, HIDDEN), nn.Linear(HIDDEN, HIDDEN)])
            self.prop = gnn.APPNP(K=10, alpha=0.1)
        elif arch == "GCNII":
            self.inp = nn.Linear(in_dim, HIDDEN)
            self.layers = nn.ModuleList([gnn.GCN2Conv(HIDDEN, alpha=0.1, theta=0.5, layer=k + 1) for k in range(4)])
        elif arch == "JKNet":
            self.layers = nn.ModuleList([gnn.GCNConv(in_dim, HIDDEN), gnn.GCNConv(HIDDEN, HIDDEN)])
            self.jk = nn.Linear(2 * HIDDEN, HIDDEN)
        else:
            raise KeyError(arch)
        self.head = nn.Linear(HIDDEN, num_classes)

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        drop = lambda h: F.dropout(h, self.dropout, self.training)
        if self.arch in ("MLP", "APPNP"):
            h = self.layers[1](drop(F.relu(self.layers[0](drop(x)))))
            return F.relu(self.prop(h, edge_index) if self.arch == "APPNP" else h)
        if self.arch == "GCNII":
            h = h0 = F.relu(self.inp(drop(x)))
            for conv in self.layers:
                h = F.relu(conv(drop(h), h0, edge_index))
            return h
        hs, h = [], x
        for conv in self.layers:
            h = F.relu(conv(drop(h), edge_index))
            hs.append(h)
        return F.relu(self.jk(torch.cat(hs, -1))) if self.arch == "JKNet" else h

    def forward(self, x, edge_index):
        h = self.embed(x, edge_index)
        return self.head(F.dropout(h, self.dropout, self.training)), h


def train(arch: str, data: Data, num_classes: int, seed: int, device, epochs: int = 300, patience: int = 50) -> Net:
    torch.manual_seed(seed)
    model = Net(arch, data.num_features, num_classes).to(device)
    data = data.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    best, best_state, wait = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        logits, _ = model(data.x, data.edge_index)
        F.cross_entropy(logits[data.train_mask], data.y[data.train_mask]).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(data.x, data.edge_index)[0].argmax(-1)
        acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
        if acc > best:
            best, best_state, wait = acc, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model.eval()


@torch.no_grad()
def predict(model: Net, data: Data, device):
    """Predictions, confidences and embeddings for every node."""
    data = data.to(device)
    logits, h = model(data.x, data.edge_index)
    conf, pred = logits.softmax(-1).max(-1)
    return pred.cpu().numpy(), conf.cpu().numpy(), h.cpu().numpy()
