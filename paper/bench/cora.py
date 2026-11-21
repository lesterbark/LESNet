"""Semi-supervised node classification on Cora and CiteSeer (Planetoid split: 20 labels per class).  The graph is given, there is no geometry: each
node's partner list is itself and its neighbours (capped at 64), the relation is two numbers (is-self, 1/sqrt(degree)).  LESNet is compared
with GCN and GAT (dense implementations).  Test accuracy at the epoch of best validation accuracy, mean over 10 seeds.  A graph regime with no
relation vector, which the layer is not designed for.

    run.bat -u bench\cora.py
"""
import json
import os
import pickle
import sys

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402

D = os.path.join(lib.HERE, "data")
KMAX = 64


def load(name):
    objs = []
    for s in ["x", "y", "tx", "ty", "allx", "ally", "graph"]:
        with open(os.path.join(D, f"ind.{name}.{s}"), "rb") as f:
            objs.append(pickle.load(f, encoding="latin1"))
    x, y, tx, ty, allx, ally, graph = objs
    test_idx = [int(l) for l in open(os.path.join(D, f"ind.{name}.test.index"))]
    test_sorted = np.sort(test_idx)
    if name == "citeseer":
        full = range(min(test_idx), max(test_idx) + 1)
        tx_ext = sp.lil_matrix((len(full), x.shape[1])); tx_ext[test_sorted - min(test_idx), :] = tx; tx = tx_ext
        ty_ext = np.zeros((len(full), y.shape[1])); ty_ext[test_sorted - min(test_idx), :] = ty; ty = ty_ext
    feats = sp.vstack((allx, tx)).tolil(); feats[test_idx, :] = feats[test_sorted, :]
    labels = np.vstack((ally, ty)); labels[test_idx, :] = labels[test_sorted, :]
    n = feats.shape[0]
    adj = sp.lil_matrix((n, n))
    for i, nb in graph.items():
        for j in nb:
            if i < n and j < n:
                adj[i, j] = 1; adj[j, i] = 1
    adj.setdiag(0)
    X = np.asarray(feats.todense(), dtype=np.float32)
    X = X / np.maximum(X.sum(1, keepdims=True), 1)
    return torch.tensor(X), torch.tensor(labels.argmax(1)), torch.tensor(np.asarray(adj.todense()) > 0), torch.arange(y.shape[0]), torch.arange(y.shape[0], y.shape[0] + 500), torch.tensor(test_idx)


class LESGraph(nn.Module):
    def __init__(self, cin, width, classes, nbr, pf, depth=2, drop=0.5):
        super().__init__()
        self.net = lib.LESList(cin, width, depth, 2)
        self.out = nn.Linear(width, classes)
        self.nbr, self.pf, self.drop = nbr, pf, drop

    def forward(self, x):
        h = self.net(F.dropout(x, self.drop, self.training), self.nbr, self.pf)
        return self.out(F.dropout(h, self.drop, self.training))


class GCN(nn.Module):
    def __init__(self, cin, width, classes, adj, drop=0.5):
        super().__init__()
        a = adj.float() + torch.eye(len(adj), device=adj.device)
        d = a.sum(1).pow(-0.5)
        self.a = d[:, None] * a * d[None]
        self.l1, self.l2, self.drop = nn.Linear(cin, width), nn.Linear(width, classes), drop

    def forward(self, x):
        h = F.relu(self.a @ self.l1(F.dropout(x, self.drop, self.training)))
        return self.a @ self.l2(F.dropout(h, self.drop, self.training))


class GATLayer(nn.Module):
    def __init__(self, cin, cout, heads, mask, concat=True):
        super().__init__()
        self.W = nn.Linear(cin, heads * cout, bias=False)
        self.al, self.ar = nn.Parameter(torch.randn(heads, cout) * 0.1), nn.Parameter(torch.randn(heads, cout) * 0.1)
        self.h, self.c, self.mask, self.concat = heads, cout, mask, concat

    def forward(self, x):
        z = self.W(x).view(len(x), self.h, self.c)
        e = F.leaky_relu((z * self.al).sum(-1)[:, None, :] + (z * self.ar).sum(-1)[None, :, :], 0.2)      # (N,N,H)
        e = e.masked_fill(~self.mask[:, :, None], float("-inf"))
        w = F.dropout(torch.softmax(e, 1), 0.6, self.training)
        o = torch.einsum("ijh,jhc->ihc", w, z)
        return o.reshape(len(x), -1) if self.concat else o.mean(1)


class GAT(nn.Module):
    def __init__(self, cin, width, classes, adj):
        super().__init__()
        mask = adj | torch.eye(len(adj), dtype=torch.bool, device=adj.device)
        self.l1, self.l2 = GATLayer(cin, width // 8, 8, mask), GATLayer(width, classes, 1, mask, concat=False)

    def forward(self, x):
        h = F.elu(self.l1(F.dropout(x, 0.6, self.training)))
        return self.l2(F.dropout(h, 0.6, self.training))


def lists(adj):
    n = len(adj)
    nbr = torch.full((n, KMAX), -1, dtype=torch.int32)
    deg = adj.sum(1)
    for i in range(n):
        nb = adj[i].nonzero().flatten()[:KMAX - 1]
        nbr[i, 0] = i
        nbr[i, 1:1 + len(nb)] = nb.to(torch.int32)
    pf = torch.zeros(n, KMAX, 2)
    pf[:, 0, 0] = 1.0
    pf[:, :, 1] = (1.0 / deg.clamp_min(1).float().sqrt())[:, None]
    return nbr.to(lib.DEV).contiguous(), pf.to(lib.DEV).contiguous()


def main():
    for name in ("cora", "citeseer"):
        X, y, adj, tr, va, te = load(name)
        X, y, adj = X.to(lib.DEV), y.to(lib.DEV), adj.to(lib.DEV)
        nbr, pf = lists(adj.cpu())
        C = int(y.max()) + 1
        for fam in ("les", "gcn", "gat"):
            for seed in range(10):
                torch.manual_seed(seed)
                model = {"les": lambda: LESGraph(X.shape[1], 64, C, nbr, pf), "gcn": lambda: GCN(X.shape[1], 64, C, adj),
                         "gat": lambda: GAT(X.shape[1], 64, C, adj)}[fam]().to(lib.DEV)
                best = dict(v=0.0, t=0.0)

                def evalf(m):
                    with torch.no_grad():
                        p = m(X).argmax(1)
                    v, t = (p[va] == y[va]).float().mean().item(), (p[te] == y[te]).float().mean().item()
                    if v >= best["v"]:
                        best.update(v=v, t=t)
                    return dict(val_acc=v, test_acc=t, test_at_best_val=best["t"])

                lib.run(f"graph_{name}_{fam}_s{seed}", model, lambda s: None, lambda m, b: F.cross_entropy(m(X)[tr], y[tr]), evalf, steps=200, lr=0.01,
                        eval_every=1, wd=5e-4, clip=0, log_every=1000, extra=dict(family=fam, seed=seed, task=name))


if __name__ == "__main__":
    main()
