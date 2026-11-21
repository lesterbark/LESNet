"""Top-quark jet tagging (Top Quark Tagging Reference Dataset, Kasieczka et al. 2019, Zenodo 2603256).

A jet is a set of up to 100 constituent particles (E, px, py, pz).  The task is to tell top-quark jets from QCD jets.
A published benchmark with a standard metric set, no grid, geometric relations (distance in the eta-phi plane), and
a variable number of entities per example: exactly LESNet's setting.  Everything here is trained from scratch by us
under one protocol (same data stream, optimiser, schedule, parameter budget), so the comparison between the models is
controlled; the comparison with numbers published for other architectures is only indicative.

    run.bat -u toptag.py --family les --epochs 4 --name les_s0
    run.bat -u toptag.py --family gnn --epochs 4 --name gnn_s0

Features per constituent follow ParticleNet: (log pT, log E, log pT/pT_jet, log E/E_jet, dR to the jet axis);
coordinates (d_eta, d_phi) relative to the jet axis; the relation between two constituents is their coordinate offset
(pair features 'pos,axis,polar').  Neighbours: the 16 nearest in the (eta, phi) plane (LESNet, GNN); full attention
for the Transformer.
"""
import argparse
import json
import math
import os
import sys
import threading
import queue
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "lesnet_cuda"))
import models as pm  # noqa: E402

D = os.path.join(HERE, "data_toptag")
OUT = os.path.join(HERE, "results", "toptag_v2")
DEV = "cuda"
COORD_SCALE = 10.0
CIN = 5
PF = "pos,axis,polar"
SLOPE = 0.1


# ------------------------------------------------------------------------------------- data
class Split:
    def __init__(self, name, limit=None):
        self.X = np.load(os.path.join(D, f"{name}_X.npy"), mmap_mode="r")
        self.y = np.load(os.path.join(D, f"{name}_y.npy"))
        self.n = len(self.y) if limit is None else min(limit, len(self.y))

    def batch(self, idx):
        idx = np.sort(idx)
        return torch.from_numpy(np.ascontiguousarray(self.X[idx])), torch.from_numpy(self.y[idx].astype(np.float32))


def featurize(X):
    """X (B,100,4) = (E,px,py,pz) on the GPU -> node features (B,N,5), coords (B,N,2), mask (B,N)."""
    E, px, py, pz = X.unbind(-1)
    pt = torch.hypot(px, py)
    mask = E > 0
    jpx, jpy, jpz, jE = (t.sum(1, keepdim=True) for t in (px, py, pz, E))
    jpt = torch.hypot(jpx, jpy).clamp_min(1e-6)
    jeta = torch.asinh(jpz / jpt)
    jphi = torch.atan2(jpy, jpx)
    eta = torch.asinh(pz / pt.clamp_min(1e-6))
    phi = torch.atan2(py, px)
    deta = eta - jeta
    dphi = torch.remainder(phi - jphi + math.pi, 2 * math.pi) - math.pi
    dr = torch.hypot(deta, dphi)
    lpt = torch.log(pt.clamp_min(1e-6))
    lE = torch.log(E.clamp_min(1e-6))
    f = torch.stack([lpt, lE, lpt - torch.log(jpt), lE - torch.log(jE.clamp_min(1e-6)), dr], -1)
    f = (f - FEAT_MEAN.to(X.device)) / FEAT_STD.to(X.device)
    coords = torch.stack([deta, dphi], -1) * COORD_SCALE
    m = mask[..., None]
    return f * m, coords * m, mask


FEAT_MEAN = torch.tensor([1.0, 2.0, -3.2, -3.0, 0.3])           # rough constants from the data; re-estimated in fit_norm()
FEAT_STD = torch.tensor([1.0, 1.0, 0.8, 0.8, 0.2])


def fit_norm(split, n=20000):
    global FEAT_MEAN, FEAT_STD
    X, _ = split.batch(np.random.default_rng(0).choice(split.n, n, replace=False))
    X = X.to(DEV)
    FEAT_MEAN, FEAT_STD = torch.zeros(5), torch.ones(5)
    f, _, mask = featurize(X)
    v = f[mask]
    FEAT_MEAN, FEAT_STD = v.mean(0).cpu(), v.std(0).cpu().clamp_min(1e-3)


# ------------------------------------------------------------------------------------- models
class JetNet(nn.Module):
    def __init__(self, family, width, depth, k=16):
        super().__init__()
        self.family = family
        if family in ("les", "les_cat"):
            self.trunk = pm.LESTrunk(CIN, width, depth, pf_spec=PF, k=k, radius=None, reinject=(family == "les_cat"))
        elif family == "gnn":
            self.trunk = pm.GatherGNNTrunk(CIN, width, depth, PF, k, None)
        elif family == "gen":
            self.trunk = pm.GatherGenTrunk(CIN, width, depth, PF, k, None)
        elif family == "transformer":
            self.trunk = pm.TransformerTrunk(CIN - 0, width, depth, heads=4, pf_spec=PF)
        elif family == "deepsets":
            self.trunk = pm.DeepSetsTrunk(CIN, width, depth)
        else:
            raise KeyError(family)
        self.head = nn.Sequential(nn.Linear(2 * width, width), nn.LeakyReLU(SLOPE), nn.Linear(width, 1))

    def forward(self, f, coords, mask):
        h = self.trunk(f, coords, mask)
        m = mask[..., None].float()
        mean = (h * m).sum(1) / m.sum(1).clamp_min(1)
        mx = h.masked_fill(~mask[..., None], torch.finfo(h.dtype).min / 4).max(1).values
        return self.head(torch.cat([mean, mx], -1)).squeeze(-1)


def build(family, width, depth, k=16):
    return JetNet(family, width, depth, k).to(DEV)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def fit_width(family, depth, target, k=16):
    step = 8 if family == "transformer" else 4
    lo, hi = 8, 512
    while hi - lo > step:
        mid = (lo + hi) // (2 * step) * step
        if n_params(build(family, mid, depth, k)) < target:
            lo = mid
        else:
            hi = mid
    return min((lo, hi), key=lambda w: abs(n_params(build(family, w, depth, k)) - target))


# ------------------------------------------------------------------------------------- train / eval
def prefetch(split, order, bs, steps, out_q):
    for s in range(steps):
        idx = order[s * bs:(s + 1) * bs]
        out_q.put(split.batch(idx))
    out_q.put(None)


@torch.no_grad()
def evaluate(net, split, bs=2048, limit=None):
    net.eval()
    n = split.n if limit is None else min(limit, split.n)
    scores = []
    for s in range(0, n, bs):
        X, y = split.batch(np.arange(s, min(n, s + bs)))
        f, c, m = featurize(X.to(DEV))
        scores.append(net(f, c, m).float().cpu())
    net.train()
    return torch.cat(scores).numpy(), split.y[:n]


def metrics(score, y):
    from sklearn.metrics import roc_auc_score, roc_curve
    auc = roc_auc_score(y, score)
    fpr, tpr, _ = roc_curve(y, score)
    eb30 = 1.0 / max(fpr[np.searchsorted(tpr, 0.3)], 1e-9)
    eb50 = 1.0 / max(fpr[np.searchsorted(tpr, 0.5)], 1e-9)
    acc = float(((score > 0) == (y > 0.5)).mean())
    return dict(auc=float(auc), acc=acc, rej30=float(eb30), rej50=float(eb50))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="les")
    ap.add_argument("--width", type=int, default=0, help="0 = fit to --params")
    ap.add_argument("--params", type=int, default=150_000)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=0, help="exact number of optimiser steps (overrides --epochs)")
    ap.add_argument("--budget_s", type=float, default=0, help="stop after this many seconds of training (schedule set from the measured step time)")
    ap.add_argument("--train_jets", type=int, default=0, help="0 = all 1.2M")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--eval_every", type=int, default=0, help="validation every N steps (0 = once per epoch)")
    ap.add_argument("--val_jets", type=int, default=40000)
    ap.add_argument("--test", action="store_true", help="evaluate the final model on the full test split")
    ap.add_argument("--eval_bs", type=int, default=2048, help="evaluation batch (the graph net's gathered messages need a small one on a 6 GB card)")
    ap.add_argument("--train_split", default="train", help="split used for training (val for smoke tests)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    name = a.name or f"{a.family}_s{a.seed}"
    torch.manual_seed(a.seed)
    train = Split(a.train_split, a.train_jets or None)
    val = Split("val", a.val_jets)
    fit_norm(train)
    width = a.width or fit_width(a.family, a.depth, a.params, a.k)
    net = build(a.family, width, a.depth, a.k)
    print(f"{name}: family {a.family} width {width} depth {a.depth} params {n_params(net)} lr {a.lr} batch {a.batch}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    per_epoch = train.n // a.batch
    total = a.steps or int(a.epochs * per_epoch)
    if a.budget_s:                                   # iso-time: measure 40 steps first, then fix the schedule length
        total = 10 ** 9
    rng = np.random.default_rng(a.seed)
    n_perm = int(math.ceil(min(total, 60000) * a.batch / train.n)) + 1
    order = np.concatenate([rng.permutation(train.n) for _ in range(n_perm)])
    q = queue.Queue(maxsize=6)
    th = threading.Thread(target=prefetch, args=(train, order, a.batch, min(total, len(order) // a.batch), q), daemon=True)
    th.start()
    hist, t0, run, step = [], time.time(), None, 0
    every = a.eval_every or per_epoch
    while True:
        item = q.get()
        if item is None:
            break
        X, y = item
        f, c, m = featurize(X.to(DEV, non_blocking=True))
        y = y.to(DEV)
        if a.budget_s and step == 40:
            torch.cuda.synchronize()
            total = int(a.budget_s / ((time.time() - t0) / 40))
            print(f"step time {1000 * (time.time() - t0) / 40:.0f} ms -> {total} steps for {a.budget_s / 60:.0f} min", flush=True)
            every = max(200, total // 6)
        if step >= total:
            break
        lr = a.lr * min(1.0, (step + 1) / min(300, max(1, total // 10))) * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * step / total)))
        for g in opt.param_groups:
            g["lr"] = lr
        loss = F.binary_cross_entropy_with_logits(net(f, c, m), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 25 == 0:
            lv = loss.item()
            run = lv if run is None else 0.9 * run + 0.1 * lv
        if step % 200 == 0:
            el = time.time() - t0
            print(f"step {step}/{total}  loss {run:.4f}  {1000 * el / step:.0f} ms/step  eta {(total - step) * el / step / 60:.1f} min", flush=True)
        if step % every == 0 or step == total or (a.budget_s and step == total):
            s, yy = evaluate(net, val, bs=a.eval_bs)
            mt = metrics(s, yy)
            hist.append(dict(step=step, train_loss=run, **{f"val_{k}": v for k, v in mt.items()}))
            print(f"  [validation] step {step}: AUC {mt['auc']:.4f}  acc {mt['acc']:.4f}  1/eB@30% {mt['rej30']:.0f}", flush=True)
    rec = dict(name=name, family=a.family, width=width, depth=a.depth, k=a.k, params=n_params(net), lr=a.lr, seed=a.seed,
               epochs=a.epochs, train_jets=train.n, batch=a.batch, steps=step, train_s=time.time() - t0,
               ms_per_step=1000 * (time.time() - t0) / max(step, 1), hist=hist)
    torch.save(net.state_dict(), os.path.join(OUT, f"{name}.pt"))        # so that a failed evaluation does not lose the training
    if a.test:
        test = Split("test")
        s, yy = evaluate(net, test, bs=a.eval_bs)
        rec["test"] = metrics(s, yy)
        print(f"TEST {name}: AUC {rec['test']['auc']:.4f}  acc {rec['test']['acc']:.4f}  1/eB@30% {rec['test']['rej30']:.0f}", flush=True)
    with open(os.path.join(OUT, f"{name}.json"), "w") as fh:
        json.dump(rec, fh, indent=1)


if __name__ == "__main__":
    main()
