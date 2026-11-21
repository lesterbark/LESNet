"""MNIST as a point set: the pixels above half intensity (at most 128, randomly kept) are an unordered set of 2-D points with an intensity feature.
Classification of the digit from the set, by LESNet (16 nearest points), a nearest-neighbour graph network, a Transformer with a relation bias,
and DeepSets, all at about 150k parameters.  A real (non-synthetic) variable-size set with geometric relations, no grid, no order.

    run.bat -u bench\pointset.py [--sweep] [--epochs 20] [--seeds 3]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import lib  # noqa: E402
import models as pm  # noqa: E402

NMAX = 128
PF = "pos,axis,polar"


def load():
    from torchvision.datasets import MNIST
    out = []
    for train in (True, False):
        d = MNIST(os.path.join(lib.HERE, "data"), train=train, download=True)
        out.append((d.data.to(lib.DEV).float() / 255, d.targets.to(lib.DEV)))
    return out


def to_points(img, gen):
    """img (B,28,28) -> x (B,N,1) intensity, coords (B,N,2) in pixels centred, mask (B,N)."""
    B = img.shape[0]
    on = img > 0.5
    score = torch.rand(B, 28, 28, device=img.device, generator=gen) * on - (~on).float()
    top = score.view(B, -1).topk(NMAX, dim=1)
    valid = top.values >= 0
    yy, xx = top.indices // 28, top.indices % 28
    coords = torch.stack([xx, yy], -1).float() - 13.5
    x = torch.gather(img.view(B, -1), 1, top.indices)[..., None]
    return x * valid[..., None], coords * valid[..., None], valid


class SetNet(nn.Module):
    def __init__(self, trunk, width, classes=10):
        super().__init__()
        self.trunk = trunk
        self.head = nn.Sequential(nn.Linear(2 * width, width), nn.LeakyReLU(0.1), nn.Linear(width, classes))

    def forward(self, x, coords, mask):
        h = self.trunk(x, coords, mask)
        m = mask[..., None].float()
        mean = (h * m).sum(1) / m.sum(1).clamp_min(1)
        mx = h.masked_fill(~mask[..., None], -1e4).max(1).values
        return self.head(torch.cat([mean, mx], -1))


def build(fam, w, depth=3, k=16):
    if fam == "les":
        t = pm.LESTrunk(1, w, depth, pf_spec=PF, k=k, radius=None)
    elif fam == "gnn":
        t = pm.GatherGNNTrunk(1, w, depth, PF, k, None)
    elif fam == "transformer":
        t = pm.TransformerTrunk(1, w, depth, heads=4, pf_spec=PF)
    else:
        t = pm.DeepSetsTrunk(1, w, depth)
    return SetNet(t, w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--params", type=int, default=150_000)
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()
    (xt, yt), (xe, ye) = load()
    B = 128
    spe = (55000 if a.sweep else 60000) // B
    for fam in ("les", "gnn", "transformer", "deepsets"):
        step = 8 if fam == "transformer" else 4
        w = lib.fit_width(lambda w_: build(fam, w_), a.params, lo=8, hi=512, step=step)
        lrs = (1e-3, 3e-3, 1e-2, 3e-2) if a.sweep else (best_lr(fam),)
        for lr in lrs:
            for seed in (range(1) if a.sweep else range(a.seeds)):
                torch.manual_seed(seed)
                g = torch.Generator(device=lib.DEV).manual_seed(seed)
                model = build(fam, w).to(lib.DEV)
                perm = torch.randperm(60000, device=lib.DEV, generator=g)
                tr_idx = perm[:55000] if a.sweep else perm
                val = (xt[perm[55000:]], yt[perm[55000:]]) if a.sweep else (xe, ye)

                def make_batch(s, tr_idx=tr_idx, g=g):
                    idx = tr_idx[torch.randint(0, len(tr_idx), (B,), device=lib.DEV, generator=g)]
                    return (*to_points(xt[idx], g), yt[idx])

                def evalf(m, val=val):
                    ge, c = torch.Generator(device=lib.DEV).manual_seed(7), 0
                    for i in range(0, len(val[0]), 500):
                        c += (m(*to_points(val[0][i:i + 500], ge)).argmax(1) == val[1][i:i + 500]).sum().item()
                    return dict(acc=c / len(val[0]))

                name = f"pointset_{'sweep_' if a.sweep else ''}{fam}_lr{lr:g}_s{seed}"
                lib.run(name, model, make_batch, lambda m, b: F.cross_entropy(m(b[0], b[1], b[2]), b[3]), evalf,
                        steps=(3 if a.sweep else a.epochs) * spe, lr=lr, eval_every=spe, wd=1e-4, extra=dict(family=fam, seed=seed, width=w, task="mnist_points"))


def best_lr(fam):
    best = {}
    for lr in (1e-3, 3e-3, 1e-2, 3e-2):
        p = os.path.join(lib.RES, f"pointset_sweep_{fam}_lr{lr:g}_s0.json")
        if os.path.exists(p):
            best[lr] = json.load(open(p))["final"]["acc"]
    return max(best, key=best.get) if best else 3e-3


if __name__ == "__main__":
    main()
