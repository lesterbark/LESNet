"""CIFAR-10 with pixels as entities: LESNet (3x3 neighbour lists, 2x2 average pooling between stages) against a convolutional network of the same
structure at the same parameter count.  A grid is the regime a CNN is built for; this is the comparison where LESNet is not expected to win.

    run.bat -u bench\cifar.py [--epochs 30] [--seeds 3]
"""
import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402

DATA = os.path.join(lib.HERE, "data")


def load():
    from torchvision.datasets import CIFAR10
    tr, te = CIFAR10(DATA, train=True, download=True), CIFAR10(DATA, train=False, download=True)
    f = lambda d: (torch.from_numpy(d.data).permute(0, 3, 1, 2).float().div(255).to(lib.DEV), torch.tensor(d.targets).to(lib.DEV))
    (xt, yt), (xe, ye) = f(tr), f(te)
    m = xt.mean((0, 2, 3), keepdim=True); s = xt.std((0, 2, 3), keepdim=True)
    return (xt - m) / s, yt, (xe - m) / s, ye


def augment(x, gen):
    B = x.shape[0]
    flip = torch.rand(B, device=x.device, generator=gen) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    p = F.pad(x, (4, 4, 4, 4))
    i = torch.randint(0, 9, (B,), device=x.device, generator=gen); j = torch.randint(0, 9, (B,), device=x.device, generator=gen)
    ar = torch.arange(32, device=x.device)
    rows = (i[:, None] + ar[None]); cols = (j[:, None] + ar[None])
    return p[torch.arange(B, device=x.device)[:, None, None, None], torch.arange(3, device=x.device)[None, :, None, None], rows[:, None, :, None], cols[:, None, None, :]]


# ----------------------------------------------------------------------------------------------- models
def grid_list(r, device, win=3):
    """3x3 neighbourhood (self included) of an r x r grid: local partner index (r*r, 9) (-1 outside) and pair features (r*r, 9, 7)."""
    ys, xs = torch.meshgrid(torch.arange(r), torch.arange(r), indexing="ij")
    ys, xs = ys.reshape(-1), xs.reshape(-1)
    nbr, pf = [], []
    h = win // 2
    for dy in range(-h, h + 1):
        for dx in range(-h, h + 1):
            ny, nx = ys + dy, xs + dx
            ok = (ny >= 0) & (ny < r) & (nx >= 0) & (nx < r)
            nbr.append(torch.where(ok, ny * r + nx, torch.full_like(ny, -1)))
            d = math.hypot(dx, dy)
            pf.append(torch.tensor([dx, dy, abs(dx), abs(dy), d, dx / d if d else 0.0, dy / d if d else 0.0]).expand(r * r, 7))
    return torch.stack(nbr, 1).to(torch.int32).to(device), torch.stack(pf, 1).float().to(device)


class LESImage(nn.Module):
    def __init__(self, w, classes=10, win=3):
        super().__init__()
        self.win = win
        self.ws = [w, 2 * w, 4 * w]
        cin = 3
        self.stages = nn.ModuleList()
        for ws in self.ws:
            self.stages.append(lib.LESList(cin, ws, 2, 7))
            cin = ws
        self.head = nn.Linear(cin, classes)
        self._lists = {}

    def lists(self, r, B):
        if (r, B) not in self._lists:
            nbr, pf = grid_list(r, lib.DEV, self.win)
            K = nbr.shape[1]
            off = (torch.arange(B, device=lib.DEV) * r * r).view(B, 1, 1)
            rows = torch.where(nbr[None] >= 0, nbr[None] + off.to(torch.int32), torch.full_like(nbr[None], -1)).reshape(B * r * r, K).contiguous()
            self._lists[(r, B)] = (rows, pf[None].expand(B, -1, -1, -1).reshape(B * r * r, K, 7).contiguous())
        return self._lists[(r, B)]

    def forward(self, x):
        B, C, r, _ = x.shape
        h = x.permute(0, 2, 3, 1).reshape(B * r * r, C)
        for si, st in enumerate(self.stages):
            nbr, pf = self.lists(r, B)
            h = st(h, nbr, pf)
            if si < len(self.stages) - 1:
                h = F.avg_pool2d(h.view(B, r, r, -1).permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1)
                r //= 2
                h = h.reshape(B * r * r, -1)
        return self.head(h.view(B, r * r, -1).mean(1))


class CNN(nn.Module):
    def __init__(self, w, classes=10):
        super().__init__()
        ws, cin, layers = [w, 2 * w, 4 * w], 3, []
        for i, c in enumerate(ws):
            for _ in range(2):
                layers += [nn.Conv2d(cin, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU()]
                cin = c
            if i < 2:
                layers.append(nn.AvgPool2d(2))
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(cin, classes)

    def forward(self, x):
        return self.head(self.net(x).mean((2, 3)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--les_width", type=int, default=48)
    ap.add_argument("--tag", default="")
    ap.add_argument("--win", type=int, default=3, help="side of the square partner window (3 = 9 partners, 5 = 25)")
    ap.add_argument("--fams", nargs="+", default=["les", "cnn"])
    ap.add_argument("--sweep", action="store_true", help="4-epoch learning-rate sweep on a held-out 5000 of the training set")
    a = ap.parse_args()
    xt, yt, xe, ye = load()
    gen = torch.Generator(device=lib.DEV).manual_seed(0)
    cnn_w = lib.fit_width(lambda w: CNN(w), lib.n_params(LESImage(a.les_width)), lo=4, hi=256, step=2)
    print("LES width", a.les_width, "params", lib.n_params(LESImage(a.les_width)), "| CNN width", cnn_w, "params", lib.n_params(CNN(cnn_w)), flush=True)
    B = 128
    spe = 45000 // B if a.sweep else 50000 // B

    def evalf(x_, y_):
        def f(model):
            c = 0
            for i in range(0, len(x_), 500):
                c += (model(x_[i:i + 500]).argmax(1) == y_[i:i + 500]).sum().item()
            return dict(acc=c / len(x_))
        return f

    for fam, mk in [f for f in (("les", lambda: LESImage(a.les_width, win=a.win)), ("cnn", lambda: CNN(cnn_w))) if f[0] in a.fams]:
        lrs = (1e-3, 3e-3, 1e-2, 3e-2) if a.sweep else (json_best(fam),)
        for lr in lrs:
            for seed in (range(1) if a.sweep else range(a.seeds)):
                torch.manual_seed(seed)
                g = torch.Generator(device=lib.DEV).manual_seed(seed)
                model = mk().to(lib.DEV)
                n_tr = 45000 if a.sweep else 50000
                perm = torch.randperm(50000, device=lib.DEV, generator=g)
                tr_idx = perm[:n_tr]

                def make_batch(s, tr_idx=tr_idx, g=g):
                    idx = tr_idx[torch.randint(0, len(tr_idx), (B,), device=lib.DEV, generator=g)]
                    return augment(xt[idx], g), yt[idx]

                ev = evalf(xt[perm[45000:]], yt[perm[45000:]]) if a.sweep else evalf(xe, ye)
                name = f"cifar{a.tag}_{'sweep_' if a.sweep else ''}{fam}_lr{lr:g}_s{seed}"
                lib.run(name, model, make_batch, lambda m, b: F.cross_entropy(m(b[0]), b[1]), ev, steps=(4 if a.sweep else a.epochs) * spe,
                        lr=lr, eval_every=spe, wd=5e-4 if fam == "cnn" else 1e-4, extra=dict(family=fam, seed=seed, task="cifar10"))


def json_best(fam):
    import json
    best = {}
    for lr in (1e-3, 3e-3, 1e-2, 3e-2):
        p = os.path.join(lib.RES, f"cifar_sweep_{fam}_lr{lr:g}_s0.json")
        if os.path.exists(p):
            best[lr] = json.load(open(p))["final"]["acc"]
    return max(best, key=best.get) if best else 3e-3


if __name__ == "__main__":
    main()
