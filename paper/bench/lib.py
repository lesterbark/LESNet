"""Shared pieces of the benchmark suite: the LESNet list stack on arbitrary partner lists, small baselines, one training loop.

Every task script builds (a) entities with features and (b) a partner list with a relation tensor per pair, and hands them to
`LESList`; the layer is the one of the paper (fused list kernel), applied `depth` times.  Baselines are ordinary PyTorch.
Each run writes one JSON record with its learning curve (step, wall-clock seconds, training loss, validation metric).
"""
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "lesnet_cuda"))
import kernels  # noqa: E402

DEV = "cuda"
SLOPE = 0.1
RES = os.path.join(HERE, "..", "results", "bench")
os.makedirs(RES, exist_ok=True)


class LESList(nn.Module):
    """depth LES layers on partner lists.  forward(x (R,C), nbr (R,K) int32 rows or -1, pf (R,K,PF)) -> (R,W).
    `nbr` indexes the rows of x (flattened over the batch)."""
    def __init__(self, cin, width, depth, pf_dim, reinject=False):
        super().__init__()
        kernels.require_extension()
        self.reinject = reinject
        dims = [cin] + [width + (cin if reinject else 0)] * (depth - 1)
        self.A, self.B, self.E, self.c = (nn.ParameterList() for _ in range(4))
        for l in range(depth):
            self.A.append(nn.Parameter(torch.randn(dims[l], width) * math.sqrt(1.0 / dims[l])))
            self.B.append(nn.Parameter(torch.randn(dims[l], width) * math.sqrt(1.0 / dims[l])))
            self.E.append(nn.Parameter(torch.randn(pf_dim, width) * 0.5))
            self.c.append(nn.Parameter(torch.zeros(width)))
        self.width = width

    def forward(self, x, nbr, pf):
        h, h0 = x, x
        for l in range(len(self.A)):
            if self.reinject and l > 0:
                h = torch.cat([h, h0], -1)
            h = F.leaky_relu(kernels.les_layer_list(h, h, nbr, pf, self.A[l], self.B[l], self.E[l], self.c[l], need_gx=(l > 0)), SLOPE)
        return h


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def fit_width(build, target, lo=8, hi=1024, step=4):
    """smallest-error width (multiple of `step`) of build(width) to `target` parameters."""
    while hi - lo > step:
        mid = (lo + hi) // (2 * step) * step
        if n_params(build(mid)) < target:
            lo = mid
        else:
            hi = mid
    return min((lo, hi), key=lambda w: abs(n_params(build(w)) - target))


def cosine_lr(step, total, base, warm=200, floor=0.02):
    return base * min(1.0, (step + 1) / warm) * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / max(1, total)))))


def run(name, model, make_batch, loss_fn, evaluate, steps, lr, eval_every, wd=0.0, clip=1.0, extra=None, log_every=100):
    """One training run.  make_batch(step) -> batch (any structure); loss_fn(model, batch) -> loss;
    evaluate(model) -> dict of validation metrics.  Writes results/bench/<name>.json (skips if present)."""
    path = os.path.join(RES, name + ".json")
    if os.path.exists(path):
        print("skip (done):", name)
        return json.load(open(path))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    curve, t_train, run_loss = [], 0.0, None
    print(f"{name}: {n_params(model):,} params, {steps} steps, lr {lr}", flush=True)
    for s in range(1, steps + 1):
        batch = make_batch(s)
        torch.cuda.synchronize(); t0 = time.time()
        for g in opt.param_groups:
            g["lr"] = cosine_lr(s, steps, lr)
        loss = loss_fn(model, batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if clip:
            nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        torch.cuda.synchronize(); t_train += time.time() - t0
        if s % 20 == 0:
            lv = loss.item()
            if not math.isfinite(lv):
                print("diverged", flush=True)
                break
            run_loss = lv if run_loss is None else 0.9 * run_loss + 0.1 * lv
        if s % log_every == 0:
            print(f"  step {s}/{steps}  loss {run_loss:.4f}  {1000 * t_train / s:.1f} ms/step  eta {(steps - s) * t_train / s / 60:.1f} min", flush=True)
        if s % eval_every == 0 or s == steps:
            model.eval()
            with torch.no_grad():
                m = evaluate(model)
            model.train()
            curve.append(dict(step=s, train_s=t_train, train_loss=run_loss, **m))
            print("  [eval]", {k: round(v, 4) for k, v in curve[-1].items() if isinstance(v, float)}, flush=True)
    rec = dict(name=name, params=n_params(model), steps=steps, lr=lr, ms_per_step=1000 * t_train / max(1, s), train_s=t_train,
               curve=curve, final=curve[-1] if curve else None, **(extra or {}))
    json.dump(rec, open(path, "w"))
    return rec
