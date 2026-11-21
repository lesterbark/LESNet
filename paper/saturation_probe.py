"""Softmax weight entropy and gradient spread over training at several learning rates (torch twin, dense).

    run.bat -u saturation_probe.py   ->  results/probe/saturation.json
"""
import json
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import models  # noqa: E402
import tasks  # noqa: E402
import train  # noqa: E402

STATS = []


class ProbeLayer(models.TorchLESLayer):
    def forward(self, x, pf, valid):
        z = (x @ self.A)[:, :, None, :] + (x @ self.B)[:, None, :, :] + self.c
        if self.E is not None:
            z = z + pf @ self.E
        v = self.act(z)
        vm = valid[..., None]
        if self.training and torch.is_grad_enabled():
            v.retain_grad()
        w = torch.softmax(v.masked_fill(~vm, torch.finfo(v.dtype).min / 4), 2) * vm
        y = (w * v).sum(2)
        STATS.append((w.detach(), v, vm))
        y = torch.where(valid.any(2)[..., None], y, torch.zeros_like(y))
        return self.act(y)


def summarise():
    ent, wmax, gtop, gfrac = [], [], [], []
    for w, v, vm in STATS:
        n = vm.sum(2).float().clamp_min(2)
        h = -(w.clamp_min(1e-30) * w.clamp_min(1e-30).log()).sum(2) / n.log()
        ok = vm.sum(2) > 1
        ent.append(h[ok.expand_as(h)].mean().item())
        wmax.append(w.max(2).values[ok.expand_as(h)].mean().item())
        if v.grad is not None:
            g = (v.grad * vm).abs()
            tot = g.sum(2).clamp_min(1e-30)
            top = g.max(2).values
            gtop.append((top / tot)[ok.expand_as(tot)].mean().item())
            gfrac.append(((g > 1e-3 * (tot / n).unsqueeze(2)).float().sum(2) / n)[ok.expand_as(tot)].mean().item())
    f = lambda a: sum(a) / max(1, len(a))  # noqa: E731
    return dict(norm_entropy=f(ent), mean_max_weight=f(wmax), grad_top_share=f(gtop), grad_active_frac=f(gfrac))


def run(task, lr, steps=1500, seed=0, width=96, depth=3, B=128):
    torch.manual_seed(seed)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    trunk = models.LESTorchTrunk(5, width, depth, pf_spec="pos,axis,polar")
    trunk.layers = torch.nn.ModuleList(
        ProbeLayer(l.A.shape[0], width, models.pf_width("pos,axis,polar")) for l in trunk.layers)
    net = models.Net(trunk, task).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 100) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))))
    rows = []
    for step in range(1, steps + 1):
        STATS.clear()
        w = tasks.sample(task, B, 8, 24, 24, gen, "cuda")
        loss = tasks.loss_fn(task, net(w["x"], w["coords"], w["mask"]), w)
        opt.zero_grad(); loss.backward()
        if step in (1, 25, 50, 100, 200, 400, 800, 1500):
            d = summarise(); d.update(step=step, loss=loss.item()); rows.append(d)
            print(task, f"lr={lr:g}", {k: round(v, 3) for k, v in d.items()}, flush=True)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
    return dict(task=task, lr=lr, final=train.evaluate(net, task, 8, 24, 24, 1024), rows=rows)


if __name__ == "__main__":
    out = [run(t, lr) for t in ("local_vel_avg", "nn_type") for lr in (1e-3, 3e-3, 1e-2, 3e-2)]
    json.dump(out, open(os.path.join(HERE, "results", "probe", "saturation.json"), "w"), indent=1)
    print("PROBE DONE", flush=True)
