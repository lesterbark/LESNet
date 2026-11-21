"""What the nbody_acc target is made of: R^2 of the force from the K nearest partners alone and of a linear function of position and centroid.

    run.bat -u nbody_check.py   ->  results/nbody_check.json
"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tasks  # noqa: E402

g = torch.Generator(device="cuda").manual_seed(0)
w = tasks.sample("nbody_acc", 2048, 8, 24, 24, g)
c, m, y = w["coords"], w["mask"], w["y"]
D = tasks._dist(w)
fin = torch.isfinite(D)
out = dict(eps=tasks.NBODY_EPS, mean_entities=float(m.sum(1).float().mean()),
           median_nn_distance=float(D.min(-1).values[m].median()),
           partners_within_eps=float(((D < tasks.NBODY_EPS) & fin).sum(-1).float()[m].mean()))
yy = y[m]
r2 = lambda p: float(1 - ((p - yy) ** 2).sum() / ((yy - yy.mean(0)) ** 2).sum())  # noqa: E731
X = torch.cat([c[m], (c.sum(1, keepdim=True) / m.sum(1)[:, None, None]).expand_as(c)[m], torch.ones(len(yy), 1, device="cuda")], 1)
out["r2_linear_position_centroid"] = r2(X @ torch.linalg.lstsq(X, yy).solution)
d = c[:, None] - c[:, :, None]
r2_ = (d ** 2).sum(-1) + tasks.NBODY_EPS ** 2
mm = torch.where(w["typ"] == 0, 4.0, 1.0) * m
valid = (m[:, :, None] & m[:, None, :]) & ~torch.eye(c.shape[1], dtype=torch.bool, device="cuda")
f = d * (mm[:, None, :] * valid / r2_.pow(1.5))[..., None] / 4.0
for K in (1, 2, 3, 5, 8):
    idx = D.topk(K, dim=-1, largest=False).indices
    sel = torch.zeros_like(D, dtype=torch.bool).scatter_(2, idx, True)
    out[f"r2_nearest_{K}"] = r2((f * sel[..., None]).sum(2)[m])
json.dump(out, open(os.path.join(HERE, "results", "nbody_check.json"), "w"), indent=1)
print(out)
