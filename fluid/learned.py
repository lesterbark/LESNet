"""Learned fluid simulators (LESNet and a GNS-style message-passing baseline) and their rollouts.

The learned step maps the particle state at frame t to the state `M` frames later:

    inputs   per particle: velocity, velocity change over the last frame, external acceleration
             (gravity + finger), distance to each wall.  Pair features: relative position / distance /
             direction to each of the K nearest neighbours within RADIUS.
    output   an acceleration a;   v' = v + dt a,   x' = x + dt v'          (dt = M frames of native time)

A frame is STRIDE native substeps.  Larger M = fewer learned steps per simulated second = faster, and a
harder thing to learn: the speed/accuracy trade-off the paper reports.
"""
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "paper"))
sys.path.insert(0, os.path.join(HERE, "..", "lesnet_cuda"))
import kernels  # noqa: E402
import neighbors  # noqa: E402
import models as pm  # noqa: E402  (paper/models.py: LESTrunk, pf_list)
import sph  # noqa: E402
import gen_data  # noqa: E402

DEV = "cuda"
SCALE = 20.0                       # coordinates are multiplied by this before pair features (1 unit = 0.05 m)
RADIUS = 0.07                      # neighbour radius in metres
K = 24
WALL_R = 0.1
PF_SPEC = "pos,axis,polar"
CIN = 10
V_SCALE = 3.0                      # m/s
A_SCALE = 30.0                     # m/s^2
LES_KW = dict(pf_spec=PF_SPEC, k=K, radius=RADIUS * SCALE)


# ------------------------------------------------------------------------------------- data
class MM:
    """Memory-mapped (T,N,2) array that indexes like a torch tensor (float32 out), so large datasets are shared
    through the OS file cache between processes instead of being copied into each one's memory."""
    def __init__(self, path):
        self.a = np.load(path, mmap_mode="r")
        self.shape = self.a.shape

    def __getitem__(self, i):
        return torch.from_numpy(np.ascontiguousarray(self.a[i]).astype(np.float32))


class Data:
    """root=None: the original data/ (npz, 64 train trajectories).  root='data2': the v2 set (npy, memory-mapped).
    mirror=True: each training sample is reflected left-right with probability 1/2 (positions, velocities, gravity and finger)."""
    def __init__(self, split, root=None, mirror=False):
        self.mirror = mirror
        self.trajs = []
        d = os.path.join(HERE, root) if root else os.path.join(gen_data.DATA)
        if root:
            for f in sorted(f for f in os.listdir(d) if f.startswith(split + "_") and f.endswith("_pos.npy")):
                b = os.path.join(d, f[:-len("_pos.npy")])
                self.trajs.append(dict(pos=MM(b + "_pos.npy"), vel=MM(b + "_vel.npy"),
                                       g=torch.from_numpy(np.load(b + "_g.npy")), finger=torch.from_numpy(np.load(b + "_finger.npy"))))
            idx = os.path.join(d, "index.json")
            kinds = json.load(open(idx)) if os.path.exists(idx) else {}
            self.kinds = [kinds.get(f"{split}_{i:03d}", {}).get("kind") for i in range(len(self.trajs))]
        else:
            files = sorted(f for f in os.listdir(d) if f.startswith(split + "_") and f.endswith(".npz"))
            for f in files:
                z = np.load(os.path.join(d, f))
                self.trajs.append(dict(pos=torch.from_numpy(z["pos"]), vel=torch.from_numpy(z["vel"].astype(np.float32)),
                                       g=torch.from_numpy(z["g"]), finger=torch.from_numpy(z["finger"])))
            self.kinds = [None] * len(self.trajs)
        self.frames = self.trajs[0]["pos"].shape[0]
        self.nmax = max(t["pos"].shape[1] for t in self.trajs)

    def sample(self, B, M, rng):
        idx = rng.integers(0, len(self.trajs), B)
        t = rng.integers(1, self.frames - M, B)
        flip = rng.random(B) < 0.5 if self.mirror else None
        return self.collect(idx, t, M, flip)

    def collect(self, idx, t, M, flip=None):
        B, N = len(idx), self.nmax
        out = {k: torch.zeros(B, N, 2) for k in ("pos", "vel", "prev", "next")}
        mask = torch.zeros(B, N, dtype=torch.bool)
        g, fin = torch.zeros(B, 2), torch.zeros(B, 5)
        for b, (i, tt) in enumerate(zip(idx, t)):
            tr = self.trajs[i]
            n = tr["pos"].shape[1]
            out["pos"][b, :n] = tr["pos"][tt]
            out["vel"][b, :n] = tr["vel"][tt]
            out["prev"][b, :n] = tr["vel"][max(tt - 1, 0)]
            out["next"][b, :n] = tr["vel"][tt + M]
            out["pos_next"] = out.get("pos_next", torch.zeros(B, N, 2))
            out["pos_next"][b, :n] = tr["pos"][tt + M]
            mask[b, :n] = True
            g[b], fin[b] = tr["g"][tt], tr["finger"][tt]
            if flip is not None and flip[b]:                      # left-right reflection of the whole sample
                for k in ("pos", "pos_next"):
                    out[k][b, :n, 0] = 1.0 - out[k][b, :n, 0]
                for k in ("vel", "prev", "next"):
                    out[k][b, :n, 0] = -out[k][b, :n, 0]
                g[b, 0] = -g[b, 0]
                fin[b, 0] = 1.0 - fin[b, 0]
        out = {k: v.to(DEV, non_blocking=True) for k, v in out.items()}
        return out, mask.to(DEV), g.to(DEV), fin.to(DEV)


def ext_batch(pos, g, fin):
    """Per-particle external acceleration, batched.  pos (B,N,2), g (B,2), fin (B,5)."""
    ext = g[:, None, :].expand_as(pos).clone()
    c, acc, rad, act = fin[:, None, 0:2], fin[:, 2, None, None], fin[:, 3, None, None], fin[:, 4, None, None]
    d = pos - c
    r = d.norm(dim=-1, keepdim=True).clamp_min(1e-4)
    fall = (1.0 - r / rad).clamp_min(0.0)
    return ext + act * acc * fall * d / r


def node_features(pos, vel, prev, ext):
    wall = torch.stack([pos[..., 0], 1 - pos[..., 0], pos[..., 1], 1 - pos[..., 1]], -1)
    wall = (wall / WALL_R).clamp(0.0, 1.0)
    return torch.cat([vel / V_SCALE, (vel - prev) / V_SCALE * 10.0, ext / 10.0, wall], -1)


# ------------------------------------------------------------------------------------- models
class LearnedSim(nn.Module):
    def __init__(self, kind="les", width=128, depth=4):
        super().__init__()
        self.kind = kind
        if kind in ("les", "les_cat"):                   # les_cat: input entity variables re-concatenated at every layer
            self.trunk = pm.LESTrunk(CIN, width, depth, reinject=(kind == "les_cat"), **LES_KW)
        else:
            self.trunk = pm.GatherGNNTrunk(CIN, width, depth, PF_SPEC, K, RADIUS * SCALE)
        self.head = nn.Linear(width, 2)
        nn.init.zeros_(self.head.bias)
        self.head.weight.data.mul_(0.1)

    def forward(self, pos, vel, prev, ext, mask):
        x = node_features(pos, vel, prev, ext)
        h = self.trunk(x, pos * SCALE, mask)
        return self.head(h) * A_SCALE * mask[..., None]

    def step(self, pos, vel, prev, ext, mask, dt):
        acc = self(pos, vel, prev, ext, mask)
        v2 = vel + dt * acc
        x2 = pos + dt * v2
        # walls: same treatment as the native integrator
        lo, hi = x2 < 0, x2 > 1
        x2 = x2.clamp(0, 1)
        v2 = torch.where(lo, v2.abs() * 0.3, v2)
        v2 = torch.where(hi, -v2.abs() * 0.3, v2)
        return x2, v2


def count_params(m):
    return sum(p.numel() for p in m.parameters())


# ------------------------------------------------------------------------------------- rollout
@torch.no_grad()
def rollout(model, traj, t0, steps, M, nmax=None):
    """Autoregressive rollout from the native state at frame t0.  Returns pos (steps+1,N,2) and vel."""
    pos = traj["pos"][t0].to(DEV)[None]
    vel = traj["vel"][t0].to(DEV)[None]
    prev = traj["vel"][max(t0 - 1, 0)].to(DEV)[None]
    N = pos.shape[1]
    mask = torch.ones(1, N, dtype=torch.bool, device=DEV)
    dt = M * sph.DT * gen_data.STRIDE
    P, V = [pos[0].clone()], [vel[0].clone()]
    for i in range(steps):
        f = min(t0 + i * M, traj["g"].shape[0] - 1)
        ext = ext_batch(pos, traj["g"][f].to(DEV)[None], traj["finger"][f].to(DEV)[None])
        new_pos, new_vel = model.step(pos, vel, prev, ext, mask, dt)
        prev = vel
        pos, vel = new_pos, new_vel
        P.append(pos[0].clone()); V.append(vel[0].clone())
    return torch.stack(P), torch.stack(V)


def chamfer(a, b):
    """Mean nearest-neighbour distance between two point sets, symmetric (permutation-free)."""
    d = torch.cdist(a, b)
    return 0.5 * (d.min(1).values.mean() + d.min(0).values.mean())
