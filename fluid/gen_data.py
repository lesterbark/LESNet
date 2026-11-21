"""Generate native-SPH trajectories for training and testing the learned simulator.

    run.bat -u gen_data.py            # data/train_*.npz, data/val_*.npz, data/test_*.npz

Each trajectory: a random scene (1-3 lattice blocks of fluid, some falling), simulated for DURATION
seconds under gravity that occasionally rotates and a "finger" that sometimes pushes or pulls on the
fluid.  Both are the controls the learned simulator is given as an input and the viewer lets you drive.

Saved every STRIDE substeps: pos (T,N,2) float32, vel (T,N,2) float16, g (T,2), finger (T,5) =
(cx, cy, accel, radius, active).  `ext_for` rebuilds the per-particle external acceleration from them.
"""
import math
import os
import sys
import time

import numpy as np
import torch

import sph

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
STRIDE = 16                         # substeps per stored frame  (16 * DT = 2.08 ms)
DURATION = 1.5                      # seconds of simulated time
N_MIN, N_MAX = 500, 1500


def ext_for(pos, g, finger):
    """Per-particle external acceleration: gravity plus the finger field.  pos (N,2) torch."""
    ext = torch.zeros_like(pos)
    ext[:, 0], ext[:, 1] = float(g[0]), float(g[1])
    cx, cy, acc, rad, active = (float(v) for v in finger)
    if active > 0.5:
        d = pos - torch.tensor([cx, cy], device=pos.device)
        r = d.norm(dim=1, keepdim=True).clamp_min(1e-4)
        fall = (1.0 - r / rad).clamp_min(0.0)
        ext = ext + acc * fall * d / r
    return ext


def random_scene(rng, n_min=None, n_max=None):
    n_min, n_max = n_min or N_MIN, n_max or N_MAX
    for _ in range(400):
        blocks, tot = [], 0
        for _b in range(rng.integers(1, 4)):
            w, h = rng.uniform(0.15, 0.5), rng.uniform(0.15, 0.6)
            x0, y0 = rng.uniform(0.0, 1.0 - w), rng.uniform(0.0, 1.0 - h)
            if any(not (x0 + w + 0.03 < bx or bx + bw + 0.03 < x0 or y0 + h + 0.03 < by or by + bh + 0.03 < y0)
                   for bx, by, bw, bh in blocks):
                continue
            blocks.append((x0, y0, w, h))
            tot += int(round(w / sph.SPACING)) * int(round(h / sph.SPACING))
        if n_min <= tot <= n_max:
            break
    pos, vel = [], []
    for x0, y0, w, h in blocks:
        p = sph.lattice_block(x0, y0, w, h)
        v0 = torch.tensor(rng.uniform(-1.0, 1.0, 2), dtype=torch.float32)
        pos.append(p)
        vel.append(v0.expand_as(p) + 0.1 * torch.randn_like(p))
    return torch.cat(pos), torch.cat(vel)


def big_scene(rng, n_min, n_max):
    """One or two large blocks filling most of the box: more particles than any training scene."""
    target = rng.integers(n_min, n_max + 1)
    w = rng.uniform(0.75, 0.97)
    h = min(0.95, target * sph.SPACING ** 2 / w)
    x0, y0 = rng.uniform(0.0, 1.0 - w), rng.uniform(0.0, max(0.0, 1.0 - h))
    pos = sph.lattice_block(x0, y0, w, h)
    vel = torch.randn(1, 2) * 0.3 + 0.1 * torch.randn_like(pos)
    return pos, vel


def control_schedule(rng, frames):
    """Per-frame gravity vectors and finger parameters."""
    dt_frame = STRIDE * sph.DT
    g = np.zeros((frames, 2), np.float32)
    finger = np.zeros((frames, 5), np.float32)
    t = 0
    ang = 0.0
    while t < frames:
        n = max(1, int(rng.uniform(0.2, 0.4) / dt_frame))
        ang = rng.uniform(-1.0, 1.0) if rng.random() < 0.4 else 0.0
        g[t:t + n] = sph.GRAVITY * np.array([math.sin(ang), -math.cos(ang)], np.float32)
        t += n
    t = 0
    while t < frames:
        gap = int(rng.uniform(0.1, 0.4) / dt_frame)
        n = int(rng.uniform(0.1, 0.35) / dt_frame)
        t += gap
        if t >= frames:
            break
        c0 = rng.uniform(0.1, 0.9, 2)
        vel = rng.uniform(-0.6, 0.6, 2)
        acc = rng.choice([-1.0, 1.0]) * rng.uniform(20, 60)
        rad = rng.uniform(0.10, 0.2)
        for k in range(min(n, frames - t)):
            c = np.clip(c0 + vel * k * dt_frame, 0.02, 0.98)
            finger[t + k] = (c[0], c[1], acc, rad, 1.0)
        t += n
    return g, finger


def simulate(seed, n_min=None, n_max=None):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    pos, vel = big_scene(rng, n_min, n_max) if (n_min or 0) >= 1500 else random_scene(rng, n_min, n_max)
    sim = sph.Sim(pos, vel)
    frames = int(DURATION / (STRIDE * sph.DT))
    g, finger = control_schedule(rng, frames)
    P = np.zeros((frames, len(pos), 2), np.float32)
    V = np.zeros((frames, len(pos), 2), np.float16)
    for f in range(frames):
        P[f] = sim.pos.cpu().numpy()
        V[f] = sim.vel.cpu().numpy().astype(np.float16)
        for sub in range(STRIDE // 8):                     # refresh the finger field every 8 substeps
            sim.ext = ext_for(sim.pos, g[f], finger[f])
            sim.step(8)
    return dict(pos=P, vel=V, g=g, finger=finger, stride=STRIDE, dt=sph.DT, spacing=sph.SPACING)


# ------------------------------------------------------------------------------------- data v2
# Longer trajectories that include the fluid coming to rest, and no control at all for part of the time.
#   active  random blocks, gravity and finger changing for the whole trajectory 
#   settle  controls for the first 35% of the trajectory, then gravity straight down and no finger: the fluid sloshes and calms
#   quiet   no finger and straight gravity throughout: blocks fall, a pool takes a drop, everything settles
DATA2 = os.path.join(HERE, "data2")
DURATION2 = 3.5


def pool_scene(rng):
    """A flat pool along the floor, usually with a small block dropped on it."""
    h = rng.uniform(0.06, 0.22)
    pos = [sph.lattice_block(0.0, 0.0, 1.0, h)]
    if rng.random() < 0.7:
        w, hh = rng.uniform(0.08, 0.25), rng.uniform(0.08, 0.2)
        x0, y0 = rng.uniform(0.0, 1.0 - w), rng.uniform(h + 0.05, max(h + 0.06, 0.9 - hh))
        pos.append(sph.lattice_block(x0, y0, w, hh))
    pos = torch.cat(pos)
    return pos, 0.05 * torch.randn_like(pos)


def schedule_for(rng, frames, kind):
    g, finger = control_schedule(rng, frames)
    down = sph.GRAVITY * np.array([0.0, -1.0], np.float32)
    if kind == "settle":
        cut = int(0.35 * frames)
        g[cut:], finger[cut:] = down, 0.0
    elif kind == "quiet":
        g[:], finger[:] = down, 0.0
    return g, finger


def simulate_v2(seed, kind):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if kind == "quiet" and rng.random() < 0.5:
        pos, vel = pool_scene(rng)
    else:
        pos, vel = random_scene(rng)
        if kind == "quiet":
            vel = 0.1 * torch.randn_like(vel)
    sim = sph.Sim(pos, vel)
    frames = int(DURATION2 / (STRIDE * sph.DT))
    g, finger = schedule_for(rng, frames, kind)
    P = np.zeros((frames, len(pos), 2), np.float32)
    V = np.zeros((frames, len(pos), 2), np.float16)
    for f in range(frames):
        P[f] = sim.pos.cpu().numpy()
        V[f] = sim.vel.cpu().numpy().astype(np.float16)
        for sub in range(STRIDE // 8):
            sim.ext = ext_for(sim.pos, g[f], finger[f])
            sim.step(8)
    return dict(pos=P, vel=V, g=g, finger=finger)


def main_v2():
    """python gen_data.py --v2   ->  data2/{split}_{i}_{pos,vel,g,finger}.npy (memory-mapped by learned.Data) + index.json"""
    import json
    os.makedirs(DATA2, exist_ok=True)
    plan = []
    for split, base, counts in (("train", 2000, (70, 45, 45)), ("val", 6000, (3, 3, 2)), ("test", 9500, (4, 4, 4))):
        i = 0
        for kind, c in zip(("active", "settle", "quiet"), counts):
            for _ in range(c):
                plan.append((split, i, base + i, kind)); i += 1
    ipath = os.path.join(DATA2, "index.json")
    index = json.load(open(ipath)) if os.path.exists(ipath) else {}
    t0 = time.time()
    for split, i, seed, kind in plan:
        name = f"{split}_{i:03d}"
        if name in index:
            continue
        d = simulate_v2(seed, kind)
        for k, v in d.items():
            np.save(os.path.join(DATA2, f"{name}_{k}.npy"), v)
        v = d["vel"].astype(np.float32)
        speed = np.linalg.norm(v, axis=-1).mean(1)
        index[name] = dict(kind=kind, seed=seed, n=int(d["pos"].shape[1]), frames=int(d["pos"].shape[0]),
                           speed_start=float(speed[:50].mean()), speed_end=float(speed[-200:].mean()))
        json.dump(index, open(ipath, "w"), indent=1)
        print(f"{name} {kind:6s} N={d['pos'].shape[1]} frames={d['pos'].shape[0]} end speed {index[name]['speed_end']:.3f} m/s  "
              f"elapsed {(time.time() - t0) / 60:.1f} min", flush=True)


def main():
    os.makedirs(DATA, exist_ok=True)
    plan = [("train", i, 1000 + i) for i in range(64)] + [("val", i, 5000 + i) for i in range(6)] \
         + [("test", i, 9000 + i) for i in range(6)] + [("big", i, 7000 + i) for i in range(4)]
    t0 = time.time()
    for split, i, seed in plan:
        path = os.path.join(DATA, f"{split}_{i:03d}.npz")
        if os.path.exists(path):
            continue
        d = simulate(seed, 1500, 2200) if split == "big" else simulate(seed)
        np.savez(path, **d)
        print(f"{split}_{i:03d}  N={d['pos'].shape[1]}  frames={d['pos'].shape[0]}  "
              f"elapsed {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    if "--v2" in sys.argv:
        main_v2()
    else:
        main()
