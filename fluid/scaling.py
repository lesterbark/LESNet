"""Speed vs particle count: native SPH against the learned simulators.

    run.bat -u scaling.py --runs les_M2 gnn_M2 les_M4 gnn_M4     ->  out/fluid_scaling.json

Wall-clock seconds to advance ONE SIMULATED SECOND, at N = 500 ... 4000 particles (a jittered lattice
filling the unit box; timing does not depend on the flow).  Run on an otherwise idle GPU.  The learned
simulators are timed for a full step: neighbour search, pair features, network, integration.
"""
import argparse
import json
import os
import time

import torch

import evaluate
import gen_data
import learned
import sph

HERE = os.path.dirname(os.path.abspath(__file__))


def dense_block(N, device="cuda"):
    """N particles as a full-width block at the solver's design spacing (a physically sensible state; N <= ~2400 fit in the box)."""
    h = N * sph.SPACING ** 2 / 1.0
    pos = sph.lattice_block(0.0, 0.0, 1.0, h)
    return pos[:N].to(device).float().contiguous() if len(pos) >= N else pos.to(device).float().contiguous()


def lattice(N, device="cuda"):
    n = int(N ** 0.5)
    xs = (torch.arange(n, device=device) + 0.5) / n
    g = torch.stack(torch.meshgrid(xs, xs, indexing="xy"), -1).reshape(-1, 2)
    g = g + 0.05 / n * torch.randn_like(g)
    return g.clamp(0.001, 0.999).float().contiguous()


@torch.no_grad()
def learned_time(model, M, pos, steps=30, reps=3):
    N = len(pos)
    p = pos[None].clone()
    v = torch.zeros_like(p)
    ext = torch.zeros_like(p); ext[..., 1] = -9.81
    mask = torch.ones(1, N, dtype=torch.bool, device=pos.device)
    dt = M * gen_data.STRIDE * sph.DT
    best = 1e9
    for _ in range(reps):
        q, w, pr = p.clone(), v.clone(), v.clone()
        for _i in range(3):
            q, w = model.step(q, w, pr, ext, mask, dt)
        torch.cuda.synchronize(); t0 = time.time()
        for _i in range(steps):
            q, w = model.step(q, w, pr, ext, mask, dt)
        torch.cuda.synchronize()
        best = min(best, (time.time() - t0) / steps)
    return best / dt            # wall seconds per simulated second


def native_time(pos, sim_seconds=0.05, reps=3, method="cell"):
    sim = sph.Sim(pos.cpu(), method=method)
    n = int(sim_seconds / sph.DT)
    best = 1e9
    for _ in range(reps):
        sim.step(20)
        torch.cuda.synchronize(); t0 = time.time()
        sim.step(n)
        torch.cuda.synchronize()
        best = min(best, time.time() - t0)
    return best / sim_seconds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--N", nargs="+", type=int, default=[500, 1000, 2000, 3000, 4000])
    ap.add_argument("--out", default="fluid_scaling.json")
    ap.add_argument("--brute_max", type=int, default=8000, help="largest N at which the all-pairs reference is timed")
    a = ap.parse_args()
    models = {n: evaluate.load_model(n) for n in a.runs}
    out = {"N": a.N, "native": [], "native_brute": [], "runs": {n: [] for n in a.runs}}
    for N in a.N:
        if N <= 2400:                                            # a physical state: a block at the design spacing
            pos = dense_block(N)
            nat = native_time(pos)                               # cell list: the speed baseline
            nb = native_time(pos, method="brute")
        else:                                                    # beyond what fits in the box: only the learned models are timed
            pos = lattice(N)
            nat = nb = None
        out["native"].append(nat)
        out["native_brute"].append(nb)
        row = f"N={N:5d} native(cell) {nat or float('nan'):7.2f} brute {nb or float('nan'):7.2f} s/sim-s |"
        for n, (m, args) in models.items():
            t = learned_time(m, args["M"], pos)
            out["runs"][n].append(t)
            row += f" {n} {t:6.2f} (x{(nat or float('nan')) / t:.1f})"
        print(row, flush=True)
    json.dump(out, open(os.path.join(HERE, "out", a.out), "w"), indent=1)


if __name__ == "__main__":
    main()
