"""Roll the learned simulators out against held-out native trajectories and time them.

    run.bat -u evaluate.py --runs les_M2 gnn_M2 les_M4 gnn_M4 --horizon 0.5

For every test trajectory and several start frames: start from the native state, step the learned model
autoregressively for `horizon` simulated seconds under the same gravity / finger schedule, and compare
with the native frames.  Reported per model:

    pos_err[t]   mean per-particle position error, in particle spacings (identity-preserving, so it grows
                 as the flow diverges)
    chamfer[t]   permutation-free: mean nearest-neighbour distance between the two particle sets
    dens_err     mean |rho_learned - rho_native| / rho0 at the end, density from the native SPH estimator
    ke_ratio     kinetic energy of the learned rollout over the native one
    ms_step      wall time of one learned step;  speedup = native wall time / learned wall time for the
                 same simulated time
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import gen_data
import learned
import sph

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)


def load_model(name):
    s = torch.load(os.path.join(HERE, "runs", name, "ckpt.pt"), map_location=learned.DEV)
    a = s["args"]
    m = learned.LearnedSim(a["kind"], a["width"], a["depth"]).to(learned.DEV)
    m.load_state_dict(s["model"])
    m.eval()
    return m, a


def native_density(pos):
    sim = sph.Sim(pos.cpu())
    return sim.density().clone()


@torch.no_grad()
def timing(model, M, N=800, steps=60, reps=5):
    pos = sph.lattice_block(0.1, 0.0, 0.4, 0.8)[:N].to(learned.DEV)[None]
    vel = torch.zeros_like(pos)
    ext = torch.zeros_like(pos); ext[..., 1] = -9.81
    mask = torch.ones(1, pos.shape[1], dtype=torch.bool, device=learned.DEV)
    dt = M * gen_data.STRIDE * sph.DT
    best = 1e9
    for _ in range(reps):
        p, v, pr = pos.clone(), vel.clone(), vel.clone()
        for _w in range(5):
            p, v = model.step(p, v, pr, ext, mask, dt)
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(steps):
            p, v = model.step(p, v, pr, ext, mask, dt)
        torch.cuda.synchronize()
        best = min(best, (time.time() - t0) / steps)
    return best


def native_timing(N=800, sim_seconds=0.25, reps=3, method="cell"):
    sim = sph.Sim(sph.lattice_block(0.1, 0.0, 0.4, 0.8)[:N], method=method)
    n = int(sim_seconds / sph.DT)
    best = 1e9
    for _ in range(reps):
        sim.step(50)
        torch.cuda.synchronize(); t0 = time.time()
        sim.step(n)
        torch.cuda.synchronize()
        best = min(best, time.time() - t0)
    return best / sim_seconds          # wall seconds per simulated second


@torch.no_grad()
def rest_eval(names, data="data2", split="test", rest_s=3.0, ref_t0=2.3, ref_s=1.0):
    """Does the learned simulator keep a calm fluid calm?
    (1) calm-down: start from the native state at the last frame of every 'settle'/'quiet' test trajectory (speed ~0.05 m/s),
        no controls beyond gravity, roll out `rest_s` seconds; report the mean particle speed over time (the native fluid
        stays at ~0.05 m/s) and the mean height of the fluid (drift = sinking or rising).
    (2) from 'ref_t0' seconds in (nearly settled), roll `ref_s` seconds and compare with the native continuation (Chamfer, kinetic energy)."""
    frame_dt = gen_data.STRIDE * sph.DT
    test = learned.Data(split, data)
    out = {}
    for name in names:
        model, args = load_model(name)
        M = args["M"]
        steps_rest = int(rest_s / (M * frame_dt))
        steps_ref = int(ref_s / (M * frame_dt))
        sp, hy, nat_sp, ch, ke = [], [], [], [], []
        by_kind = {}
        for tr, kind in zip(test.trajs, test.kinds):
            if kind not in ("settle", "quiet"):
                continue
            T = tr["pos"].shape[0]
            P, V = learned.rollout(model, tr, T - 2, steps_rest, M)
            sp.append(V.norm(dim=-1).mean(1).cpu().numpy())
            by_kind.setdefault(kind, []).append(sp[-1])
            hy.append((P[..., 1].mean(1) - P[0, :, 1].mean()).cpu().numpy())
            nat_sp.append(float(tr["vel"][T - 2].norm(dim=-1).mean()))
            t0 = int(ref_t0 / frame_dt)
            if t0 + steps_ref * M < T - 1:
                P, V = learned.rollout(model, tr, t0, steps_ref, M)
                idx = [t0 + i * M for i in range(steps_ref + 1)]
                ref, refv = tr["pos"][idx].to(learned.DEV), tr["vel"][idx].to(learned.DEV)
                ch.append(learned.chamfer(P[-1], ref[-1]).item() / sph.SPACING)
                ke.append(((V[-1] ** 2).sum() / (refv[-1] ** 2).sum().clamp_min(1e-9)).item())
        sp, hy = np.mean(sp, 0), np.mean(hy, 0)
        t = np.arange(len(sp)) * M * frame_dt
        out[name] = dict(M=M, native_speed=float(np.mean(nat_sp)), t=t[::max(1, len(t) // 60)].tolist(),
                         speed=sp[::max(1, len(t) // 60)].tolist(), height_drift=hy[::max(1, len(t) // 60)].tolist(),
                         speed_late=float(sp[-max(3, len(sp) // 4):].mean()), height_drift_end=float(hy[-1]),
                         chamfer_ref=float(np.mean(ch)) if ch else None, ke_ref=float(np.median(ke)) if ke else None,
                         n_traj=len(nat_sp),
                         speed_late_by_kind={k: float(np.mean([x[-max(3, len(x) // 4):].mean() for x in v])) for k, v in by_kind.items()})
        o = out[name]
        print(f"{name:16s} rest: speed late {o['speed_late']:.3f} m/s (native {o['native_speed']:.3f})  height drift {o['height_drift_end'] * 1000:+.1f} mm  "
              f"chamfer@1s {o['chamfer_ref']:.2f} sp  KE ratio {o['ke_ref']:.2f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="data2 for the v2 set")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--horizon", type=float, default=0.5)
    ap.add_argument("--starts", type=int, default=4)
    ap.add_argument("--out", default="fluid_eval.json")
    ap.add_argument("--split", default="test", help="test (scenes like training) or big (more particles than any training scene)")
    ap.add_argument("--mode", default="all", choices=["all", "accuracy", "timing", "rest"],
                    help="accuracy can run while other jobs use the GPU; timing needs an idle GPU; rest = calm-down test (needs --data data2)")
    a = ap.parse_args()
    frame_dt = gen_data.STRIDE * sph.DT
    if a.mode == "rest":
        res = rest_eval(a.runs, a.data or "data2", a.split)
        json.dump(res, open(os.path.join(OUT, a.out), "w"), indent=1)
        return
    tpath = os.path.join(OUT, "fluid_timing.json")
    timing_res = json.load(open(tpath)) if os.path.exists(tpath) else {"native_wall_per_sim_s": None, "runs": {}}
    if a.mode in ("all", "timing"):
        timing_res["native_wall_per_sim_s"] = native_timing()                       # cell list: the speed baseline
        timing_res["native_brute_wall_per_sim_s"] = native_timing(method="brute")   # all pairs: the reference used so far
        print(f"native: {timing_res['native_wall_per_sim_s']:.2f} wall-s per sim-s with a cell list, "
              f"{timing_res['native_brute_wall_per_sim_s']:.2f} all-pairs (N=800)", flush=True)
        for name in a.runs:
            model, args = load_model(name)
            t_step = timing(model, args["M"])
            sim_per_step = args["M"] * frame_dt
            timing_res["runs"][name] = dict(ms_step=t_step * 1000, wall_per_sim_s=t_step / sim_per_step)
            print(f"{name:12s} {t_step * 1000:.2f} ms/step  {t_step / sim_per_step:.2f} wall-s per sim-s", flush=True)
        json.dump(timing_res, open(tpath, "w"), indent=1)
    if a.mode == "timing":
        return
    test = learned.Data(a.split, a.data)
    results = {"native_wall_per_sim_s": timing_res["native_wall_per_sim_s"], "runs": {}, "split": a.split,
               "n_particles": [int(t["pos"].shape[1]) for t in test.trajs]}
    for name in a.runs:
        model, args = load_model(name)
        M = args["M"]
        steps = int(a.horizon / (M * frame_dt))
        pe, ch, de, ke, ids = [], [], [], [], []
        for ti, tr in enumerate(test.trajs):
            T = tr["pos"].shape[0]
            for t0 in np.linspace(5, T - 2 - steps * M, a.starts).astype(int):
                P, V = learned.rollout(model, tr, int(t0), steps, M)
                ref_idx = [int(t0) + i * M for i in range(steps + 1)]
                ref = tr["pos"][ref_idx].to(learned.DEV)
                refv = tr["vel"][ref_idx].to(learned.DEV)
                pe.append(((P - ref).norm(dim=-1).mean(1) / sph.SPACING).cpu().numpy())
                ch.append(np.array([learned.chamfer(P[i], ref[i]).item() / sph.SPACING for i in range(steps + 1)]))
                rl = native_density(P[-1]); rn = native_density(ref[-1])
                de.append(((rl - rn).abs().mean() / sph.RHO0).item())
                ke.append(((V[-1] ** 2).sum() / (refv[-1] ** 2).sum().clamp_min(1e-9)).item())
                ids.append((ti, int(t0)))
        sim_per_step = M * frame_dt
        r = dict(M=M, kind=args["kind"], steps=steps, dt_ms=sim_per_step * 1000,
                 pos_err=np.mean(pe, 0).tolist(), chamfer=np.mean(ch, 0).tolist(),
                 pos_err_end=float(np.mean([p[-1] for p in pe])), chamfer_end=float(np.mean([c[-1] for c in ch])),
                 chamfer_end_sd=float(np.std([c[-1] for c in ch])),
                 dens_err=float(np.mean(de)), ke_ratio=float(np.median(ke)),
                 params=learned.count_params(model), n_rollouts=len(pe),
                 rollouts=dict(traj=[i[0] for i in ids], t0=[i[1] for i in ids], chamfer_end=[float(c[-1]) for c in ch],
                               pos_err_end=[float(p[-1]) for p in pe], dens_err=[float(x) for x in de]))
        t = timing_res["runs"].get(name)
        if t and timing_res["native_wall_per_sim_s"]:
            r.update(ms_step=t["ms_step"], wall_per_sim_s=t["wall_per_sim_s"],
                     speedup_vs_native=timing_res["native_wall_per_sim_s"] / t["wall_per_sim_s"])
        results["runs"][name] = r
        print(f"{name:12s} dt={r['dt_ms']:.1f}ms  pos_err@end {r['pos_err_end']:.2f} sp  chamfer@end {r['chamfer_end']:.2f} sp  "
              f"dens {r['dens_err']:.3f}  KE {r['ke_ratio']:.2f}", flush=True)
    json.dump(results, open(os.path.join(OUT, a.out), "w"), indent=1)


if __name__ == "__main__":
    main()