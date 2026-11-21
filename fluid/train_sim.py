"""Train a learned simulator on the native-SPH trajectories.

    run.bat -u train_sim.py --kind les --M 2 --minutes 40 --name les_M2

Resumable: a checkpoint is written every few minutes and picked up again on restart.  The schedule is
set from a wall-clock budget (after 30 steps it measures the step time and fixes the total step count),
so every model gets the same training time.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

import gen_data
import learned
import sph

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="les")
    ap.add_argument("--M", type=int, default=2, help="frames per learned step")
    ap.add_argument("--minutes", type=float, default=40)
    ap.add_argument("--steps", type=int, default=0, help="fixed step count (overrides --minutes)")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--noise", type=float, default=0.5, help="velocity noise as a fraction of one step's velocity change")
    ap.add_argument("--name", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default=None, help="data2 for the v2 set (longer, with rest phases); default is the original data/")
    ap.add_argument("--mirror", action="store_true", help="reflect each training sample left-right with probability 1/2")
    a = ap.parse_args()
    name = a.name or f"{a.kind}_M{a.M}"
    run = os.path.join(HERE, "runs", name)
    os.makedirs(run, exist_ok=True)
    ck = os.path.join(run, "ckpt.pt")

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    train = learned.Data("train", a.data, mirror=a.mirror)
    val = learned.Data("val", a.data)
    model = learned.LearnedSim(a.kind, a.width, a.depth).to(learned.DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    dt = a.M * gen_data.STRIDE * sph.DT
    step, total, elapsed, hist = 0, (a.steps or None), 0.0, []
    if os.path.exists(ck):
        s = torch.load(ck, map_location=learned.DEV)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"])
        step, total, elapsed, hist = s["step"], s["total"], s["elapsed"], s["hist"]
        print(f"resumed {name} at step {step}/{total}", flush=True)
    print(f"{name}: {learned.count_params(model)} params, {len(train.trajs)} train trajectories, dt={dt*1000:.2f} ms", flush=True)

    # velocity noise level: a fraction of the typical per-step velocity change
    vnoise = a.noise * 9.81 * dt
    budget = a.minutes * 60
    last_ck = time.time()
    run_loss = None
    while True:
        t0 = time.time()
        if total is not None and step >= total:
            break
        if total is None and step == 30:
            total = int(30 * budget / max(elapsed, 1e-3))
            print(f"step time {1000*elapsed/30:.1f} ms -> {total} steps for {a.minutes} min", flush=True)
        frac = 0.0 if total is None else step / total
        lr = a.lr * (min(1.0, (step + 1) / 200)) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * frac)))
        for gp in opt.param_groups:
            gp["lr"] = lr
        b, mask, g, fin = train.sample(a.batch, a.M, rng)
        ext = learned.ext_batch(b["pos"], g, fin)
        nv = torch.randn_like(b["vel"]) * vnoise
        vel_n = b["vel"] + nv
        prev_n = b["prev"] + nv
        pos_n = b["pos"] + torch.randn_like(b["pos"]) * vnoise * dt
        acc = model(pos_n, vel_n, prev_n, ext, mask)
        target = (b["next"] - vel_n) / dt
        w = mask[..., None].float()
        loss = (((acc - target) / learned.A_SCALE) ** 2 * w).sum() / (w.sum() * 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        torch.cuda.synchronize()
        elapsed += time.time() - t0
        step += 1
        lv = loss.item() if step % 20 == 0 else None
        if lv is not None:
            run_loss = lv if run_loss is None else 0.95 * run_loss + 0.05 * lv
        if step % 100 == 0 and step % 500 != 0:
            eta = (total - step) * elapsed / step / 60 if total else float("nan")
            print(f"step {step}/{total}  train {run_loss:.5f}  {1000 * elapsed / step:.1f} ms/step  eta {eta:.1f} min", flush=True)
        if step % 500 == 0:
            with torch.no_grad():
                vb, vm, vg, vf = val.sample(8, a.M, np.random.default_rng(1))
                vext = learned.ext_batch(vb["pos"], vg, vf)
                vacc = model(vb["pos"], vb["vel"], vb["prev"], vext, vm)
                vt = (vb["next"] - vb["vel"]) / dt
                vw = vm[..., None].float()
                vloss = (((vacc - vt) / learned.A_SCALE) ** 2 * vw).sum().item() / (vw.sum().item() * 2)
            hist.append((step, run_loss, vloss))
            print(f"step {step}/{total}  train {run_loss:.5f}  val {vloss:.5f}  lr {lr:.2e}  {elapsed/60:.1f} min  eta {(total - step) * elapsed / step / 60:.1f} min   <- validation", flush=True)
        if time.time() - last_ck > 300 or (total is not None and step >= total):
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, total=total,
                            elapsed=elapsed, hist=hist, args=vars(a)), ck)
            last_ck = time.time()
    torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, total=total,
                    elapsed=elapsed, hist=hist, args=vars(a)), ck)
    json.dump(dict(args=vars(a), params=learned.count_params(model), steps=step, minutes=elapsed / 60, hist=hist),
              open(os.path.join(run, "train_log.json"), "w"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
