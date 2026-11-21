"""Training-step time and peak memory of every benchmark model on an otherwise idle GPU:  run.bat -u bench\timing.py  ->  results/bench_timing.json"""
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import lib  # noqa: E402
import cifar, pointset, charlm  # noqa: E402


def bench(model, fn, reps=3, steps=15):
    model = model.to(lib.DEV)
    opt = torch.optim.AdamW(model.parameters(), 1e-3)
    best = 1e9
    torch.cuda.reset_peak_memory_stats()
    for _ in range(reps):
        for _w in range(3):
            l = fn(model); opt.zero_grad(); l.backward(); opt.step()
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(steps):
            l = fn(model); opt.zero_grad(); l.backward(); opt.step()
        torch.cuda.synchronize()
        best = min(best, (time.time() - t0) / steps)
    return dict(params=lib.n_params(model), ms_per_step=best * 1000, peak_mem_mb=torch.cuda.max_memory_allocated() / 2 ** 20)


def main():
    out = {}
    x = torch.randn(128, 3, 32, 32, device=lib.DEV); y = torch.randint(0, 10, (128,), device=lib.DEV)
    out["cifar"] = {"les": bench(cifar.LESImage(48), lambda m: F.cross_entropy(m(x), y)),
                    "cnn": bench(cifar.CNN(lib.fit_width(lambda w: cifar.CNN(w), lib.n_params(cifar.LESImage(48)), lo=4, hi=256, step=2)), lambda m: F.cross_entropy(m(x), y))}
    (xt, yt), _ = pointset.load()
    g = torch.Generator(device=lib.DEV).manual_seed(0)
    px, pc, pm_ = pointset.to_points(xt[:128], g)
    out["pointset"] = {}
    for fam in ("les", "gnn", "transformer", "deepsets"):
        w = lib.fit_width(lambda w_: pointset.build(fam, w_), 150_000, lo=8, hi=512, step=8 if fam == "transformer" else 4)
        out["pointset"][fam] = bench(pointset.build(fam, w), lambda m: F.cross_entropy(m(px, pc, pm_), yt[:128]))
    tr, va, V = charlm.load()
    xl = tr[:charlm.B * charlm.T].view(charlm.B, charlm.T); yl = tr[1:charlm.B * charlm.T + 1].view(charlm.B, charlm.T)
    lw = lib.fit_width(lambda w: charlm.LESLM(V, w, 6), 800_000, lo=16, hi=512, step=4)
    gd = lib.fit_width(lambda w: charlm.GPT(V, w, 6), 800_000, lo=16, hi=512, step=4)
    out["charlm"] = {"les": bench(charlm.LESLM(V, lw, 6), lambda m: F.cross_entropy(m(xl).reshape(-1, V), yl.reshape(-1))),
                     "gpt": bench(charlm.GPT(V, gd, 6), lambda m: F.cross_entropy(m(xl).reshape(-1, V), yl.reshape(-1)))}
    for t, d in out.items():
        for f, r in d.items():
            print(f"{t:9s} {f:12s} {r['params']:>9,} params {r['ms_per_step']:7.1f} ms/step {r['peak_mem_mb']:7.0f} MB", flush=True)
    json.dump(out, open(os.path.join(lib.RES, "..", "bench_timing.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
