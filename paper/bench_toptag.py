"""Training-step time of each jet-tagging model on an otherwise idle GPU (forward + backward + optimiser, batch 512,
100 constituents, ~150k parameters), and peak memory.

    run.bat -u bench_toptag.py     ->  results/bench_toptag.json
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import toptag as tt  # noqa: E402


def bench(family, reps=3, steps=20, B=512):
    width = tt.fit_width(family, 3, 150_000)
    net = tt.build(family, width, 3)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    split = tt.Split("val")
    tt.fit_norm(split)
    X, y = split.batch(np.arange(B))
    f, c, m = tt.featurize(X.to("cuda"))
    y = y.to("cuda")
    best = 1e9
    torch.cuda.reset_peak_memory_stats()
    for _ in range(reps):
        for _w in range(3):
            loss = F.binary_cross_entropy_with_logits(net(f, c, m), y); opt.zero_grad(); loss.backward(); opt.step()
        torch.cuda.synchronize(); t0 = time.time()
        for _i in range(steps):
            loss = F.binary_cross_entropy_with_logits(net(f, c, m), y); opt.zero_grad(); loss.backward(); opt.step()
        torch.cuda.synchronize()
        best = min(best, (time.time() - t0) / steps)
    return dict(family=family, width=width, params=tt.n_params(net), ms_per_step=best * 1000,
                peak_mem_mb=torch.cuda.max_memory_allocated() / 2 ** 20)


if __name__ == "__main__":
    out = []
    for fam in ("les", "les_cat", "deepsets", "transformer", "gnn", "gen"):
        r = bench(fam)
        out.append(r)
        print(f"{fam:12s} width {r['width']:4d}  {r['params']:>8,} params  {r['ms_per_step']:7.1f} ms/step  peak {r['peak_mem_mb']:6.0f} MB", flush=True)
    json.dump(out, open(os.path.join(HERE, "results", "bench_toptag.json"), "w"), indent=1)
