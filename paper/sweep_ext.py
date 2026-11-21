"""Extended learning-rate grids for the jet baselines (400 steps, same protocol as run_toptag.py).

    run.bat -u sweep_ext.py      (resumable; one JSON per point in results/toptag_v2/)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_toptag import run, RES  # noqa: E402

EXTRA = {"deepsets": (3e-2,), "transformer": (1e-2, 3e-2), "gnn": (1e-2, 3e-2)}
GRID = {"les": (3e-3, 1e-2, 3e-2), "gnn": (1e-3, 3e-3), "transformer": (1e-3, 3e-3), "deepsets": (3e-3, 1e-2)}
for fam in EXTRA:
    GRID[fam] = tuple(GRID[fam]) + EXTRA[fam]
if __name__ == "__main__":
    out = {}
    for fam, lrs in EXTRA.items():
        for lr in lrs:
            print(f"[sweep_ext] {fam} lr={lr:g}", flush=True)
            run(f"lr_{fam}_{lr:g}", ["--family", fam, "--lr", lr, "--steps", 400, "--val_jets", 40000])
    for fam, lrs in GRID.items():
        out[fam] = {lr: json.load(open(os.path.join(RES, f"lr_{fam}_{lr:g}.json")))["hist"][-1]["val_auc"] for lr in lrs}
        print(fam, {f"{k:g}": round(v, 4) for k, v in out[fam].items()}, flush=True)
    json.dump({f: {str(k): v for k, v in d.items()} for f, d in out.items()}, open(os.path.join(RES, "lr_grid_ext.json"), "w"), indent=1)
    print("SWEEP_EXT DONE", flush=True)
