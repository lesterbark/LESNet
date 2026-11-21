"""Jet runs after the extended learning-rate grids (sweep_ext.py): the graph network at its new best rate, and a third seed for every model.

    run.bat -u rerun_jets.py     (resumable)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_toptag import run, RES  # noqa: E402

best = json.load(open(os.path.join(RES, "best_lr.json")))
best["gnn"] = 0.01
json.dump(best, open(os.path.join(RES, "best_lr.json"), "w"))
PLAN = [(1000, "gnn", (0, 1, 2)), (2000, "gnn", (0, 1, 2)), (1000, "transformer", (2,)), (2000, "transformer", (2,)),
        (1000, "deepsets", (2,)), (2000, "deepsets", (2,))]
if __name__ == "__main__":
    for steps, fam, seeds in PLAN:
        for seed in seeds:
            run(f"{fam}_n{steps}_s{seed}", ["--family", fam, "--lr", best[fam], "--steps", steps, "--seed", seed, "--test", "--eval_bs", 512])
    print("RERUN_JETS DONE", flush=True)
