"""Longer jet runs (one seed each) at each family's learning rate.

    run.bat -u long_jets.py     (resumable)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_toptag import run, RES  # noqa: E402

best = json.load(open(os.path.join(RES, "best_lr.json")))
PLAN = [(6400, "les"), (11800, "gnn"), (11800, "deepsets"), (23600, "les"), (23600, "transformer")]
if __name__ == "__main__":
    for steps in (1000, 2000):
        for fam in ("deepsets", "transformer", "gnn", "les"):
            for seed in (0, 1, 2):
                run(f"{fam}_n{steps}_s{seed}", ["--family", fam, "--lr", best[fam], "--steps", steps, "--seed", seed, "--test", "--eval_bs", 512])
    for steps, fam in PLAN:
        run(f"{fam}_n{steps}_s0", ["--family", fam, "--lr", best[fam], "--steps", steps, "--seed", 0, "--test", "--eval_bs", 512])
    print("LONG_JETS DONE", flush=True)
