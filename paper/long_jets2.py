"""Longer jet runs for the families without one yet (one seed each).

    run.bat -u long_jets2.py     (resumable)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_toptag import run, RES  # noqa: E402

best = json.load(open(os.path.join(RES, "best_lr.json")))
PLAN = [(11800, "gen"), (23600, "deepsets"), (23600, "gnn")]
if __name__ == "__main__":
    for steps, fam in PLAN:
        run(f"{fam}_n{steps}_s0", ["--family", fam, "--lr", best[fam], "--steps", steps, "--seed", 0, "--test", "--eval_bs", 512])
    print("LONG_JETS2 DONE", flush=True)
