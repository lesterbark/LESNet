"""Seeds 1 and 2 of the 11 800-step jet runs (seed 0 exists for every family).

    run.bat -u long_jets3.py     (resumable)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_toptag import run, RES  # noqa: E402

best = json.load(open(os.path.join(RES, "best_lr.json")))
PLAN = [("les", 1), ("les", 2), ("gnn", 1), ("gnn", 2), ("deepsets", 1), ("deepsets", 2), ("transformer", 1), ("transformer", 2)]
if __name__ == "__main__":
    for fam, seed in PLAN:
        run(f"{fam}_n11800_s{seed}", ["--family", fam, "--lr", best[fam], "--steps", 11800, "--seed", seed, "--test", "--eval_bs", 512])
    print("LONG_JETS3 DONE", flush=True)
