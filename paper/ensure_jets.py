"""Fill in missing seeds for the 1000- and 2000-step jet runs.

    run.bat -u ensure_jets.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_toptag import run, RES  # noqa: E402

best = json.load(open(os.path.join(RES, "best_lr.json")))
if __name__ == "__main__":
    for steps in (1000, 2000):
        for fam in ("deepsets", "transformer", "gnn", "les"):
            for seed in (0, 1, 2):
                run(f"{fam}_n{steps}_s{seed}", ["--family", fam, "--lr", best[fam], "--steps", steps, "--seed", seed, "--test", "--eval_bs", 512])
    print("ENSURE_JETS DONE", flush=True)
