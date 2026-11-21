"""The top-tagging campaign in one foreground window:  python -u run_toptag.py

  1. learning-rate sweep (400 steps, batch 512) per family; best validation AUC picks the rate
  2. main runs with the same number of optimiser steps for every family, full test split at the end:
        1000 steps:  les (3 seeds), les_cat, gnn, transformer, deepsets (1 seed each)
        2000 steps:  les (3 seeds), les_cat, transformer, deepsets   (the graph net costs ~10x a LESNet step on this card, so its
                     2000-step run is optional: --gnn2000)
  3. (--isotime) an iso-time comparison: LESNet given the wall-clock time the graph net needed
Resumable (a finished run leaves results/toptag_v2/<name>.json).
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "toptag_v2")
LRS = {"les": (3e-3, 1e-2, 3e-2), "gnn": (1e-3, 3e-3), "transformer": (1e-3, 3e-3), "deepsets": (3e-3, 1e-2)}


def run(name, args):
    if os.path.exists(os.path.join(RES, name + ".json")):
        return json.load(open(os.path.join(RES, name + ".json")))
    cmd = [sys.executable, "-u", "toptag.py", "--name", name] + [str(x) for x in args]
    print("\n>>>", " ".join(cmd[2:]), flush=True)
    subprocess.run(cmd, cwd=HERE, check=True)
    return json.load(open(os.path.join(RES, name + ".json")))


def main():
    best = {}
    for fam, lrs in LRS.items():
        res = {}
        for lr in lrs:
            r = run(f"lr_{fam}_{lr:g}", ["--family", fam, "--lr", lr, "--steps", 400, "--val_jets", 40000])
            res[lr] = r["hist"][-1]["val_auc"]
        best[fam] = max(res, key=res.get)
        print(f"\n== lr sweep {fam}: " + "  ".join(f"{k:g}: {v:.4f}" for k, v in res.items()) + f"  -> {best[fam]:g}", flush=True)
    best["les_cat"] = best["les"]
    json.dump(best, open(os.path.join(RES, "best_lr.json"), "w"))
    plan = [(1000, "les", 3), (1000, "gnn", 1), (1000, "transformer", 1), (1000, "deepsets", 1), (1000, "les_cat", 1),
            (2000, "les", 3), (2000, "transformer", 1), (2000, "deepsets", 1), (2000, "les_cat", 1)]
    if "--gnn2000" in sys.argv:
        plan.append((2000, "gnn", 1))
    for steps, fam, seeds in plan:
        for seed in range(seeds):
            run(f"{fam}_n{steps}_s{seed}", ["--family", fam, "--lr", best[fam], "--steps", steps, "--seed", seed, "--test"])
    if "--isotime" in sys.argv:
        g = json.load(open(os.path.join(RES, "gnn_n1000_s0.json")))
        run("les_isotime_s0", ["--family", "les", "--lr", best["les"], "--budget_s", g["train_s"], "--seed", 0, "--test"])
    print("\nALL TOPTAG RUNS DONE", flush=True)


if __name__ == "__main__":
    main()
