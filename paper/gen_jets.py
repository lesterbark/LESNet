"""GENConv-style softmax-aggregation baseline on the jet benchmark: learning-rate grid (400 steps), then 1000 and 2000 steps with three seeds.

    run.bat -u gen_jets.py     (resumable)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_toptag import run, RES  # noqa: E402

if __name__ == "__main__":
    grid = [1e-3, 3e-3, 1e-2, 3e-2]
    res = {}
    while True:
        for lr in grid:
            if lr not in res:
                res[lr] = run(f"lr_gen_{lr:g}", ["--family", "gen", "--lr", lr, "--steps", 400, "--val_jets", 40000, "--eval_bs", 512])["hist"][-1]["val_auc"]
        best = max(res, key=res.get)
        print("gen lr sweep:", {f"{k:g}": round(v, 4) for k, v in res.items()}, "->", f"{best:g}", flush=True)
        if best == max(res) and best < 0.1:
            grid = grid + [best * 3.0 if abs(best * 3.0 - 3e-2 * 3) > 1e-9 else 1e-1]
            continue
        break
    b = json.load(open(os.path.join(RES, "best_lr.json")))
    b["gen"] = best
    json.dump(b, open(os.path.join(RES, "best_lr.json"), "w"))
    for steps in (1000, 2000):
        for seed in (0, 1, 2):
            run(f"gen_n{steps}_s{seed}", ["--family", "gen", "--lr", best, "--steps", steps, "--seed", seed, "--test", "--eval_bs", 512])
    print("GEN_JETS DONE", flush=True)
