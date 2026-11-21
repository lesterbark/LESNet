"""Evaluate the v2 fluid campaign in one foreground window:  python -u eval_v2.py [--timing]

  accuracy   0.5 s rollouts on the 12 held-out data2 trajectories (4 active, 4 settle, 4 quiet), 6 start frames each
  rest       the calm-down test (does a settled fluid stay settled?)
  timing     idle-GPU speed against both native baselines (all-pairs and cell list); run with --timing only when nothing else uses the GPU
  scaling    seconds per simulated second from 500 to 8000 particles
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OLD = ["les_M4", "lescat_M4_old", "gnn72_M4", "les_M2", "gnn72_M2"]
OLD = [n for n in ("les_M4", "les_cat_M4", "gnn72_M4", "les_M2", "les_cat_M2", "gnn72_M2")]
NEW = ["les_M4_d2", "lescat_M4_d2", "gnn72_M4_d2", "les256_M4_d2", "les_M2_d2", "gnn72_M2_d2"]


def run(args):
    print("\n>>> evaluate", " ".join(args), flush=True)
    subprocess.run([sys.executable, "-u", "evaluate.py"] + args, cwd=HERE, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timing", action="store_true")
    a = ap.parse_args()
    runs = [n for n in OLD + NEW if os.path.exists(os.path.join(HERE, "runs", n, "ckpt.pt"))]
    run(["--runs"] + runs + ["--mode", "accuracy", "--data", "data2", "--starts", "6", "--out", "fluid_eval_v2.json"])
    run(["--runs"] + runs + ["--mode", "rest", "--data", "data2", "--out", "fluid_rest_v2.json"])
    # scenes with more particles than any training scene (1600-2200)
    run(["--runs"] + runs + ["--mode", "accuracy", "--split", "big", "--out", "fluid_eval_big_v2.json"])
    if a.timing:
        run(["--runs"] + [n for n in NEW if n in runs] + ["--mode", "timing"])
        print("\n>>> scaling", flush=True)
        subprocess.run([sys.executable, "-u", "scaling.py", "--runs", "les_M4_d2", "gnn72_M4_d2", "les256_M4_d2", "les_M8", "gnn72_M8",
                        "--N", "500", "1000", "2000", "4000", "8000", "--out", "fluid_scaling_v2.json"], cwd=HERE, check=True)
    print("\nEVAL DONE", flush=True)


if __name__ == "__main__":
    main()
