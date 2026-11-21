"""Transformer jet run of 23 600 steps (one seed).

    run.bat -u tf_long.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_toptag import run, RES  # noqa: E402

best = json.load(open(os.path.join(RES, "best_lr.json")))
run("transformer_n23600_s0", ["--family", "transformer", "--lr", best["transformer"], "--steps", 23600, "--seed", 0, "--test", "--eval_bs", 512])
print("TF_LONG DONE", flush=True)
