"""The v2 fluid training campaign, one or more foreground windows:  python -u run_v2.py   (open it twice for two lanes)

  1. learning-rate sweep (5000 steps each, M=4, data2 + mirroring) for LESNet and the graph net; the best validation
     loss per family decides the learning rate of the long runs,
  2. the long runs (STEPS_MAIN each, same data stream and schedule): les_M4_d2, lescat_M4_d2 (inputs re-concatenated at
     every layer), gnn72_M4_d2, les256_M4_d2 (LESNet widened to the graph net's step cost),
  3. (--m2) the same pair at M=2.
Several windows can run it at once: a run is claimed by creating runs/<name>/claimed, so no run is done twice.
Resumable: finished runs (train_log.json present) are skipped.  Delete a stale `claimed` file to re-run an interrupted one
(its checkpoint is picked up).
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
STEPS_SWEEP, STEPS_MAIN = 5000, int(os.environ.get("STEPS_MAIN", 40000))
FAMS = (("les", "les", 128), ("gnn", "gnn", 72))
LRS = (1e-3, 3e-3, 1e-2)


def done(name):
    return os.path.exists(os.path.join(RUNS, name, "train_log.json"))


def claim(name):
    os.makedirs(os.path.join(RUNS, name), exist_ok=True)
    try:
        os.close(os.open(os.path.join(RUNS, name, "claimed"), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


def train(name, kind, width, lr, steps, M=4):
    cmd = [sys.executable, "-u", "train_sim.py", "--kind", kind, "--M", str(M), "--width", str(width), "--lr", str(lr),
           "--steps", str(steps), "--data", "data2", "--mirror", "--name", name]
    print("\n>>>", " ".join(cmd[2:]), flush=True)
    subprocess.run(cmd, cwd=HERE, check=True)


def final_val(name):
    return json.load(open(os.path.join(RUNS, name, "train_log.json")))["hist"][-1][2]


def best_lr(fam):
    res = {lr: final_val(f"sweep_{fam}_lr{lr:g}") for lr in LRS}
    b = min(res, key=res.get)
    json.dump(dict(fam=fam, val=res, best=b), open(os.path.join(HERE, "out", f"lr_sweep_v2_{fam}.json"), "w"))
    print(f"\n== lr sweep {fam}: " + "  ".join(f"{k:g}: {v:.4f}" for k, v in res.items()) + f"  -> {b:g}", flush=True)
    return b


def main():
    sweeps = [(f"sweep_{fam}_lr{lr:g}", kind, w, lr) for fam, kind, w in FAMS for lr in LRS]
    for name, kind, w, lr in sweeps:                              # 1. sweeps (any lane picks up whatever is unclaimed)
        if not done(name) and claim(name):
            train(name, kind, w, lr, STEPS_SWEEP)
    while not all(done(n) for n, *_ in sweeps):                    # another lane (or a leftover run) is still sweeping
        time.sleep(15)
    best = {fam: best_lr(fam) for fam, *_ in FAMS}
    mains = []
    for M in ((4, 2) if "--m2" in sys.argv else (4,)):
        mains += [(f"les_M{M}_d2", "les", 128, "les", M), (f"gnn72_M{M}_d2", "gnn", 72, "gnn", M)]
        if M == 4:
            mains += [(f"lescat_M{M}_d2", "les_cat", 128, "les", M), (f"les256_M{M}_d2", "les", 256, "les", M)]
    for name, kind, w, fam, M in mains:                            # 2. long runs
        if not done(name) and claim(name):
            train(name, kind, w, best[fam], STEPS_MAIN, M)
    print("\nthis lane is finished (other lanes may still be running)", flush=True)


if __name__ == "__main__":
    main()
