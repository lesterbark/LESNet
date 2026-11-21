"""Fluid figures for the paper.

    run.bat -u figures.py --runs les_M2 gnn_M2 --eval fluid_eval.json

    fluid_strips.png        native vs learned rollouts at a few times, same start, same controls
    fluid_error.png         rollout error vs simulated time, and the speed / accuracy trade-off
    fluid_training.png      validation loss during training
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import evaluate
import gen_data
import learned
import sph

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
PAPER_FIG = os.path.join(HERE, "..", "paper", "figures")
os.makedirs(PAPER_FIG, exist_ok=True)
INK = "#1d2330"
COL = {"les": "#d1495b", "les_cat": "#e9a03b", "gnn": "#2e86ab"}
LABEL = {"les_M4_d2": "LESNet, 105k", "gnn72_M4_d2": "graph network, 107k", "les256_M4_d2": "LESNet, 407k"}


def strips(runs, test_idx=1, t0=60, horizon=0.45, times=(0.1, 0.2, 0.3, 0.45), data=None, tag=""):
    test = learned.Data("test", data)
    tr = test.trajs[test_idx]
    frame_dt = gen_data.STRIDE * sph.DT
    models = [(n, *evaluate.load_model(n)) for n in runs]
    rows = [("native SPH", None)]
    rollouts = {}
    for n, m, a in models:
        steps = int(horizon / (a["M"] * frame_dt))
        P, V = learned.rollout(m, tr, t0, steps, a["M"])
        rollouts[n] = (P.cpu().numpy(), V.cpu().numpy(), a["M"])
        rows.append((LABEL.get(n, f"{n}\n{a['kind'].upper()}, {sum(p.numel() for p in m.parameters()) // 1000}k"), n))
    fig, axs = plt.subplots(len(rows), len(times), figsize=(1.55 * len(times) + 0.5, 1.6 * len(rows)))
    for r, (title, name) in enumerate(rows):
        for c, t in enumerate(times):
            ax = axs[r, c]
            if name is None:
                f = t0 + int(round(t / frame_dt))
                pos, vel = tr["pos"][f].numpy(), tr["vel"][f].numpy()
            else:
                P, V, M = rollouts[name]
                i = int(round(t / (M * frame_dt)))
                pos, vel = P[i], V[i]
            sp = np.linalg.norm(vel, axis=1)
            sc = ax.scatter(pos[:, 0], pos[:, 1], c=sp, s=2.5, cmap="viridis", vmin=0, vmax=4, linewidths=0)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#b8bdc9")
            if r == 0:
                ax.set_title(f"t = {t:.2f} s", fontsize=9, color=INK)
            if c == 0:
                ax.set_ylabel(title, fontsize=9, color=INK)
    fig.tight_layout()
    fig.colorbar(sc, ax=axs, shrink=0.6, pad=0.02, label="speed (m/s)")
    fig.savefig(os.path.join(PAPER_FIG, f"fluid_strips{tag}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def error_figure(eval_json, tag=""):
    res = json.load(open(os.path.join(OUT, eval_json)))
    runs = {n: r for n, r in res["runs"].items() if r["M"] == 4 and n in LABEL}
    tp = os.path.join(OUT, "fluid_timing.json")
    tim = json.load(open(tp)) if os.path.exists(tp) else {"runs": {}}
    native = tim.get("native_wall_per_sim_s")
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), dpi=150)
    for name, r in runs.items():
        t = np.arange(len(r["chamfer"])) * r["dt_ms"] / 1000
        c = COL[r["kind"]]
        ls = "--" if name == "les256_M4_d2" else "-"
        if name == "les256_M4_d2":
            c = "#7a1f2b"
        axs[0].plot(t, r["chamfer"], ls, color=c, label=LABEL[name])
        axs[1].plot(t, r["pos_err"], ls, color=c)
    axs[0].set_title("Chamfer distance to native (permutation-free)", fontsize=9)
    axs[1].set_title("per-particle position error", fontsize=9)
    for a in axs[:2]:
        a.set_xlabel("simulated time (s)"); a.set_ylabel("particle spacings"); a.grid(alpha=0.25)
    axs[0].legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(PAPER_FIG, f"fluid_error{tag}.png"))
    plt.close(fig)


def rest_figure(rest_json, tag=""):
    res = json.load(open(os.path.join(OUT, rest_json)))
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 3.6), dpi=150)
    nat = float(np.mean([r["native_speed"] for r in res.values()]))
    axs[0].axhline(nat, color=INK, lw=1.2, ls=":", label=f"native SPH at the start of the test ({nat:.2f} m/s); {0.065:.3f} when long calm")
    for name, r in res.items():
        c = COL[learned_kind(name)]
        ls = "-" if name.endswith("_d2") else "--"
        axs[0].plot(r["t"], r["speed"], ls, color=c, alpha=0.9 if name.endswith("_d2") else 0.5, label=name)
        axs[1].plot(r["t"], np.array(r["height_drift"]) * 1000, ls, color=c, alpha=0.9 if name.endswith("_d2") else 0.5)
    axs[0].set_xlabel("simulated time after a settled start (s)"); axs[0].set_ylabel("mean particle speed (m/s)")
    axs[1].set_xlabel("simulated time after a settled start (s)"); axs[1].set_ylabel("change of mean fluid height (mm)")
    for a in axs:
        a.grid(alpha=0.25)
    axs[0].set_yscale("log"); axs[0].legend(frameon=False, fontsize=6.5, ncol=2)
    axs[0].set_title("does a calm fluid stay calm?", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(PAPER_FIG, f"fluid_rest{tag}.png"))
    plt.close(fig)


def learned_kind(name):
    return "les_cat" if "cat" in name else ("gnn" if "gnn" in name else "les")


def training_figure(runs):
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=150)
    for n in runs:
        p = os.path.join(HERE, "runs", n, "train_log.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        h = np.array(d["hist"], float)
        ax.plot(h[:, 0], h[:, 2], color=COL[d["args"]["kind"]], label=n)
    ax.set_xlabel("step"); ax.set_ylabel("validation loss (normalised)"); ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PAPER_FIG, "fluid_training.png"))
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["les_M2", "gnn_M2"])
    ap.add_argument("--eval", default="fluid_eval.json")
    ap.add_argument("--data", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--rest", default=None, help="rest-test json (evaluate.py --mode rest)")
    ap.add_argument("--traj", type=int, default=1)
    ap.add_argument("--t0", type=int, default=60)
    a = ap.parse_args()
    strips(a.runs, test_idx=a.traj, t0=a.t0, data=a.data, tag=a.tag)
    if os.path.exists(os.path.join(OUT, a.eval)):
        error_figure(a.eval, a.tag)
    if a.rest and os.path.exists(os.path.join(OUT, a.rest)):
        rest_figure(a.rest, a.tag)
    training_figure(a.runs)
    print("figures written to", PAPER_FIG)
