"""Tables and learning-curve figures of the benchmark suite (results/bench/*.json), called from aggregate.py."""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from style import STYLE

HERE = os.path.dirname(os.path.abspath(__file__))
RB = os.path.join(HERE, "results", "bench")
FIG = os.path.join(HERE, "figures")
LABEL = {"les": "LESNet", "cnn": "CNN", "gpt": "GPT", "gnn": "graph network", "transformer": "Transformer + relation bias", "deepsets": "DeepSets",
         "gcn": "GCN", "gat": "GAT"}
COL = {"les": STYLE["LESNet"][0], "cnn": STYLE["CNN"][0], "gpt": STYLE["Transformer"][0], "gnn": STYLE["graph network"][0], "transformer": STYLE["Transformer"][0],
       "deepsets": STYLE["DeepSets"][0], "gcn": STYLE["GCN"][0], "gat": STYLE["GAT"][0]}
TASKS = {  # key: (file prefix, metric, higher-is-better, title)
    "cifar": ("cifar_", "acc", True, "CIFAR-10, pixels as entities (test accuracy)"),
    "cifark25": ("cifark25_", "acc", True, "CIFAR-10, LESNet with a 5x5 partner window (test accuracy)"),
    "cifarlong": ("cifarlong_", "acc", True, "CIFAR-10, 200 epochs (test accuracy)"),
    "charlmlong": ("charlmlong_", "bpc", False, "Tiny Shakespeare, 20 000 steps (validation bits per character)"),
    "pointset": ("pointset_", "acc", True, "MNIST as a point set (test accuracy)"),
    "charlm": ("charlm_", "bpc", False, "Tiny Shakespeare (validation bits per character)"),
    "cora": ("graph_cora_", "test_at_best_val", True, "Cora (test accuracy at best validation)"),
    "citeseer": ("graph_citeseer_", "test_at_best_val", True, "CiteSeer (test accuracy at best validation)"),
}


def step_ratio(key, fam):
    p = os.path.join(HERE, "results", "bench_timing.json")
    t = {"cifar": "cifar", "cifarlong": "cifar", "cifark25": "cifar", "pointset": "pointset", "charlm": "charlm", "charlmlong": "charlm"}.get(key)
    if not t or not os.path.exists(p):
        return 1.0
    d = json.load(open(p)).get(t, {})
    if fam not in d or "les" not in d:
        return 1.0
    return d[fam]["ms_per_step"] / d["les"]["ms_per_step"]


def runs(prefix):
    out = []
    for p in sorted(glob.glob(os.path.join(RB, prefix + "*.json"))):
        if "_sweep_" in os.path.basename(p):
            continue
        out.append(json.load(open(p)))
    return out


REAL_ROWS = {"pointset": "MNIST point set, accuracy (%)", "cora": "Cora, accuracy (%)", "citeseer": "CiteSeer, accuracy (%)",
             "cifar": "CIFAR-10, accuracy after 40 epochs (%)", "cifark25": "CIFAR-10, 5×5 window, accuracy after 40 epochs (%)",
             "charlm": "Tiny Shakespeare, validation bits per character"}


def tables():
    out = {}
    real = []
    for key, (prefix, metric, hib, title) in TASKS.items():
        rs = runs(prefix)
        if not rs:
            continue
        fams = [f for f in LABEL if any(r["family"] == f for r in rs)]
        rows = []
        for f in fams:
            sub = [r for r in rs if r["family"] == f]
            v = np.array([min(c[metric] for c in r["curve"]) if metric == "bpc" else r["final"][metric] for r in sub], float)   # language model: best validation point (both models overfit this corpus)
            fmt = (lambda x: f"{x:.3f}") if metric == "bpc" else (lambda x: f"{100 * x:.2f}")
            sd = f" ± {fmt(v.std(ddof=1)) if len(v) > 1 else ''}" if len(v) > 1 else ""
            rows.append([LABEL[f], f"{int(np.mean([r['params'] for r in sub])):,}", f"{fmt(v.mean())}{sd}", len(sub)])
            if key in REAL_ROWS:
                from scipy import stats
                ci = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
                real.append([REAL_ROWS[key], LABEL[f].replace("Transformer + relation bias", "Transformer + relation"), f"{int(np.mean([r['params'] for r in sub])):,}", f"{fmt(v.mean())} ± {fmt(ci)}", len(sub)])
        unit = "bits per character" if metric == "bpc" else "accuracy (%)"
        out["bench_" + key] = "| model | parameters | " + unit + " | seeds |\n|---|---|---|---|\n" + "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows)
        fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4))
        for f in fams:
            sub = [r for r in rs if r["family"] == f]
            L = min(len(r["curve"]) for r in sub)
            M = np.array([[c[metric] for c in r["curve"][:L]] for r in sub], float)
            S = np.array([c["step"] for c in sub[0]["curve"][:L]]); Tm = S * step_ratio(key, f)
            for ax, xs in ((axs[0], S), (axs[1], Tm)):
                ax.plot(xs, M.mean(0), "-", color=COL[f], label=LABEL[f], lw=1.3)
                if len(sub) > 1:
                    ax.fill_between(xs, M.min(0), M.max(0), color=COL[f], alpha=0.18)
        axs[0].set_xlabel("training step"); axs[1].set_xlabel("relative compute (LESNet-step equivalents)")
        for ax in axs:
            ax.set_ylabel({"acc": "test accuracy", "bpc": "validation bits per character", "test_at_best_val": "test accuracy"}[metric]); ax.grid(alpha=0.25)
        axs[0].legend(frameon=False)
        fig.tight_layout()
        if key == "charlm":
            fig.savefig(os.path.join(FIG, f"bench_{key}.png"))
        plt.close(fig)
    order = list(REAL_ROWS.values())
    real.sort(key=lambda r: order.index(r[0]))
    out["realdata"] = "| benchmark | model | parameters | score | seeds |\n|---|---|---|---|---|\n" + "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in real)
    tp = os.path.join(HERE, "results", "bench_timing.json")
    if os.path.exists(tp):
        d = json.load(open(tp))
        names = {"cifar": "CIFAR-10", "pointset": "MNIST point set", "charlm": "character language model"}
        rows = []
        for t, fams in d.items():
            for f, r in fams.items():
                rows.append(f"| {names[t]} | {LABEL[f]} | {r['params']:,} | {r['ms_per_step'] / d[t]['les']['ms_per_step']:.1f}x | {r['peak_mem_mb'] / d[t]['les']['peak_mem_mb']:.1f}x |")
        out["bench_timing"] = "| task | model | parameters | step cost relative to LESNet | peak memory relative to LESNet |\n|---|---|---|---|---|\n" + "\n".join(rows)
    return out
