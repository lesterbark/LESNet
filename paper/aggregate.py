"""Turn results/*.jsonl into tables (Markdown) and figures (PNG).

    run.bat -u aggregate.py            -> results/tables.md, figures/*.png

Statistics: every cell is a mean over seeds with a 95% confidence interval from the t distribution
(n-1 degrees of freedom); with 3 seeds the interval is wide on purpose.  "Wins" are claimed only when the
intervals separate.  A paired comparison across seeds is reported where the same seeds were used.
"""
import json
import math
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086}


def load(name):
    import glob
    paths = glob.glob(os.path.join(R, f"{name}.jsonl")) + sorted(glob.glob(os.path.join(R, f"{name}_s[0-9].jsonl")))
    out = []
    for p in paths:
        for line in open(p):
            try:
                out.append(json.loads(line))
            except Exception:  # noqa: BLE001
                pass
    return out


def mean_ci(v):
    v = np.array([x for x in v if x is not None and x == x], float)
    v = np.clip(v, -1.0, None)          # a diverged regression run (R^2 of -1000) is reported as -1: "worse than a constant, by a lot"
    if len(v) == 0:
        return float("nan"), float("nan"), 0
    if len(v) == 1:
        return float(v[0]), float("nan"), 1
    t = T975.get(len(v) - 1, 1.96)
    return float(v.mean()), float(t * v.std(ddof=1) / math.sqrt(len(v))), len(v)


def group(recs, key=lambda r: (r["task"], r["label"]), field="id"):
    g = defaultdict(list)
    for r in recs:
        g[key(r)].append(r.get(field))
    return g


def fmt(m, ci, n, digits=3):
    if n == 0 or m != m:
        return "-"
    return f"{m:.{digits}f} ± {ci:.{digits}f}" if n > 1 else f"{m:.{digits}f}"


DISPLAY = {"LES": "LESNet", "LES-kNN8": "LESNet, 8 nearest", "GNN": "graph network", "GNN-kNN8": "graph network, 8 nearest",
           "Transformer+pf": "Transformer + relation", "Transformer": "Transformer, no relation", "MLP-flat": "MLP-flat"}


def _fix(c):
    import re as _re
    c = _re.sub(r"(?<![\w.])-(?=\d)", "\u2212", c)
    return "n/a" if c == "-" else c


def md_table(header, rows):
    d = lambda c: _fix(DISPLAY.get(str(c), str(c)))
    out = ["| " + " | ".join(d(c) for c in header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(d(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ============================================================================================ tables
ABL_NAME = {"softmax (LES)": "relation: position, axis", "no pair features": "no relation", "pf: pos,axis,polar": "relation: position, axis, distance, direction"}
TASK_ORDER = ["nn_type", "count_r", "rank_ball", "closest_team", "exist_pair", "local_vel_avg", "nbody_acc"]
MODEL_ORDER = ["LES", "LES-kNN8", "Transformer+pf", "Transformer", "GNN", "GNN-kNN8", "GENConv", "DeepSets", "MLP-flat"]


def table_main(out):
    recs = load("main")
    if not recs:
        return
    g = group(recs)
    labels = [m for m in MODEL_ORDER if any((t, m) in g for t in TASK_ORDER)]
    rows = []
    for t in TASK_ORDER:
        base = [r.get("baseline") for r in recs if r["task"] == t and r.get("baseline") is not None and r.get("baseline") == r.get("baseline")]
        row = [f"`{t}`" + (f" (majority {base[0]:.2f})" if base else "")]
        best = max((mean_ci(g[(t, m)])[0] for m in labels if (t, m) in g), default=float("nan"))
        for m in labels:
            mu, ci, n = mean_ci(g.get((t, m), []))
            cell = fmt(mu, ci, n, 5 if t == "nbody_acc" else 3)
            row.append(f"**{cell}**" if n and mu + (ci if ci == ci else 0.0) >= best - 1e-12 else cell)
        rows.append(row)
    out["main"] = md_table(["task"] + labels, rows)
    # parameters and measured step time
    rows = []
    for m in labels:
        p = [r["params"] for r in recs if r["label"] == m]
        w = sorted({r["width"] for r in recs if r["label"] == m})
        ms = [r["ms_per_step"] for r in recs if r["label"] == m]
        rows.append([m, f"{int(np.mean(p)):,}", ", ".join(map(str, w[:4])) + ("…" if len(w) > 4 else ""), np.median(ms)])
    ref = next((r[3] for r in rows if r[0] == "LES"), 1.0)
    for r in rows:
        r[3] = f"{r[3] / ref:.1f}x"
    out["main_cost"] = md_table(["model", "params", "width", "step cost relative to LESNet (indicative)"], rows)


def table_ood(out):
    recs = load("main")
    if not recs:
        return
    rows = []
    labels = [m for m in MODEL_ORDER if any(r["label"] == m for r in recs)]
    for t in ("nn_type", "count_r", "local_vel_avg", "closest_team"):
        for m in labels:
            sub = [r for r in recs if r["task"] == t and r["label"] == m]
            if not sub:
                continue
            cells = []
            for fld in ("id", "ood48", "ood96"):
                mu, ci, n = mean_ci([r.get(fld) for r in sub])
                cells.append(fmt(mu, ci, n, 3))
            rows.append([f"`{t}`", m] + cells)
    out["ood"] = md_table(["task", "model", "in-distribution (n 8-24)", "n = 48", "n = 96"], rows)


def table_ablate(out):
    recs = load("ablate2")
    if not recs:
        return
    g = group(recs)
    order = []
    for r in recs:
        if r["label"] not in order:
            order.append(r["label"])
    order = [l for l in order if l in ABL_NAME]
    tasks = [t for t in TASK_ORDER if any(r["task"] == t for r in recs)]
    rows = []
    for lab in order:
        row = [ABL_NAME[lab]]
        for t in tasks:
            mu, ci, n = mean_ci(g.get((t, lab), []))
            row.append(fmt(mu, ci, n, 3))
        sd = [r["params"] for r in recs if r["label"] == lab]
        lrs = sorted({r["lr"] for r in recs if r["label"] == lab})
        row.append(f"{int(np.mean(sd)):,}")
        row.append(", ".join(f"{x:g}" for x in lrs))
        rows.append(row)
    out["ablate"] = md_table(["variant"] + [f"`{t}`" for t in tasks] + ["params"], [r[:-1] for r in rows])


def table_capacity(out):
    recs = load("capacity")
    if not recs:
        return
    for t in sorted({r["task"] for r in recs}):
        W = sorted({r["w"] for r in recs})
        D = sorted({r["d"] for r in recs})
        g = defaultdict(list)
        for r in recs:
            if r["task"] == t:
                g[(r["w"], r["d"])].append(r["id"])
        rows = [[f"width {w}"] + [fmt(*mean_ci(g.get((w, d), [])), 3) for d in D] for w in W]
        out[f"capacity_{t}"] = md_table([""] + [f"depth {d}" for d in D], rows)


def table_ksweep(out):
    recs = load("ksweep")
    if not recs:
        return
    ks = [1, 2, 4, 8, 16, 0]
    rows = []
    for t in sorted({r["task"] for r in recs}, key=TASK_ORDER.index):
        row = [f"`{t}`"]
        for k in ks:
            v = [r["id"] for r in recs if r["task"] == t and r["k"] == k]
            row.append(fmt(*mean_ci(v), 3))
        rows.append(row)
    out["ksweep"] = md_table(["task"] + [("K=%d" % k if k else "all-to-all") for k in ks], rows)


def table_speed(out):
    recs = load("speed")
    if not recs:
        return
    models = []
    for r in recs:
        if r["model"] not in models:
            models.append(r["model"])
    Ns = sorted({r["N"] for r in recs})
    ms = {(r["model"], r["N"]): r for r in recs}
    rows = []
    for m in models:
        row = [m]
        for N in Ns:
            r = ms.get((m, N))
            best = min((q["ms"] for (mm, nn), q in ms.items() if nn == N and not q.get("oom")), default=None)
            row.append("-" if r is None else ("OOM" if r.get("oom") else f"{r['ms'] / best:.1f}x"))
        rows.append(row)
    lm = "LES list K=16 (CUDA)"
    if (lm, Ns[0]) in ms and not ms[(lm, Ns[0])].get("oom"):
        b0 = ms[(lm, Ns[0])]["ms"]
        rows.append([f"{lm}, relative to its own N={Ns[0]}"] + [f"{ms[(lm, N)]['ms'] / b0:.2f}x" if (lm, N) in ms and not ms[(lm, N)].get("oom") else "-" for N in Ns])
    out["speed_ms"] = md_table(["model"] + [f"N={N}" for N in Ns], rows)
    rows = []
    for m in models:
        row = [m]
        for N in Ns:
            r = ms.get((m, N))
            bm = min((q["peak_mb"] for (mm, nn), q in ms.items() if nn == N and not q.get("oom")), default=None)
            row.append("-" if r is None else ("OOM" if r.get("oom") else f"{r['peak_mb'] / bm:.1f}x"))
        rows.append(row)
    out["speed_mem"] = md_table(["model"] + [f"N={N}" for N in Ns], rows)


def table_correctness(out):
    """Worst-case errors from tests.py / tests_sizes.py output (saved in results/)."""
    import re
    p1 = os.path.join(R, "correctness_tests.txt")
    p2 = os.path.join(R, "correctness_sizes.txt")
    if not (os.path.exists(p1) and os.path.exists(p2)):
        return
    t1 = open(p1, encoding="utf-8", errors="replace").read()
    t2 = open(p2, encoding="utf-8", errors="replace").read()
    fwd = [float(x) for x in re.findall(r"fwd\s+([0-9.]+e[-+]?\d+)", t1)]
    pg = [float(x) for x in re.findall(r"param-grad\s+([0-9.]+e[-+]?\d+)", t1)]
    ig = [float(x) for x in re.findall(r"input-grad\s+([0-9.]+e[-+]?\d+)", t1)]
    n_ok1 = len(re.findall(r"\bok\b", t1)); n_ok2 = len(re.findall(r"^\s+OK\b", t2, re.M))
    sf = [float(x) for x in re.findall(r"fwd\s+([0-9.]+e[-+]?\d+)", t2)]
    sg = [float(x) for k in ("gab", "gE", "gc1") for x in re.findall(k + r"\s+([0-9.]+e[-+]?\d+)", t2)]
    lst = [float(x) for x in re.findall(r"worst rel err over fwd/ga/gb/gE/gc1 and 5 tile sizes:\s+([0-9.]+e[-+]?\d+)", t2)]
    rows = [
        ["network-level checks (`tests.py`)", f"{n_ok1}", f"{max(fwd):.1e}", f"{max(max(pg), max(ig)):.1e}"],
        ["kernel size suite, all-to-all (`tests_sizes.py`)", f"{n_ok2 - len(lst)}", f"{max(sf):.1e}", f"{max(sg):.1e}"],
        ["kernel size suite, list layout", f"{len(lst)}", f"{max(lst):.1e}", f"{max(lst):.1e}"],
    ]
    out["correctness"] = md_table(["suite", "cases passed", "max forward error", "max gradient error"], rows)

def table_lr(out):
    """Learning-rate tuning (one seed, three tasks): mean score over the tasks, per family and lr."""
    recs = load("lr_tune")
    if not recs:
        return
    lrs = sorted({r["lr"] for r in recs})
    labels = [m for m in MODEL_ORDER if any(r["label"] == m for r in recs)]
    rows = []
    for m in labels:
        row = [m]
        for lr in lrs:
            v = [max(r["id"], -1.0) for r in recs if r["label"] == m and r["lr"] == lr and r["id"] == r["id"]]
            row.append(f"{np.mean(v):.3f}" if len(v) == 3 else "-")
        best = json.load(open(os.path.join(R, "lr_best.json"))).get(m)
        row.append(f"{best:g}" if best else "-")
        rows.append(row)
    out["lr_tune"] = md_table(["model"] + [f"lr {lr:g}" for lr in lrs] + ["chosen"], rows)

def write_tables():
    out = {}
    for f in (table_main, table_ood, table_ablate, table_capacity, table_ksweep, table_speed, table_fluid,
              table_switch, table_initsens, table_scale, table_isotime, table_pairs, table_correctness, table_lr,
              table_toptag, table_washout, table_multiswitch, table_hiddeneq, table_fluid_v2, table_fluid_cluster, table_softmax_probe, table_nbody_check, table_multiswitch_remedy, table_toptag_bench, table_fluid_seeds, table_noact):
        f(out)
    import aggregate_bench
    out.update(aggregate_bench.tables())
    rows = []
    def body(t):
        return [[c.strip() for c in l.strip().strip("|").split("|")] for l in t.split("\n")[2:] if l.startswith("|")]
    for r in body(out.get("toptag_bench", "")):
        rows.append(["jet tagging", r[0], r[1], r[2], r[3], r[4]])
    for r in body(out.get("bench_timing", "")):
        rows.append([r[0], r[1].replace("Transformer + relation bias", "Transformer + relation"), "n/a", r[2], r[3], r[4]])
    if rows:
        out["cost"] = md_table(["benchmark", "model", "parameters", "step cost", "peak memory"], [[r[0], r[1], r[3], r[4].replace("x", "\u00d7"), r[5].replace("x", "\u00d7")] for r in rows])
    txt = "\n\n".join(f"<!-- table:{k} -->\n{v}\n<!-- /table -->" for k, v in out.items())
    open(os.path.join(R, "tables.md"), "w", encoding="utf-8").write(txt)
    print("tables:", ", ".join(out))


# ============================================================================================ figures
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from style import STYLE, LABELS, style_of  # noqa: E402

INK = "#1d2330"
COLORS = {"LES": "#d1495b", "LES-kNN8": "#e98a97", "Transformer+pf": "#2e86ab", "Transformer": "#7fb6cf",
          "GNN": "#3a7d44", "GNN-kNN8": "#8fc29a", "GENConv": "#7a4fa3", "DeepSets": "#9a7d0a", "MLP-flat": "#8a93a6"}


def fig_main():
    recs = load("main")
    if not recs:
        return
    g = group(recs)
    labels = [m for m in MODEL_ORDER if any(r["label"] == m for r in recs)]
    fig, axs = plt.subplots(2, 4, figsize=(15, 6.2), dpi=150)
    for ax, t in zip(axs.flat, TASK_ORDER):
        for i, m in enumerate(labels):
            mu, ci, n = mean_ci(g.get((t, m), []))
            if n:
                ax.bar(i, mu, yerr=0 if n < 2 else ci, color=COLORS[m], capsize=2, width=0.78)
        base = [r.get("baseline") for r in recs if r["task"] == t and r.get("baseline") is not None]
        if base:
            ax.axhline(base[0], color="#999", lw=0.8, ls="--")
        ax.set_title(t, fontsize=9.5, color=INK)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
        lo = min([mean_ci(g.get((t, m), []))[0] for m in labels if (t, m) in g] + [1.0])
        ax.set_ylim(max(-0.2, min(0.4, lo - 0.1)), 1.02)
        ax.grid(axis="y", alpha=0.25)
    axs.flat[-1].axis("off")
    axs.flat[-1].text(0, 0.6, "bars: mean over seeds\nerror bars: 95% CI\ndashed: majority class\n(classification tasks)", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "main_results.png"))
    plt.close(fig)


def fig_ablate():
    recs = load("ablate2")
    if not recs:
        return
    g = group(recs)
    order = []
    for r in recs:
        if r["label"] not in order:
            order.append(r["label"])
    order = [l for l in order if l in ABL_NAME]
    tasks = [t for t in TASK_ORDER if any(r["task"] == t for r in recs)]
    fig, axs = plt.subplots(1, len(tasks), figsize=(3.0 * len(tasks), 4.6), dpi=150, sharey=False)
    for ax, t in zip(np.atleast_1d(axs), tasks):
        mus = [mean_ci(g.get((t, lab), []))[0] for lab in order]
        cis = [mean_ci(g.get((t, lab), []))[1] for lab in order]
        cols = ["#d1495b" if lab.startswith("softmax") else "#8a93a6" for lab in order]
        ax.barh(range(len(order)), mus, xerr=[0 if c != c else c for c in cis], color=cols, capsize=2)
        ax.set_yticks(range(len(order))); ax.set_yticklabels([ABL_NAME[o] for o in order] if ax is np.atleast_1d(axs)[0] else [], fontsize=7.5)
        ax.invert_yaxis(); ax.set_title(t, fontsize=9.5, color=INK); ax.grid(axis="x", alpha=0.25)
        lo = min(m for m in mus if m == m)
        ax.set_xlim(max(-0.3, lo - 0.05), 1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "ablation.png"))
    plt.close(fig)


def fig_capacity():
    recs = load("capacity")
    if not recs:
        return
    tasks = sorted({r["task"] for r in recs}, key=TASK_ORDER.index)
    fig, axs = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 3.4), dpi=150)
    for ax, t in zip(np.atleast_1d(axs), tasks):
        W = sorted({r["w"] for r in recs}); D = sorted({r["d"] for r in recs})
        M = np.array([[mean_ci([r["id"] for r in recs if r["task"] == t and r["w"] == w and r["d"] == d])[0] for d in D] for w in W])
        im = ax.imshow(M, cmap="viridis", origin="lower", aspect="auto")
        ax.set_xticks(range(len(D))); ax.set_xticklabels(D); ax.set_yticks(range(len(W))); ax.set_yticklabels(W)
        ax.set_xlabel("depth"); ax.set_ylabel("width"); ax.set_title(t, fontsize=9.5, color=INK)
        for i in range(len(W)):
            for j in range(len(D)):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if M[i, j] < M.mean() else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "capacity.png"))
    plt.close(fig)


def fig_ksweep():
    recs = load("ksweep")
    if not recs:
        return
    tasks = sorted({r["task"] for r in recs}, key=TASK_ORDER.index)
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ks = [1, 2, 4, 8, 16]
    xs = [1, 2, 4, 8, 16, 32]
    for t, c, mk in zip(tasks, ["#222222", "#777777", "#bbbbbb", "#e69f00"], ["o", "s", "^", "v"]):
        mu = [mean_ci([r["id"] for r in recs if r["task"] == t and r["k"] == k])[0] for k in ks]
        ci = [mean_ci([r["id"] for r in recs if r["task"] == t and r["k"] == k])[1] for k in ks]
        full, fci, _ = mean_ci([r["id"] for r in recs if r["task"] == t and r["k"] == 0])
        ax.errorbar(xs, mu + [full], yerr=[0 if x != x else x for x in ci + [fci]], marker=mk, color=c, label=t, capsize=2, lw=1.1)
    ax.set_xscale("log", base=2); ax.set_xticks(xs); ax.set_xticklabels(["1", "2", "4", "8", "16", "all"])
    ax.set_xlabel("nearest partners per entity K"); ax.set_ylabel("score"); ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="lower right", prop={"family": "monospace", "size": 6.5})
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "ksweep.png"))
    plt.close(fig)


def fig_speed():
    recs = load("speed")
    if not recs:
        return
    models = []
    for r in recs:
        if r["model"] not in models:
            models.append(r["model"])
    fam = {"LES all-to-all (CUDA)": ("LESNet", "-"), "LES list K=16 (CUDA)": ("LESNet", "-"), "LES all-to-all (PyTorch)": ("LESNet", "--"),
           "Transformer": ("Transformer", "-"), "Transformer+pf": ("Transformer", ":"), "GNN all-to-all (PyTorch)": ("graph network", "--"),
           "GNN gather K=16": ("graph network", "-")}
    shown = {"LES all-to-all (CUDA)": "LESNet all-to-all, fused", "LES list K=16 (CUDA)": "LESNet list K = 16, fused", "LES all-to-all (PyTorch)": "LESNet all-to-all, PyTorch",
             "Transformer": "Transformer", "Transformer+pf": "Transformer + relation", "GNN all-to-all (PyTorch)": "graph network all-to-all, PyTorch",
             "GNN gather K=16": "graph network list K = 16"}
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7))
    hs = {}
    for m in models:
        f, ls = fam[m]
        c, mk = STYLE[f]
        mk = "s" if "list" in m or "gather" in m else mk
        pts = [(r["N"], r["ms"], r["peak_mb"]) for r in recs if r["model"] == m and not r.get("oom")]
        if not pts:
            continue
        n, ms, mem = zip(*pts)
        h, = axs[0].plot(n, ms, ls=ls, marker=mk, color=c)
        axs[1].plot(n, mem, ls=ls, marker=mk, color=c)
        hs[shown[m]] = h
        oom = [r["N"] for r in recs if r["model"] == m and r.get("oom")]
        if oom:
            axs[0].annotate("OOM", (oom[0], ms[-1]), textcoords="offset points", xytext=(0, 5), fontsize=6, color=c, ha="center")
    for a, yl in zip(axs, ["ms per forward and backward\n(8192 entities)", "peak GPU memory (MB)"]):
        a.set_xscale("log", base=2); a.set_yscale("log"); a.set_xlabel("entities per set N"); a.set_ylabel(yl); a.grid(alpha=0.25, which="both")
    fig.legend(list(hs.values()), list(hs.keys()), loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "speed.png"))
    plt.close(fig)


def fig_curves():
    recs = load("main")
    if not recs:
        return
    fig, axs = plt.subplots(2, 4, figsize=(15, 6.2), dpi=150)
    for ax, t in zip(axs.flat, TASK_ORDER):
        for m in MODEL_ORDER:
            cs = [r["curve"] for r in recs if r["task"] == t and r["label"] == m and r["curve"]]
            if not cs:
                continue
            L = min(len(c) for c in cs)
            x = [p[0] for p in cs[0][:L]]
            y = np.mean([[p[1] for p in c[:L]] for c in cs], 0)
            ax.plot(x, y, color=COLORS[m], label=m, lw=1.4)
        ax.set_title(t, fontsize=9.5, color=INK); ax.grid(alpha=0.25); ax.set_xlabel("step")
    axs.flat[-1].axis("off")
    h, l = axs.flat[0].get_legend_handles_labels()
    axs.flat[-1].legend(h, l, loc="center", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "curves.png"))
    plt.close(fig)


# ============================================================================================ fluid + extras
FLUID_OUT = os.path.join(HERE, "..", "fluid", "out")


def _fluid(name):
    p = os.path.join(FLUID_OUT, name)
    return json.load(open(p)) if os.path.exists(p) else None


def table_fluid(out):
    for key, fname in (("fluid", "fluid_eval.json"), ("fluid_big", "fluid_eval_big.json"), ("fluid_noise", "fluid_eval_noise.json")):
        d = _fluid(fname)
        if not d:
            continue
        tim = _fluid("fluid_timing.json") or {"native_wall_per_sim_s": None, "runs": {}}
        native = tim["native_wall_per_sim_s"]
        rows = []
        for name, r in d["runs"].items():
            t = tim["runs"].get(name)
            ms = f"{t['ms_step']:.2f}" if t else "-"
            sp = f"{native / t['wall_per_sim_s']:.1f}x" if (t and native) else "-"
            sd = r.get("chamfer_end_sd")
            rows.append([name, r["kind"].upper(), f"{r['params']:,}", f"{r['dt_ms']:.1f}",
                         f"{r['chamfer_end']:.2f}" + (f" (sd {sd:.2f})" if sd is not None else ""),
                         f"{r['pos_err_end']:.1f}", f"{r['dens_err'] * 100:.1f}%", f"{r['ke_ratio']:.2f}", ms, sp])
        hdr = ["run", "model", "params", "step (ms of sim)", "Chamfer @0.5 s (spacings)", "position err (spacings)",
               "density err", "KE ratio", "ms / learned step", "speed-up vs native"]
        n_part = d.get("n_particles")
        note = (f"\n\nnative SPH: {native:.2f} wall-seconds per simulated second at 800 particles (timings on an idle GPU)." if native else "")
        if n_part:
            note += f" Held-out scenes have {min(n_part)}-{max(n_part)} particles."
        out[key] = md_table(hdr, rows) + note
    d = _fluid("fluid_scaling.json")
    if d:
        rows = []
        for i, N in enumerate(d["N"]):
            row = [N, f"{d['native'][i]:.2f}"]
            for n in d["runs"]:
                row.append(f"{d['runs'][n][i]:.2f} ({d['native'][i] / d['runs'][n][i]:.1f}x)")
            rows.append(row)
        out["fluid_scaling"] = md_table(["particles", "native"] + list(d["runs"]), rows)


def table_switch(out):
    recs = load("switch")
    if not recs:
        return
    rows = []
    for comb in ("softmax", "max", "sum"):
        for a, b in (("local_vel_avg", "nbody_acc"), ("nbody_acc", "local_vel_avg")):
            sub = [r for r in recs if r["label"] == comb and r["task_a"] == a]
            if not sub:
                continue
            rows.append([comb, f"{a} -> {b}", fmt(*mean_ci([r["score_a"] for r in sub]), 3),
                         fmt(*mean_ci([r["score_b_after_switch"] for r in sub]), 3),
                         fmt(*mean_ci([r["score_b_scratch"] for r in sub]), 3)])
    out["switch"] = md_table(["combine", "switch", "score on A before", "score on B after switch", "score on B from scratch"], rows)


def table_initsens(out):
    recs = load("initsens")
    if not recs:
        return
    inits = sorted({r["init"] for r in recs})
    rows = []
    for t in sorted({r["task"] for r in recs}):
        for comb in ("softmax", "sum", "max"):
            row = [f"`{t}`", comb]
            for i in inits:
                row.append(fmt(*mean_ci([r["id"] for r in recs if r["task"] == t and r["combine"] == comb and r["init"] == i]), 3))
            rows.append(row)
    out["initsens"] = md_table(["task", "combine"] + [f"init x{i:g}" for i in inits], rows)


def table_scale(out):
    recs = load("scale")
    if not recs:
        return
    sizes = sorted({r["target_params"] for r in recs})
    labels = [m for m in MODEL_ORDER if any(r["label"] == m for r in recs)]
    rows = []
    for t in sorted({r["task"] for r in recs}):
        for m in labels:
            row = [f"`{t}`", m]
            for p in sizes:
                row.append(fmt(*mean_ci([r["id"] for r in recs if r["task"] == t and r["label"] == m and r["target_params"] == p]), 3))
            rows.append(row)
    out["scale"] = md_table(["task", "model"] + [f"~{p // 1000}k params" for p in sizes], rows)


def table_isotime(out):
    recs = load("isotime")
    if not recs:
        return
    g = group(recs)
    labels = [m for m in MODEL_ORDER if any(r["label"] == m for r in recs)]
    rows = []
    for t in sorted({r["task"] for r in recs}, key=TASK_ORDER.index):
        row = [f"`{t}`"]
        for m in labels:
            row.append(fmt(*mean_ci(g.get((t, m), [])), 3))
        rows.append(row)
    steps = ["steps: " + ", ".join(f"{m} {int(np.mean([r['steps'] for r in recs if r['label'] == m]))}" for m in labels)]
    out["isotime"] = md_table(["task"] + labels, rows)


def table_pairs(out):
    """Paired comparisons across seeds (same data stream per seed): LES vs the strongest baselines."""
    from scipy import stats
    recs = load("main")
    if not recs:
        return
    rows = []
    for t in TASK_ORDER:
        by = {m: {r["seed"]: r["id"] for r in recs if r["task"] == t and r["label"] == m} for m in MODEL_ORDER}
        for other in ("Transformer+pf", "GNN", "DeepSets"):
            seeds = sorted(set(by["LES"]) & set(by.get(other, {})))
            if len(seeds) < 3:
                continue
            a = np.array([by["LES"][s] for s in seeds]); b = np.array([by[other][s] for s in seeds])
            a, b = np.clip(a, -1, None), np.clip(b, -1, None)
            p = stats.ttest_rel(a, b).pvalue if np.std(a - b) > 0 else float("nan")
            rows.append([f"`{t}`", f"LES - {other}", f"{(a - b).mean():+.3f}", f"{p:.3f}" if p == p else "-"])
    out["pairs"] = md_table(["task", "comparison", "mean paired difference", "paired t-test p (n=3)"], rows)


def fig_fluid_scaling():
    d = _fluid("fluid_scaling_v2.json")
    if not d:
        return
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    N = d["N"]
    base = d["runs"]["les_M4_d2"][0]
    nat = [(n, v) for n, v in zip(N, d["native"]) if v]
    bru = [(n, v) for n, v in zip(N, d["native_brute"]) if v]
    ax.plot([a for a, _ in nat], [b / base for _, b in nat], "o-", color="#1d2330", label="native SPH, cell list")
    ax.plot([a for a, _ in bru], [b / base for _, b in bru], "o--", color="#8a93a6", label="native SPH, all pairs")
    cols = {"les_M4_d2": STYLE["LESNet"][0], "les256_M4_d2": STYLE["LESNet"][0], "gnn72_M4_d2": STYLE["graph network"][0]}
    names = {"les_M4_d2": "LESNet 105k", "les256_M4_d2": "LESNet 407k", "gnn72_M4_d2": "graph network 107k"}
    for n, v in d["runs"].items():
        if n not in names:
            continue
        big = n == "les256_M4_d2"
        ax.plot(N, [x / base for x in v], ls="--" if big else "-", marker="o" if "les" in n else "s", color=cols.get(n, "#555"), label=names.get(n, n))
    ax.axvspan(2400, max(N) * 1.2, color="#f2f4f8", zorder=0)
    ax.set_ylim(0.55, None); ax.text(2500, 9.0, "learned\nmodels only", fontsize=6, color="#4a5368", va="top")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("particles"); ax.set_ylabel("cost per simulated second\n(relative to LESNet at 500 particles)")
    ax.grid(alpha=0.25, which="both"); ax.legend(frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.45, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fluid_scaling.png"))
    plt.close(fig)


def fig_switch():
    recs = load("switch")
    if not recs:
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=150)
    for i, comb in enumerate(("softmax", "max", "sum")):
        aft = [r["score_b_after_switch"] for r in recs if r["label"] == comb]
        scr = [r["score_b_scratch"] for r in recs if r["label"] == comb]
        ax.bar(i - 0.2, np.mean(aft), 0.38, color="#d1495b" if comb == "softmax" else "#8a93a6", label="after switch" if i == 0 else None)
        ax.bar(i + 0.2, np.mean(scr), 0.38, color="#c7ccd8", label="from scratch" if i == 0 else None)
    ax.set_xticks(range(3)); ax.set_xticklabels(["softmax", "hard max", "sum"]); ax.set_ylabel("R^2 on the new task")
    ax.grid(axis="y", alpha=0.25); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "switch.png"))
    plt.close(fig)


def fig_ood():
    recs = load("main")
    if not recs:
        return
    tasks = ["nn_type", "count_r", "local_vel_avg", "closest_team"]
    labels = [m for m in MODEL_ORDER if m != "MLP-flat"]
    fig, axs = plt.subplots(1, 4, figsize=(7.0, 2.3), sharey=False)
    handles = {}
    for ax, t in zip(axs, tasks):
        for m in labels:
            sub = [r for r in recs if r["task"] == t and r["label"] == m]
            ys = [mean_ci([r.get(f) for r in sub])[0] for f in ("id", "ood48", "ood96")]
            st = style_of(m)
            h, = ax.plot([24, 48, 96], np.clip(ys, -1, 1), color=st["color"], ls=st["ls"], marker=st["marker"], mfc=st["mfc"], lw=1.2, ms=3.5)
            handles[st["name"]] = h
        ax.set_xscale("log", base=2); ax.set_xticks([24, 48, 96]); ax.set_xticklabels(["8-24", "48", "96"])
        ax.set_xlim(20, 110)
        if t != "closest_team":
            ax.axhline(-1, color="#999", lw=0.6, ls=":")
        ax.set_title(t, fontfamily="monospace"); ax.set_xlabel("entities per set"); ax.grid(alpha=0.25)
    axs[0].set_ylabel("score (clipped at \u22121)")
    fig.legend(list(handles.values()), list(handles.keys()), loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.17))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "ood.png"))
    plt.close(fig)


# ============================================================================================ benchmark tables
def _jsons(sub):
    import glob
    out = []
    for p in sorted(glob.glob(os.path.join(R, sub, "*.json"))):
        try:
            out.append(json.load(open(p)))
        except Exception:  # noqa: BLE001
            pass
    return out


TT_LABEL = {"les": "LESNet", "les_cat": "LESNet + input re-concatenation", "gnn": "graph network", "transformer": "Transformer + relation",
            "deepsets": "DeepSets", "gen": "GENConv"}


def table_toptag(out):
    recs = [r for r in _jsons("toptag_v2") if "test" in r and not r["name"].startswith(("smoke", "lr_")) and r["family"] != "les_cat"]
    if not recs:
        return
    def tag(r):
        n = r["name"]
        return "iso_gnn" if "isotime_gnn" in n else ("iso_tf" if "isotime_tf" in n else "")
    iso_label = {"iso_gnn": "LESNet, at the cost of 1000 graph-network steps (relative compute)", "iso_tf": "LESNet, at the cost of 1000 Transformer steps (relative compute)"}
    rows = []
    groups = []
    for fam in ("deepsets", "transformer", "gnn", "gen", "les"):
        for st in sorted({r["steps"] for r in recs if r["family"] == fam and tag(r) == ""}):
            groups.append((fam, "", st))
    for fam, tg, st in groups:
        sub = [r for r in recs if r["family"] == fam and tag(r) == tg and (st is None or r["steps"] == st)]
        if not sub:
            continue
        def ms(key):
            v = np.array([r["test"][key] for r in sub], float)
            h = T975.get(len(v) - 1, float("nan")) * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else None
            if key in ("auc", "acc"):
                return f"{v.mean():.4f}" + (f" ± {h:.4f}" if h is not None else "")
            return f"{v.mean():.0f}" + (f" ± {h:.0f}" if h is not None else "")
        r0 = sub[0]
        rows.append([iso_label.get(tg, TT_LABEL[fam]), f"{r0['params']:,}", f"{int(np.mean([r['steps'] for r in sub])):,}",
                     ms("acc"), ms("auc"), ms("rej50"), ms("rej30"), len(sub)])
    rows.append(["ParticleNet (published)†", "366,000", "-", "0.940", "0.9858", "397", "1615", "-"])
    rows.append(["Particle Flow Network (published)†", "-", "-", "0.932", "0.9819", "247", "888", "-"])
    out["toptag"] = md_table(["model", "params", "steps", "accuracy", "AUC", "1/ε_B at ε_S = 50%", "1/ε_B at ε_S = 30%", "seeds"], rows)


def table_washout(out):
    recs = load("depth_washout")
    if not recs:
        return
    fams = [f for f in ("les", "les_cat", "gnn", "gnn_nores") if any(r["family"] == f for r in recs)]
    names = {"les": "LESNet", "les_cat": "LESNet + re-concatenated input", "gnn": "graph net (residual)", "gnn_nores": "graph net (no residual)"}
    rows = []
    for d in sorted({r["depth"] for r in recs}):
        row = [d]
        for f in fams:
            v = [r["acc"] for r in recs if r["family"] == f and r["depth"] == d]
            row.append(fmt(*mean_ci(v), 3) + (" (diverged: %d)" % sum(1 for r in recs if r["family"] == f and r["depth"] == d and r["diverged"]) if any(r["diverged"] for r in recs if r["family"] == f and r["depth"] == d) else ""))
        rows.append(row)
    out["washout"] = md_table(["layers"] + [names[f] for f in fams], rows)


def table_multiswitch(out):
    recs = load("memswitch_multi")
    if not recs:
        return
    rows = []
    for w in sorted({r["width"] for r in recs}):
        for v in ("softmax+leaky", "max+leaky"):
            sub = [r for r in recs if r["width"] == w and r["variant"] == v]
            if not sub:
                continue
            P = len(sub[0]["loss"])
            sel = [0, 1, 3, P - 1] if P >= 4 else list(range(P))
            row = [w, v, len(sub)]
            for ph in sel:
                row.append(f"{np.mean([r['loss'][ph] for r in sub]):.4f} / {np.mean([r['scratch'][ph] for r in sub]):.4f}")
            rows.append(row)
    P = len(recs[0]["loss"])
    sel = [0, 1, 3, P - 1] if P >= 4 else list(range(P))
    out["multiswitch"] = md_table(["width", "variant", "seeds"] + [f"target {i + 1}: switched / from scratch" for i in sel], rows)


FLUID_KEEP = ("les_M4_d2", "les256_M4_d2", "gnn72_M4_d2")
FLUID_ROUND = {"les_M4": "1", "les_cat_M4": "1", "gnn72_M4": "1", "les_M2": "1", "les_cat_M2": "1", "gnn72_M2": "1",
               "les_M4_d2": "2", "lescat_M4_d2": "2", "les256_M4_d2": "2", "gnn72_M4_d2": "2"}
FLUID_NAME = {"les_M4": "LESNet", "les_cat_M4": "LESNet + re-concatenation", "gnn72_M4": "graph net",
              "les_M4_d2": "LESNet (105k)", "lescat_M4_d2": "LESNet + re-concatenation", "les256_M4_d2": "LESNet (407k)",
              "gnn72_M4_d2": "graph network (107k)"}


def table_nbody_check(out):
    p = os.path.join(R, "nbody_check.json")
    if not os.path.exists(p):
        return
    d = json.load(open(p))
    rows = [[f"force of the {k} nearest partner" + ("s" if k > 1 else "") + " only", f"{d[f'r2_nearest_{k}']:.2f}"] for k in (1, 2, 3, 5, 8)]
    rows.append(["linear function of the entity's position and the centroid of the set", f"{d['r2_linear_position_centroid']:.2f}"])
    out["nbody_check"] = md_table(["predictor of the `nbody_acc` target", "R²"], rows)


def table_softmax_probe(out):
    p = os.path.join(R, "probe", "saturation.json")
    if not os.path.exists(p):
        return
    rows = []
    for r in json.load(open(p)):
        f = r["rows"][-1]
        rows.append([f"`{r['task']}`", f"{r['lr']:g}", f"{r['final']:.3f}", f"{f['norm_entropy']:.3f}", f"{f['mean_max_weight']:.3f}", f"{f['grad_top_share']:.3f}"])
    out["softmax_probe"] = md_table(["task", "learning rate", "score", "normalised entropy of the weights", "mean largest weight", "share of the gradient on the largest-gradient partner"], rows)


def table_fluid_cluster(out):
    """Rollout error with the uncertainty computed over trajectories (the six rollouts of a trajectory are averaged first)."""
    d = _fluid("fluid_eval_v2_rollouts.json")
    if not d:
        return
    per = {}
    for name, r in d["runs"].items():
        ro = r.get("rollouts")
        if not ro:
            continue
        traj = np.array(ro["traj"])
        per[name] = {m: np.array([np.mean(np.array(ro[m])[traj == t]) for t in sorted(set(traj))])
                     for m in ("chamfer_end", "pos_err_end", "dens_err")}

    def cell(x, scale=1.0, nd=2):
        x = np.asarray(x, float) * scale
        mu, ci = float(x.mean()), float(T975.get(len(x) - 1, 1.96) * x.std(ddof=1) / math.sqrt(len(x)))
        return f"{mu:.{nd}f} ± {ci:.{nd}f}"

    sfx = ["", "_s1", "_s2", "_s3"]
    les_n = [f"les_M4_d2{s}" for s in sfx]
    gnn_n = [f"gnn72_M4_d2{s}" for s in sfx]
    rows = []
    for k, (a, b) in enumerate(zip(les_n, gnn_n)):
        for name, lab in ((a, f"LESNet (105k), seed {k}"), (b, f"graph network (107k), seed {k}")):
            if name in per:
                p = per[name]
                rows.append([lab, cell(p["chamfer_end"]), cell(p["pos_err_end"], nd=1), cell(p["dens_err"], 100.0, 1) + "%"])
    if "les256_M4_d2" in per:
        p = per["les256_M4_d2"]
        rows.append(["LESNet (407k), seed 0", cell(p["chamfer_end"]), cell(p["pos_err_end"], nd=1), cell(p["dens_err"], 100.0, 1) + "%"])
    out["fluid_cluster"] = md_table(["model", "Chamfer @0.5 s (spacings)", "position error (spacings)", "density error"], rows)
    # paired differences LESNet - graph network over the 12 trajectories, same training seed, and over the mean of the seeds
    rows = []
    pairs = [(a, b, f"seed {k}") for k, (a, b) in enumerate(zip(les_n, gnn_n)) if a in per and b in per]
    ms = ("chamfer_end", "pos_err_end", "dens_err")

    def row(lab, dif):
        return [lab] + [cell(dif[m], 100.0 if m == "dens_err" else 1.0, 3 if m == "chamfer_end" else 2) + ("%" if m == "dens_err" else "") for m in ms]

    for a, b, lab in pairs:
        rows.append(row(lab, {m: per[a][m] - per[b][m] for m in ms}))
    if len(pairs) > 1:
        rows.append(row(f"mean of {len(pairs)} seeds", {m: np.mean([per[a][m] - per[b][m] for a, b, _ in pairs], 0) for m in ms}))
    out["fluid_cluster_diff"] = md_table(["LESNet minus graph network", "Chamfer", "position error (spacings)", "density error"], rows)

def table_fluid_v2(out):
    acc = _fluid("fluid_eval_v2.json")
    if not acc:
        return
    tim = _fluid("fluid_timing.json") or {"native_wall_per_sim_s": None, "native_brute_wall_per_sim_s": None, "runs": {}}
    nat_c, nat_b = tim.get("native_wall_per_sim_s"), tim.get("native_brute_wall_per_sim_s")
    rows = []
    for name, r in acc["runs"].items():
        if name not in FLUID_KEEP:
            continue
        t = tim["runs"].get(name)
        ms = f"{t['ms_step']:.1f}" if t else "-"
        sp_c = f"{nat_c / t['wall_per_sim_s']:.1f}x" if (t and nat_c) else "-"
        sp_b = f"{nat_b / t['wall_per_sim_s']:.1f}x" if (t and nat_b) else "-"
        rows.append([FLUID_NAME.get(name, name), f"{r['params']:,}", f"{r['chamfer_end']:.2f} (sd {r['chamfer_end_sd']:.2f})",
                     f"{r['pos_err_end']:.1f}", f"{r['dens_err'] * 100:.1f}%", f"{r['ke_ratio']:.2f}"])
    hdr = ["model", "params", "Chamfer @0.5 s (spacings)", "position err", "density err", "KE ratio"]
    note = f"At 800 particles the cell-list solver is {nat_b / nat_c:.1f}x faster than the all-pairs solver." if nat_c and nat_b else ""
    out["fluid_v2"] = md_table(hdr, rows)
    rest = _fluid("fluid_rest_v2.json")
    if rest:
        rows = []
        for name, r in rest.items():
            if name not in FLUID_KEEP:
                continue
            bk = r.get("speed_late_by_kind", {})
            rows.append([FLUID_NAME.get(name, name), f"{r['speed_late']:.3f}", f"{r['height_drift_end'] * 1000:+.1f}".replace("-", "\u2212"),
                         f"{r['chamfer_ref']:.2f}" if r["chamfer_ref"] is not None else "n/a"])
        nat = np.mean([r["native_speed"] for r in rest.values()])
        out["fluid_rest"] = md_table(["model", "mean speed, last 0.75 s (m/s)", "change of fluid height (mm)", "Chamfer distance at 1 s (spacings)"], rows)
    big = _fluid("fluid_eval_big_v2.json")
    if big:
        rows = []
        for name, r in big["runs"].items():
            if name not in FLUID_KEEP:
                continue
            rows.append([FLUID_NAME.get(name, name), f"{r['chamfer_end']:.2f}", f"{r['dens_err'] * 100:.1f}%", f"{r['ke_ratio']:.2f}"])
        out["fluid_big_v2"] = md_table(["model", "Chamfer @0.5 s", "density err", "KE ratio"], rows) + \
            ""
    sc = _fluid("fluid_scaling_v2.json")
    if sc:
        rows = []
        unit = sc["runs"]["les_M4_d2"][0]
        names = {"les_M4_d2": "LESNet 105k, 8.3 ms", "gnn72_M4_d2": "graph network, 8.3 ms", "les256_M4_d2": "LESNet 407k, 8.3 ms"}
        for i, N in enumerate(sc["N"]):
            row = [N] + [f"{sc[k][i] / unit:.2f}" if sc[k][i] else "-" for k in ("native", "native_brute")]
            row += [f"{sc['runs'][n][i] / unit:.2f}" for n in names]
            rows.append(row)
        out["fluid_scaling_v2"] = md_table(["particles", "SPH, cell list", "SPH, all pairs"] + list(names.values()), rows)


def table_multiswitch_remedy(out):
    import glob
    rows = []
    for tag, label in (("", "Adam, lr 0.03 (the main run)"), ("_lr01", "Adam, lr 0.01"), ("_wd", "AdamW, lr 0.03, weight decay 0.1")):
        p = os.path.join(R, f"memswitch_multi{tag}.jsonl")
        if not os.path.exists(p):
            continue
        recs = [json.loads(l) for l in open(p)]
        for v in ("softmax+leaky", "max+leaky"):
            sub = [r for r in recs if r["variant"] == v and r["width"] == 24]
            if not sub:
                continue
            L = np.array([r["loss"] for r in sub]); S = np.array([r["scratch"] for r in sub])
            rows.append([label, v, len(sub)] + [f"{L[:, k].mean():.3f} / {S[:, k].mean():.3f}" for k in (0, 2, 5)])
    if rows:
        out["multiswitch_remedy"] = md_table(["optimiser", "combine", "seeds", "target 1: switched / scratch", "target 3", "target 6"], rows)


def table_toptag_bench(out):
    p = os.path.join(R, "bench_toptag.json")
    if not os.path.exists(p):
        return
    d = json.load(open(p))
    d = [r for r in d if r["family"] != "les_cat"]
    base = {r["family"]: r["ms_per_step"] for r in d}.get("les")
    mbase = {r["family"]: r["peak_mem_mb"] for r in d}.get("les")
    rows = [[TT_LABEL[r["family"]], r["width"], f"{r['params']:,}", f"{r['ms_per_step'] / base:.1f}x", f"{r['peak_mem_mb'] / mbase:.1f}x"]
            for r in sorted(d, key=lambda r: r["ms_per_step"])]
    out["toptag_bench"] = md_table(["model", "width", "params", "step cost relative to LESNet", "peak memory relative to LESNet"], rows)


def table_fluid_seeds(out):
    a = _fluid("fluid_eval_v2_rollouts.json")
    b = a
    if not a or not b["runs"]:
        return
    ra = _fluid("fluid_rest_v2.json") or {}
    rb = {}
    for f in ("fluid_rest_seed1_les.json", "fluid_rest_seed1_gnn.json"):
        rb.update(_fluid(f) or {})
    rb.update(_fluid("fluid_rest_seeds23.json") or {})
    rows = []
    for fam, ns in (("LESNet (105k)", ("les_M4_d2", "les_M4_d2_s1", "les_M4_d2_s2", "les_M4_d2_s3")), ("graph network (107k)", ("gnn72_M4_d2", "gnn72_M4_d2_s1", "gnn72_M4_d2_s2", "gnn72_M4_d2_s3"))):
        for seed, name in enumerate(ns):
            ev, rs = a, (ra if seed == 0 else rb)
            if name not in ev["runs"]:
                continue
            r = ev["runs"][name]
            lp = os.path.join(HERE, "..", "fluid", "runs", name, "train_log.json")
            vl = json.load(open(lp))["hist"][-1][2] if os.path.exists(lp) else float("nan")
            q = rs.get(name, {}).get("speed_late_by_kind", {}).get("quiet", float("nan"))
            k1 = rs.get(name, {}).get("ke_ref", float("nan"))
            rows.append([fam, seed, f"{r['ke_ratio']:.2f}", f"{vl:.4f}", f"{q:.3f}", f"{k1:.2f}"])
    out["fluid_seeds"] = md_table(["model", "training seed", "kinetic-energy ratio at 0.5 s (all test trajectories)", "one-step val loss", "calm-down speed, quiet scenes (m/s)", "calm-down kinetic-energy ratio at 1 s"], rows)


def _noact_best():
    recs = load("noact")
    best = {}
    for lab in ("leaky ReLU (LESNet)", "no activation"):
        for t in TASK_ORDER:
            cand = {}
            for lr in sorted({r["lr"] for r in recs}):
                sub = [r for r in recs if r["label"] == lab and r["task"] == t and r["lr"] == lr]
                if sub:
                    cand[lr] = (np.mean([max(r["id"], -1.0) if r["id"] == r["id"] else -1.0 for r in sub]), sub)
            if cand:
                lr = max(cand, key=lambda k: cand[k][0])
                best[(lab, t)] = (lr, cand[lr][1])
    return best


def table_noact(out):
    best = _noact_best()
    if not best:
        return
    tasks = [t for t in TASK_ORDER if any(k[1] == t for k in best)]
    rows = []
    for lab in ("leaky ReLU (LESNet)", "no activation"):
        row = [lab]
        for t in tasks:
            lr, sub = best[(lab, t)]
            row.append(fmt(*mean_ci([r["id"] for r in sub]), 3))
        rows.append(row)
    out["noact"] = md_table(["variant"] + [f"`{t}`" for t in tasks], rows)


def fig_noact():
    best = _noact_best()
    if not best:
        return
    tasks = [t for t in TASK_ORDER if any(k[1] == t for k in best)]
    fig, axs = plt.subplots(1, len(tasks), figsize=(3.2 * len(tasks), 3.2), dpi=150)
    for ax, t in zip(np.atleast_1d(axs), tasks):
        for lab, col in (("leaky ReLU (LESNet)", "#d1495b"), ("no activation", "#8a93a6")):
            lr, sub = best[(lab, t)]
            C = np.array([[c[1] for c in r["curve"]] for r in sub]); S = [c[0] for c in sub[0]["curve"]]
            ax.plot(S, C.mean(0), "-o", ms=3, color=col, label=lab)
            ax.fill_between(S, C.min(0), C.max(0), color=col, alpha=0.18)
        ax.set_title(t, fontsize=9.5, color=INK); ax.set_xlabel("training step"); ax.grid(alpha=0.25)
    np.atleast_1d(axs)[0].set_ylabel("score"); np.atleast_1d(axs)[0].legend(frameon=False, fontsize=7.5)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "noact.png")); plt.close(fig)


def fig_toptag():
    """Test AUC against optimiser steps and against relative compute (idle-GPU step time x steps)."""
    p = os.path.join(R, "bench_toptag.json")
    recs = [r for r in _jsons("toptag_v2") if "test" in r and not r["name"].startswith(("smoke", "lr_")) and r["family"] != "les_cat"]
    if not recs or not os.path.exists(p):
        return
    ms = {r["family"]: r["ms_per_step"] for r in json.load(open(p))}
    fam = {"les": "LESNet", "gnn": "graph network", "transformer": "Transformer", "deepsets": "DeepSets", "gen": "GENConv"}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    groups = {}
    for r in recs:
        groups.setdefault((r["family"], r["steps"]), []).append(r)
    pts = []
    for (f, steps), sub in groups.items():
        t = steps * ms[f] / ms["les"] / 1000.0
        auc = [x["test"]["auc"] for x in sub]
        ci = (T975.get(len(auc) - 1, float("nan")) * float(np.std(auc, ddof=1)) / np.sqrt(len(auc))) if len(auc) > 1 else 0.0
        pts.append((f, steps, t, float(np.mean(auc)), ci))
    hs = {}
    for ax, xi, xl in ((axes[0], 1, "optimiser steps"), (axes[1], 2, "relative compute (thousand LESNet steps)")):
        for f, nm in fam.items():
            sel = sorted([p_ for p_ in pts if p_[0] == f], key=lambda p_: p_[xi])
            if not sel:
                continue
            c, mk = STYLE[nm]
            h = ax.errorbar([s_[xi] for s_ in sel], [s_[3] for s_ in sel], yerr=[s_[4] for s_ in sel], marker=mk, ls="-", color=c, capsize=2, lw=1.1)
            hs[{"Transformer": "Transformer + relation"}.get(nm, nm)] = h
        ax.axhline(0.9819, color="#8a93a6", ls=":", lw=0.9)
        ax.set_xscale("log"); ax.set_xlabel(xl); ax.set_ylabel("test AUC"); ax.grid(alpha=0.25, which="both")
        if xi == 1:
            ax.text(ax.get_xlim()[0] * (1.15 if xi == 2 else 1.0) if xi == 2 else ax.get_xlim()[1], 0.9817 if xi == 2 else 0.9821, "published Particle Flow Network", fontsize=6, color="#4a5368", va="top" if xi == 2 else "bottom", ha="left" if xi == 2 else "right")
    fig.legend(list(hs.values()), list(hs.keys()), loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "toptag.png")); plt.close(fig)


def table_hiddeneq(out):
    recs = load("hiddeneq")
    if not recs:
        return
    labels = []
    for r in recs:
        if r["label"] not in labels:
            labels.append(r["label"])
    tasks_ = [t for t in TASK_ORDER if any(r["task"] == t for r in recs)]
    rows = []
    for lab in labels:
        sub = [r for r in recs if r["label"] == lab]
        row = [lab, f"{int(np.mean([r['params'] for r in sub])):,}", sub[0]["width"], sub[0]["depth"]]
        for t in tasks_:
            row.append(fmt(*mean_ci([r["id"] for r in sub if r["task"] == t]), 3))
        rows.append(row)
    out["hiddeneq"] = md_table(["variant", "parameters", "width", "layers"] + [f"`{t}`" for t in tasks_], rows)


def fig_washout():
    recs = load("depth_washout")
    if not recs:
        return
    fams = [f for f in ("les", "les_cat", "gnn", "gnn_nores") if any(r["family"] == f for r in recs)]
    col = {"les": "#d1495b", "les_cat": "#e9a03b", "gnn": "#3a7d44", "gnn_nores": "#8fc29a"}
    fig, ax = plt.subplots(figsize=(5.4, 3.5), dpi=150)
    for f in fams:
        ds = sorted({r["depth"] for r in recs if r["family"] == f})
        ys = [mean_ci([r["acc"] for r in recs if r["family"] == f and r["depth"] == d]) for d in ds]
        ax.errorbar(ds, [y[0] for y in ys], yerr=[0 if y[1] != y[1] else y[1] for y in ys], fmt="o-", color=col[f], label=f, capsize=3, lw=1.6)
    ax.set_xscale("log", base=2); ax.set_xticks(sorted({r["depth"] for r in recs})); ax.set_xticklabels(sorted({r["depth"] for r in recs}))
    ax.set_xlabel("layers"); ax.set_ylabel("accuracy (3-hop pointer chasing)"); ax.grid(alpha=0.25); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "washout.png")); plt.close(fig)


def fig_multiswitch():
    recs = load("memswitch_multi")
    if not recs:
        return
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.4), dpi=150, sharey=False)
    for ax, w in zip(axs, sorted({r["width"] for r in recs})):
        for v, c in (("softmax+leaky", "#d1495b"), ("max+leaky", "#2e86ab")):
            sub = [r for r in recs if r["width"] == w and r["variant"] == v]
            if not sub:
                continue
            L = np.array([r["loss"] for r in sub]); S = np.array([r["scratch"] for r in sub])
            x = np.arange(1, L.shape[1] + 1)
            ax.plot(x, L.mean(0), "o-", color=c, label=f"{v}: switched")
            ax.plot(x, S.mean(0), "--", color=c, alpha=0.6, label=f"{v}: from scratch")
        ax.set_title(f"width {w}", fontsize=9.5); ax.set_xlabel("target index (new noise each time)"); ax.set_yscale("log"); ax.grid(alpha=0.25, which="both")
    axs[0].set_ylabel("loss reached"); axs[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "multiswitch.png")); plt.close(fig)

if __name__ == "__main__":
    write_tables()
    for f in (fig_ksweep, fig_speed, fig_fluid_scaling, fig_ood, fig_toptag):
        f()
    print("figures:", sorted(os.listdir(FIG)))