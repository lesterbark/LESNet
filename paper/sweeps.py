"""Resumable sweeps.  Every finished run is one JSON line in results/<sweep>.jsonl, keyed so a rerun
skips what is already there.

    run.bat -u sweeps.py pilot
    run.bat -u sweeps.py lr_tune
    run.bat -u sweeps.py main
    run.bat -u sweeps.py ablate
    run.bat -u sweeps.py capacity
    run.bat -u sweeps.py ksweep
"""
import json
import os
import sys
import time

import train

R = train.RESULTS
ALL_TASKS = ["nn_type", "count_r", "rank_ball", "closest_team", "exist_pair", "local_vel_avg", "nbody_acc"]

PF = "pos,axis,polar"          # relation given to every model that can use one (LES, GNN, Transformer+pf)
MAIN_SPECS = [
    {"family": "les", "label": "LES", "pf_spec": PF},
    {"family": "les", "label": "LES-kNN8", "pf_spec": PF, "k": 8},
    {"family": "mlp", "label": "MLP-flat"},
    {"family": "deepsets", "label": "DeepSets"},
    {"family": "transformer", "label": "Transformer"},
    {"family": "transformer", "label": "Transformer+pf", "pf_spec": PF},
    {"family": "gnn", "label": "GNN", "pf_spec": PF},
    {"family": "gnn", "label": "GNN-kNN8", "pf_spec": PF, "k": 8},
]
# learning rate is tuned once per architecture family; variants inherit it
LR_REP = {"LES-kNN8": "LES", "Transformer+pf": "Transformer", "GNN-kNN8": "GNN"}
TUNE_LABELS = ["LES", "MLP-flat", "DeepSets", "Transformer", "GNN"]
LRS = [3e-4, 1e-3, 3e-3]


def lr_for(label, default=1e-3):
    p = os.path.join(R, "lr_best.json")
    label = LR_REP.get(label, label)
    if os.path.exists(p):
        return json.load(open(p)).get(label, default)
    return default


def execute(name, jobs, shard=None):
    """Run the jobs whose key is not already in any results/<name>*.jsonl; write to <name>[_<shard>].jsonl."""
    import glob
    suffix = ("_" + shard) if shard else ""
    path = os.path.join(R, f"{name}{suffix}.jsonl")
    done = set()
    for p in glob.glob(os.path.join(R, f"{name}.jsonl")) + glob.glob(os.path.join(R, f"{name}_s[0-9].jsonl")):
        done |= train.done_keys(p)
    todo = [j for j in jobs if j["key"] not in done]
    print(f"[{name}] {len(jobs)} jobs, {len(done)} done, {len(todo)} to run", flush=True)
    t0 = time.time()
    for i, j in enumerate(todo):
        rec = train.run_one(j["task"], j["spec"], seed=j["seed"], **j["kw"])
        rec["key"] = j["key"]
        rec["sweep"] = name
        rec.update(j.get("extra", {}))
        train.append(path, rec)
        el = time.time() - t0
        eta = el / (i + 1) * (len(todo) - i - 1)
        print(f"[{name}] {i + 1}/{len(todo)} {j['task']:13s} {rec['label']:18s} s{j['seed']} lr={rec['lr']:g} "
              f"params={rec['params']:>7d} id={rec['id']:.3f} {rec['wall_s']:.0f}s  elapsed {el / 60:.1f}m eta {eta / 60:.1f}m",
              flush=True)


def job(task, spec, seed, extra_key="", extra=None, **kw):
    kw.setdefault("steps", 3000)
    kw.setdefault("params", 150_000)
    kw.setdefault("depth", 3)
    kw.setdefault("lr", lr_for(spec["label"]))
    key = f"{task}|{spec['label']}|{seed}|{kw['steps']}|{kw['params']}|{kw['depth']}|{kw['lr']:g}|{kw.get('width')}|{extra_key}"
    return dict(task=task, spec=spec, seed=seed, kw=kw, key=key, extra=extra or {})


def sweep_pilot():
    specs = [s for s in MAIN_SPECS if s["label"] in ("LES", "LES-kNN8", "Transformer+pf", "GNN")]
    jobs = [job(t, s, 0, lr=1e-3) for t in ("nn_type", "count_r", "local_vel_avg", "nbody_acc") for s in specs]
    execute("pilot", jobs)


def sweep_pilot2():
    """Does giving LESNet the distance/direction relation (polar) close the gap?  More steps?"""
    specs = [{"family": "les", "label": "LES-polar", "pf_spec": "pos,axis,polar"},
             {"family": "les", "label": "LES-polar-kNN8", "pf_spec": "pos,axis,polar", "k": 8},
             {"family": "gnn", "label": "GNN-polar", "pf_spec": "pos,axis,polar"}]
    jobs = [job(t, s, 0, lr=1e-3) for t in ("nn_type", "count_r", "local_vel_avg") for s in specs]
    jobs += [job(t, specs[0], 0, lr=1e-3, steps=9000, extra_key="long") for t in ("nn_type", "count_r")]
    execute("pilot2", jobs)


def pick_best_lr():
    """Per label, the lr with the best mean score over the tuning tasks (all lrs run so far)."""
    recs = [json.loads(l) for l in open(os.path.join(R, "lr_tune.jsonl"))]
    best = {}
    for label in {r["label"] for r in recs}:
        score = {}
        for lr in sorted({r["lr"] for r in recs if r["label"] == label}):
            v = [max(r["id"], -1.0) for r in recs if r["label"] == label and r["lr"] == lr and r["id"] == r["id"]]
            score[lr] = sum(v) / len(v) if v else -9
        best[label] = max(score, key=score.get)
    json.dump(best, open(os.path.join(R, "lr_best.json"), "w"), indent=1)
    return best


def sweep_lr_tune():
    jobs = [job(t, s, 0, lr=lr) for t in ("nn_type", "closest_team", "local_vel_avg")
            for s in MAIN_SPECS if s["label"] in TUNE_LABELS for lr in LRS]
    execute("lr_tune", jobs)
    print("lr_best:", pick_best_lr())


def sweep_lr_tune3():
    """LES and DeepSets were still improving at 1e-2: one more step up (3e-2) for them."""
    jobs = [job(t, s, 0, lr=3e-2) for t in ("nn_type", "closest_team", "local_vel_avg")
            for s in MAIN_SPECS if s["label"] in ("LES", "DeepSets")]
    execute("lr_tune", jobs)
    print("lr_best:", pick_best_lr())


def sweep_lr_tune4():
    """LES and DeepSets were still improving at 3e-2: one more step up (1e-1)."""
    jobs = [job(t, s, 0, lr=1e-1) for t in ("nn_type", "closest_team", "local_vel_avg")
            for s in MAIN_SPECS if s["label"] in ("LES", "DeepSets")]
    execute("lr_tune", jobs)
    print("lr_best:", pick_best_lr())


GEN_SPEC = {"family": "gen", "label": "GENConv", "pf_spec": PF}


def sweep_gen_tune():
    """Softmax-aggregation message passing (DeeperGCN style) baseline: learning-rate grid on the three tuning tasks."""
    jobs = [job(t, GEN_SPEC, 0, lr=lr) for t in ("nn_type", "closest_team", "local_vel_avg") for lr in (1e-3, 3e-3, 1e-2, 3e-2)]
    execute("lr_tune", jobs)
    print("lr_best:", pick_best_lr())


def sweep_gen_main():
    """Seven tasks, three seeds, at the tuned rate; written as shard s7 of the main sweep so the tables pick it up."""
    jobs = [job(t, GEN_SPEC, sd) for sd in (0, 1, 2) for t in ALL_TASKS]
    execute("main", jobs, shard="s7")


def lr_extension():
    """If a family's best learning rate sits on the edge of the grid, extend the grid one step past it
    (an optimum on the boundary has not been found)."""
    best = pick_best_lr()
    hi, lo = max(LRS), min(LRS)
    jobs = []
    for s in MAIN_SPECS:
        if s["label"] not in TUNE_LABELS:
            continue
        b = best.get(s["label"])
        extra = [1e-2] if b == hi else [1e-4] if b == lo else []
        for lr in extra:
            jobs += [job(t, s, 0, lr=lr) for t in ("nn_type", "closest_team", "local_vel_avg")]
    if jobs:
        print("extending the lr grid for:", sorted({j["spec"]["label"] for j in jobs}), flush=True)
        execute("lr_tune", jobs)
        print("lr_best after extension:", pick_best_lr(), flush=True)

def sweep_main(seeds=(0, 1, 2)):
    """`sweeps.py main 1` runs only seed 1 (and writes main_s1.jsonl), so seeds can run as parallel processes."""
    if len(sys.argv) > 2:
        seeds = tuple(int(a) for a in sys.argv[2:])
    lr_extension()
    jobs = [job(t, s, sd) for sd in seeds for t in ALL_TASKS for s in MAIN_SPECS]
    execute("main", jobs, shard=f"s{seeds[0]}" if len(seeds) == 1 else None)


def sweep_ablate(seeds=(0, 1)):
    tasks = ["nn_type", "count_r", "closest_team", "nbody_acc"]
    base = {"family": "les_torch"}
    variants = [
        dict(label="softmax (LES)", **base),
        dict(label="sum", combine="sum", **base),
        dict(label="mean", combine="mean", **base),
        dict(label="max", combine="max", **base),
        dict(label="scalar-sum", combine="scalar", **base),
        dict(label="logsumexp", combine="lse", **base),
        dict(label="separate score", sep=True, **base),
        dict(label="hidden pair layer", hidden=True, **base),
        dict(label="ReLU", act="relu", **base),
        dict(label="tanh", act="tanh", **base),
        dict(label="no pair features", pf_spec="none", **base),
        dict(label="pf: pos,axis,polar", pf_spec="pos,axis,polar", **base),
    ]
    # smaller than the main sweep (the PyTorch twin materialises the N^2 W pair tensor): 50k params, 2500 steps
    jobs = [job(t, v, sd, lr=1e-3, params=50_000, steps=2000, B=64, extra={"ablation": True})
            for sd in seeds for t in tasks for v in variants]
    execute("ablate", jobs)


def abl_variants():
    base = {"family": "les_torch"}
    return [
        dict(label="softmax (LES)", **base),
        dict(label="sum", combine="sum", **base),
        dict(label="mean", combine="mean", **base),
        dict(label="max", combine="max", **base),
        dict(label="scalar-sum", combine="scalar", **base),
        dict(label="separate score", sep=True, **base),
        dict(label="hidden pair layer", hidden=True, **base),
        dict(label="ReLU", act="relu", **base),
        dict(label="no pair features", pf_spec="none", **base),
        dict(label="pf: pos,axis,polar", pf_spec="pos,axis,polar", **base),
    ]


def sweep_ablate2(seeds=(0, 1)):
    """Ablations with a learning rate tuned PER VARIANT (every variant gets its best of 3), on smaller sets
    (8-16 entities) so the pair tensor the PyTorch twin materialises stays cheap."""
    train.N_HI, train.NMAX = 16, 16
    V = abl_variants()
    common = dict(params=50_000, steps=2000, B=64, ood=False)
    # stage 1: per-variant learning rate on nn_type, seed 0
    execute("ablate2_lr", [job("nn_type", v, 0, lr=lr, extra_key="abl2lr", **common) for v in V for lr in (1e-3, 3e-3, 1e-2)])
    recs = [json.loads(l) for l in open(os.path.join(R, "ablate2_lr.jsonl"))]
    best = {}
    for v in V:
        rs = [r for r in recs if r["label"] == v["label"]]
        best[v["label"]] = max(rs, key=lambda r: max(r["id"], -1.0) if r["id"] == r["id"] else -9)["lr"]
    json.dump(best, open(os.path.join(R, "ablate2_lr_best.json"), "w"), indent=1)
    print("ablation learning rates:", best, flush=True)
    tasks = ["nn_type", "count_r", "closest_team"]
    execute("ablate2", [job(t, v, sd, lr=best[v["label"]], extra_key="abl2", extra={"ablation": True}, **common)
                        for sd in seeds for t in tasks for v in V])


def matched_width(w_hidden, n=12, p=7):
    """Width of a plain (no hidden layer) LESNet layer whose analytic cost (dense, n entities) equals that of a layer
    with width w_hidden and a hidden pair layer: the hidden layer adds a W x W product (2 W^2 FLOPs) per pair."""
    import flops
    pairs = n * (n - 1)
    target = flops.les(n, w_hidden, w_hidden, p) + pairs * 2 * w_hidden * w_hidden
    lo, hi = w_hidden, 8 * w_hidden
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if flops.les(n, mid, mid, p) < target:
            lo = mid
        else:
            hi = mid
    return int(round(lo / 4.0)) * 4


def sweep_hiddeneq(seeds=(0, 1)):
    """Is the hidden pair layer worth its compute?  At equal parameters it is the largest gain (ablate2); here it is
    compared with a plain LESNet widened until the analytic cost per layer is the same (and with the plain
    parameter-matched LESNet), per the argument that spending the cost on width is the better use."""
    train.N_HI, train.NMAX = 16, 16
    import models
    base = {"family": "les_torch"}
    hid = dict(label="hidden pair layer (50k params)", hidden=True, **base)
    w_h = models.fit_width(hid, 3, 16, "nn_type", 50_000)
    w_m = matched_width(w_h)
    variants = [(dict(label="plain LES (50k params)", **base), None),
                (hid, None),
                (dict(label=f"plain LES, same FLOPs as hidden (width {w_m})", **base), w_m)]
    print(f"hidden-layer width {w_h}; FLOPs-matched plain width {w_m}", flush=True)
    jobs = []
    for sd in seeds:
        for t in ("nn_type", "count_r", "closest_team"):
            for spec, w in variants:
                kw = dict(lr=1e-2, params=50_000, steps=2000, B=64, ood=False, extra_key="hiddeneq", extra={"width_used": w or 0})
                if w:
                    kw["width"] = w
                jobs.append(job(t, spec, sd, **kw))
    execute("hiddeneq", jobs)


def sweep_hiddeneq2(seeds=(0, 1)):
    """Follow-up to hiddeneq: widening a plain LESNet to the hidden layer's FLOPs did not recover the gain, so spend the same
    FLOPs on DEPTH instead (plain LESNet with 6 and 9 layers, width solved so the total analytic cost equals that of the
    3-layer net with a hidden pair layer)."""
    train.N_HI, train.NMAX = 16, 16
    import flops
    import models
    base = {"family": "les_torch"}
    hid = dict(label="hidden pair layer (50k params)", hidden=True, **base)
    w_h = models.fit_width(hid, 3, 16, "nn_type", 50_000)
    n, p = 12, 7
    target = 3 * (flops.les(n, w_h, w_h, p) + n * (n - 1) * 2 * w_h * w_h)
    variants = []
    for d in (6, 9):
        lo, hi = 8.0, 512.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if d * flops.les(n, mid, mid, p) < target:
                lo = mid
            else:
                hi = mid
        w = int(round(lo / 4.0)) * 4
        variants.append((dict(label=f"plain LES, {d} layers, same FLOPs as hidden (width {w})", **base), w, d))
    print("depth variants:", [(v[0]["label"], v[1], v[2]) for v in variants], flush=True)
    jobs = []
    for sd in seeds:
        for t in ("nn_type", "count_r", "closest_team"):
            for spec, w, d in variants:
                jobs.append(job(t, spec, sd, lr=1e-2, params=50_000, steps=2000, B=64, ood=False, width=w, depth=d,
                                extra_key="hiddeneq2", extra={"width_used": w}))
    execute("hiddeneq", jobs)             # same results file: it is the same experiment


def sweep_noact(seeds=(0, 1, 2)):
    """The layer with and without its activation (identity in place of the leaky ReLU, in the pair filter and after the combine):
    does the nonlinearity matter?  Small sets (8-16 entities), 50k parameters, 2000 steps, learning rates 3e-3 and 1e-2 for both."""
    train.N_HI, train.NMAX = 16, 16
    base = {"family": "les_torch"}
    variants = [dict(label="leaky ReLU (LESNet)", **base), dict(label="no activation", act="none", **base)]
    jobs = [job(t, v, sd, lr=lr, params=50_000, steps=2000, B=64, ood=False, extra_key="noact", extra={"noact": True})
            for sd in seeds for t in ("nn_type", "count_r", "closest_team", "local_vel_avg") for v in variants for lr in (3e-3, 1e-2)]
    execute("noact", jobs)


def sweep_ablate_lr(seeds=(0, 1)):
    """The combine functions again at two other learning rates: a combine that looks worse at one lr may
    just want a different one, and the paper's central claim is about the combine."""
    base = {"family": "les_torch"}
    variants = [dict(label="softmax (LES)", **base), dict(label="sum", combine="sum", **base),
                dict(label="mean", combine="mean", **base), dict(label="max", combine="max", **base),
                dict(label="scalar-sum", combine="scalar", **base), dict(label="logsumexp", combine="lse", **base)]
    jobs = [job("nn_type", v, sd, lr=lr, params=50_000, steps=2000, B=64, extra={"ablation": True})
            for sd in seeds for lr in (3e-4, 3e-3) for v in variants]
    execute("ablate_lr", jobs)


def sweep_capacity(seeds=(0,)):
    spec = {"family": "les", "label": "LES", "pf_spec": PF}
    jobs = []
    for sd in seeds:
        for t in ("nn_type", "closest_team", "local_vel_avg"):
            for w in (32, 64, 128, 256):
                for d in (1, 2, 4):
                    jobs.append(job(t, spec, sd, width=w, depth=d, extra={"w": w, "d": d}, extra_key=f"w{w}d{d}"))
    execute("capacity", jobs)


def sweep_ksweep(seeds=(0, 1)):
    jobs = []
    for sd in seeds:
        for t in ("nn_type", "count_r", "local_vel_avg"):
            for k in (1, 2, 4, 8, 16):
                jobs.append(job(t, {"family": "les", "label": f"LES-kNN{k}", "k": k, "pf_spec": PF}, sd, lr=lr_for("LES"), extra={"k": k}))
            jobs.append(job(t, {"family": "les", "label": "LES-all", "pf_spec": PF}, sd, lr=lr_for("LES"), extra={"k": 0}))
    execute("ksweep", jobs)


def sweep_scale(seeds=(0, 1)):
    """Parameter scaling: does the ranking change with model size?"""
    fams = [s for s in MAIN_SPECS if s["label"] in ("LES", "Transformer+pf", "GNN")]
    jobs = [job(t, s, sd, params=p, extra_key=f"p{p}", extra={"target_params": p})
            for sd in seeds for t in ("nn_type",) for s in fams for p in (30_000, 150_000, 600_000)]
    execute("scale", jobs)


def sweep_initsens(seeds=(0, 1)):
    """Is the soft-max combine less sensitive to initialisation than the alternatives?"""
    jobs = []
    for sd in seeds:
        for t in ("nn_type",):
            for comb in ("softmax", "sum", "max"):
                for init in (0.1, 1.0, 10.0):
                    spec = {"family": "les_torch", "label": f"{comb} init x{init:g}", "combine": comb, "init": init}
                    jobs.append(job(t, spec, sd, lr=1e-3, params=50_000, steps=2000, B=64, extra={"combine": comb, "init": init}))
    execute("initsens", jobs)


def sweep_isotime(seeds=(0,), budget_s=60):
    """Equal wall-clock training budget instead of equal steps (run with the GPU to itself)."""
    labels = ("LES", "LES-kNN8", "Transformer+pf", "GNN", "GNN-kNN8")
    specs = [s for s in MAIN_SPECS if s["label"] in labels]
    jobs = [job(t, s, sd, budget_s=budget_s, extra_key=f"iso{budget_s}", extra={"iso": True})
            for sd in seeds for t in ("nn_type", "closest_team", "local_vel_avg") for s in specs]
    execute("isotime", jobs)


def sweep_switch(seeds=(0, 1)):
    """Recoverability after a task switch (same output shape): soft-max vs hard max vs sum."""
    path = os.path.join(R, "switch.jsonl")
    done = train.done_keys(path)
    for sd in seeds:
        for comb in ("softmax", "max", "sum"):
            for a, b in (("local_vel_avg", "nbody_acc"), ("nbody_acc", "local_vel_avg")):
                key = f"{comb}|{a}|{b}|{sd}"
                if key in done:
                    continue
                spec = {"family": "les_torch", "label": comb, "combine": comb}
                rec = train.run_switch(spec, a, b, seed=sd)
                rec["key"] = key
                train.append(path, rec)
                print(f"[switch] {comb:8s} {a}->{b} s{sd}  after={rec['score_b_after_switch']:.3f} scratch={rec['score_b_scratch']:.3f}", flush=True)


if __name__ == "__main__":
    name = sys.argv[1]
    globals()[f"sweep_{name}"]()
