"""Train one model on one task for a fixed budget and evaluate it in-distribution and at larger N.

    run_one("nn_type", {"family": "les", "label": "LES"}, seed=0)  -> dict (one JSONL record)

Everything that could favour one model is held equal: same data stream (per seed), same batch, steps,
optimiser, schedule, readout head and parameter budget (width is fitted per model family).
"""
import json
import math
import os
import time

import torch

import models
import tasks

DEV = "cuda"
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)

N_LO, N_HI, NMAX = 8, 24, 24
OOD_NS = (48, 96)
_eval_cache = {}


def eval_set(task, n_lo, n_hi, nmax, count=2048, seed=777):
    key = (task, n_lo, n_hi, nmax, count)
    if key not in _eval_cache:
        gen = torch.Generator(device=DEV).manual_seed(seed + nmax)
        out = []
        for _ in range(count // 512):
            out.append(tasks.sample(task, 512, n_lo, n_hi, nmax, gen, DEV))
        _eval_cache[key] = out
    return _eval_cache[key]


@torch.no_grad()
def evaluate(net, task, n_lo, n_hi, nmax, count=2048):
    net.eval()
    vals = []
    bs = max(8, int(512 * (24.0 / nmax) ** 2))          # pair tensors grow as N^2: keep memory bounded
    for w in eval_set(task, n_lo, n_hi, nmax, count):
        for i in range(0, w["mask"].shape[0], bs):
            sub = {k: (v[i:i + bs] if torch.is_tensor(v) and v.dim() > 0 else v) for k, v in w.items()}
            vals.append(tasks.score(task, net(sub["x"], sub["coords"], sub["mask"]), sub))
    net.train()
    return sum(vals) / len(vals)


def baseline_of(task):
    w = eval_set(task, N_LO, N_HI, NMAX)[0]
    return tasks.majority_baseline(task, w)


def run_one(task, spec, seed=0, steps=3000, B=128, depth=3, params=150_000, width=None, lr=1e-3,
            ood=True, curve_points=10, wd=0.0, budget_s=None):
    """`budget_s` trains for a wall-clock budget instead of a step count: after 30 steps the step time is
    measured and the total (and the cosine schedule) is fixed from it, so slower models get fewer steps."""
    t_start = time.time()
    torch.manual_seed(seed)
    gen = torch.Generator(device=DEV).manual_seed(seed)
    if width is None:
        width = models.fit_width(spec, depth, NMAX, task, params)
    net = models.Net(models.build_trunk(spec, width, depth, NMAX), task).to(DEV)
    n_params = models.count_params(net)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    warm = 100
    horizon = {"steps": steps}
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / horizon["steps"])))))
    curve, train_t, diverged, run_loss = [], 0.0, False, 0.0
    marks = {int(steps * (i + 1) / curve_points) for i in range(curve_points)}
    step = 0
    while step < horizon["steps"]:
        step += 1
        steps = horizon["steps"]
        torch.cuda.synchronize(); t0 = time.time()
        w = tasks.sample(task, B, N_LO, N_HI, NMAX, gen, DEV)
        pred = net(w["x"], w["coords"], w["mask"])
        loss = tasks.loss_fn(task, pred, w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        torch.cuda.synchronize(); train_t += time.time() - t0
        if budget_s and step == 10:
            t10 = train_t                                  # steps 1-10 absorb CUDA warm-up; time the next 20
        if budget_s and step == 30:
            horizon["steps"] = steps = max(60, int(budget_s / ((train_t - t10) / 20)))
            marks = {int(steps * (i + 1) / curve_points) for i in range(curve_points)}
        if step % 50 == 0 or step == steps:
            lv = loss.item()
            if not math.isfinite(lv):
                diverged = True
                break
            run_loss = lv
        if step in marks:
            curve.append((step, evaluate(net, task, N_LO, N_HI, NMAX, 512)))
    rec = dict(task=task, label=spec.get("label", spec["family"]), spec={k: v for k, v in spec.items()},
               seed=seed, width=width, depth=depth, params=n_params, lr=lr, steps=horizon["steps"], batch=B, budget_s=budget_s,
               final_train_loss=run_loss, diverged=diverged, train_s=round(train_t, 2),
               ms_per_step=round(1000 * train_t / max(1, step), 3), curve=curve)
    if diverged:
        rec["id"] = float("nan")
    else:
        rec["id"] = evaluate(net, task, N_LO, N_HI, NMAX)
        rec["baseline"] = baseline_of(task)
        if ood and (task in tasks.LOCAL_TASKS or task == "closest_team"):
            for n in OOD_NS:
                try:
                    rec[f"ood{n}"] = evaluate(net, task, n, n, n, 1024)
                except Exception as e:  # noqa: BLE001 -- flat MLP cannot change N
                    rec[f"ood{n}"] = None
    rec["wall_s"] = round(time.time() - t_start, 1)
    return rec


def run_switch(spec, task_a, task_b, seed=0, steps_a=2000, steps_b=1500, B=128, depth=3, params=150_000, lr=1e-3):
    """Recoverability: train on task A, then switch to task B (same output shape) with a fresh optimiser.
    Returns the score on B after the switch next to a network trained on B from scratch for the same steps."""
    assert tasks.TASKS[task_a]["kind"] == tasks.TASKS[task_b]["kind"] and tasks.TASKS[task_a]["out"] == tasks.TASKS[task_b]["out"]

    def fit(net, task, steps, gen):
        opt = torch.optim.AdamW(net.parameters(), lr=lr)
        for s in range(steps):
            f = 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * s / steps))
            for g in opt.param_groups:
                g["lr"] = lr * min(1.0, (s + 1) / 100) * f
            w = tasks.sample(task, B, N_LO, N_HI, NMAX, gen, DEV)
            loss = tasks.loss_fn(task, net(w["x"], w["coords"], w["mask"]), w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

    width = models.fit_width(spec, depth, NMAX, task_a, params)
    out = dict(spec=spec, label=spec["label"], task_a=task_a, task_b=task_b, seed=seed, width=width, lr=lr)
    torch.manual_seed(seed)
    gen = torch.Generator(device=DEV).manual_seed(seed)
    net = models.Net(models.build_trunk(spec, width, depth, NMAX), task_a).to(DEV)
    fit(net, task_a, steps_a, gen)
    out["score_a"] = evaluate(net, task_a, N_LO, N_HI, NMAX)
    out["score_b_before"] = evaluate(net, task_b, N_LO, N_HI, NMAX)
    fit(net, task_b, steps_b, gen)
    out["score_b_after_switch"] = evaluate(net, task_b, N_LO, N_HI, NMAX)
    torch.manual_seed(seed)
    gen2 = torch.Generator(device=DEV).manual_seed(seed + 1)
    scratch = models.Net(models.build_trunk(spec, width, depth, NMAX), task_b).to(DEV)
    fit(scratch, task_b, steps_b, gen2)
    out["score_b_scratch"] = evaluate(scratch, task_b, N_LO, N_HI, NMAX)
    out["id"] = out["score_b_after_switch"]
    out.update(wall_s=0, params=models.count_params(net), steps=steps_b, ms_per_step=0)
    return out


def done_keys(path):
    keys = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    keys.add(r["key"])
                except Exception:  # noqa: BLE001
                    pass
    return keys


def append(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
