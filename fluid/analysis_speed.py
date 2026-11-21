"""How do the learned simulators treat fast and slow particles?  On the active held-out scenes, short rollouts (0.1 s and 0.25 s) from the native state;
particles are binned by their NATIVE speed at the horizon, and for every bin we report (i) the ratio of the learned mean speed to the native one
(1 = the right amplitude, < 1 = damped) and (ii) the velocity error relative to the native speed.  Also the kinetic-energy ratio over time.

    run.bat -u analysis_speed.py   ->  out/fluid_speed_bins.json  and  ../paper/figures/fluid_speed_bins.png
"""
import json
import os
import sys

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
RUNS = [(f"les_M4_d2{s}", f"LESNet, seed {k}", "#d1495b", ls) for k, (s, ls) in enumerate(zip(("", "_s1", "_s2", "_s3"), ("-", "--", ":", "-.")))] + \
       [(f"gnn72_M4_d2{s}", f"graph net, seed {k}", "#2e86ab", ls) for k, (s, ls) in enumerate(zip(("", "_s1", "_s2", "_s3"), ("-", "--", ":", "-.")))]
EDGES = [0.0, 0.3, 1.0, 2.0, 3.0, 99.0]
HOR = (0.10, 0.25)


@torch.no_grad()
def main():
    frame_dt = gen_data.STRIDE * sph.DT
    test = learned.Data("test", "data2")
    act = [tr for tr, k in zip(test.trajs, test.kinds) if k == "active"]
    res = {}
    for name, label, col, ls in RUNS:
        model, args = evaluate.load_model(name)
        M = args["M"]
        steps = int(0.5 / (M * frame_dt))
        sv_l = {h: [] for h in HOR}; sv_n = {h: [] for h in HOR}; err = {h: [] for h in HOR}; ke = []
        for tr in act:
            T = tr["pos"].shape[0]
            for t0 in np.linspace(5, T - 2 - steps * M, 6).astype(int):
                P, V = learned.rollout(model, tr, int(t0), steps, M)
                ref_v = tr["vel"][[int(t0) + i * M for i in range(steps + 1)]].to(learned.DEV)
                ke.append(((V ** 2).sum((1, 2)) / (ref_v ** 2).sum((1, 2)).clamp_min(1e-9)).cpu().numpy())
                for h in HOR:
                    i = int(round(h / (M * frame_dt)))
                    sn, sl = ref_v[i].norm(dim=-1), V[i].norm(dim=-1)
                    sv_n[h].append(sn.cpu()); sv_l[h].append(sl.cpu()); err[h].append((V[i] - ref_v[i]).norm(dim=-1).cpu())
        out = {"ke_ratio_t": (np.arange(steps + 1) * M * frame_dt).tolist(), "ke_ratio": np.mean(ke, 0).tolist(), "bins": {}}
        for h in HOR:
            sn, sl, er = torch.cat(sv_n[h]), torch.cat(sv_l[h]), torch.cat(err[h])
            rows = []
            for lo, hi in zip(EDGES[:-1], EDGES[1:]):
                m = (sn >= lo) & (sn < hi)
                rows.append(dict(lo=lo, hi=hi, n=int(m.sum()), speed_ratio=float(sl[m].mean() / sn[m].mean()) if m.any() else None,
                                 rel_err=float(er[m].mean() / sn[m].mean()) if m.any() else None))
            out["bins"][str(h)] = rows
        res[name] = out
        print(name, {h: [round(r["speed_ratio"], 2) if r["speed_ratio"] else None for r in out["bins"][str(h)]] for h in HOR}, flush=True)
    json.dump(res, open(os.path.join(HERE, "out", "fluid_speed_bins.json"), "w"), indent=1)
    plot(res)


def plot(res):
    sys.path.insert(0, os.path.join(HERE, "..", "paper"))
    from style import STYLE
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.5))
    names = [f"{lo:g}-{hi:g}" if hi < 90 else f">{lo:g}" for lo, hi in zip(EDGES[:-1], EDGES[1:])]
    groups = (("les_M4_d2", "LESNet", "LESNet"), ("gnn72_M4_d2", "graph network", "graph network"))
    for prefix, label, fam in groups:
        col, mk = STYLE[fam]
        runs = [n for n in res if n.startswith(prefix)]
        for ax, h in zip(axs[:2], HOR):
            vals = np.array([[r["speed_ratio"] if r["speed_ratio"] is not None else np.nan for r in res[n]["bins"][str(h)]][:-1] for n in runs], float)
            m = np.nanmean(vals, 0)
            ax.errorbar(range(len(m)), m, yerr=[m - np.nanmin(vals, 0), np.nanmax(vals, 0) - m], color=col, marker=mk, capsize=2, label=label)
        t = np.array(res[runs[0]]["ke_ratio_t"])
        ke = np.array([res[n]["ke_ratio"] for n in runs], float)
        axs[2].plot(t, ke.mean(0), color=col, label=label)
        axs[2].fill_between(t, ke.min(0), ke.max(0), color=col, alpha=0.2)
    for ax, h, tag in zip(axs[:2], HOR, "ab"):
        ax.axhline(1.0, color="#888", lw=0.8, ls=":"); ax.set_xticks(range(len(names) - 1)); ax.set_xticklabels(names[:-1])
        ax.set_xlabel("native particle speed (m/s)"); ax.set_ylabel("learned speed / native speed"); ax.set_title(f"({tag}) after {h:g} s", loc="left"); ax.grid(alpha=0.25)
    axs[2].legend(frameon=False, loc="upper left")
    axs[2].axhline(1.0, color="#888", lw=0.8, ls=":"); axs[2].set_xlabel("simulated time (s)"); axs[2].set_ylabel("kinetic energy, learned / native"); axs[2].grid(alpha=0.25)
    axs[2].set_title("(c) active scenes", loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "..", "paper", "figures", "fluid_speed_bins.png"), dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    if "--plot" in sys.argv:
        plot(json.load(open(os.path.join(HERE, "out", "fluid_speed_bins.json"))))
    else:
        main()
