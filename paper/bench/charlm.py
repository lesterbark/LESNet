"""Character-level language modelling (Tiny Shakespeare).  LESNet reads each position's previous K characters as its partner list, with the
offset to each partner as the relation (no positional encoding); the baseline is a small GPT (pre-LN, learned positions) of the same
parameter count.  Metric: validation bits per character.  Not a regime LESNet is designed for; included to show how it behaves there.

    run.bat -u bench\charlm.py [--sweep] [--steps 5000] [--seeds 3]
"""
import argparse
import json
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402

T, K, B = 256, 64, 64


def load():
    txt = open(os.path.join(lib.HERE, "data", "shakespeare.txt"), encoding="utf-8").read()
    chars = sorted(set(txt))
    ids = torch.tensor([chars.index(c) for c in txt], device=lib.DEV)
    n = int(0.9 * len(ids))
    return ids[:n], ids[n:], len(chars)


def offsets_pf():
    d = torch.arange(1, K + 1).float()
    f = [d / K, torch.log(d) / math.log(K), 1.0 / d, torch.sin(2 * math.pi * d / 8), torch.cos(2 * math.pi * d / 8),
         torch.sin(2 * math.pi * d / 32), torch.cos(2 * math.pi * d / 32)]
    return torch.stack(f, 1).to(lib.DEV)                         # (K, 7)


class LESLM(nn.Module):
    def __init__(self, vocab, width, depth):
        super().__init__()
        self.emb = nn.Embedding(vocab, width)
        self.net = lib.LESList(width, width, depth, 7)
        self.head = nn.Linear(width, vocab)
        self.pf_d = offsets_pf()
        self._rows = {}

    def lists(self, b):
        if b not in self._rows:
            t = torch.arange(T, device=lib.DEV)[:, None]
            j = t - torch.arange(1, K + 1, device=lib.DEV)[None]           # partner position, (T,K)
            rows = (torch.arange(b, device=lib.DEV)[:, None, None] * T + j[None]).where(j[None] >= 0, torch.full_like(j[None], -1))
            self._rows[b] = (rows.reshape(b * T, K).to(torch.int32).contiguous(), self.pf_d[None].expand(b * T, -1, -1).contiguous())
        return self._rows[b]

    def forward(self, idx):
        b = idx.shape[0]
        rows, pf = self.lists(b)
        h = self.net(self.emb(idx).reshape(b * T, -1), rows, pf)
        return self.head(h).view(b, T, -1)


class GPT(nn.Module):
    def __init__(self, vocab, d, layers, heads=4):
        super().__init__()
        self.tok, self.pos = nn.Embedding(vocab, d), nn.Embedding(T, d)
        blk = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True, activation="gelu")
        self.blocks = nn.TransformerEncoder(blk, layers, enable_nested_tensor=False)
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, vocab)
        self.register_buffer("mask", torch.triu(torch.ones(T, T, dtype=torch.bool), 1), persistent=False)

    def forward(self, idx):
        h = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        return self.head(self.ln(self.blocks(h, mask=self.mask, is_causal=True)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--target", type=int, default=800_000)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--fam", default="les,gpt")
    ap.add_argument("--sweep_lrs", default="3e-4,1e-3,3e-3,1e-2")
    a = ap.parse_args()
    tr, va, V = load()
    les_w = lib.fit_width(lambda w: LESLM(V, w, a.depth), a.target, lo=16, hi=512, step=4)
    gpt_d = lib.fit_width(lambda w: GPT(V, w, a.depth), a.target, lo=16, hi=512, step=4)
    print("LESNet width", les_w, lib.n_params(LESLM(V, les_w, a.depth)), "| GPT width", gpt_d, lib.n_params(GPT(V, gpt_d, a.depth)), flush=True)

    def batcher(g, data):
        def mb(s):
            i = torch.randint(0, len(data) - T - 1, (B,), device=lib.DEV, generator=g)
            ar = torch.arange(T + 1, device=lib.DEV)
            blk = data[i[:, None] + ar[None]]
            return blk[:, :-1], blk[:, 1:]
        return mb

    def evalf(model):
        g = torch.Generator(device=lib.DEV).manual_seed(123)
        mb, tot = batcher(g, va), 0.0
        for k in range(20):
            x, y = mb(k)
            tot += F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1)).item()
        return dict(bpc=tot / 20 / math.log(2))

    for fam, mk in (("les", lambda: LESLM(V, les_w, a.depth)), ("gpt", lambda: GPT(V, gpt_d, a.depth))):
        if fam not in a.fam.split(","):
            continue
        if a.sweep:
            lrs, seeds, steps = tuple(float(x) for x in a.sweep_lrs.split(",")), range(1), 3000
        else:
            import glob
            best = {}
            for pth in glob.glob(os.path.join(lib.RES, f"charlm_sweep_{fam}_lr*_s0.json")):
                dd = json.load(open(pth))
                best[dd["lr"]] = min(c["bpc"] for c in dd["curve"])   # lowest validation bpc, the metric that is reported
            lrs, seeds, steps = (min(best, key=best.get),), range(a.seeds), a.steps
        for lr in lrs:
            for seed in seeds:
                torch.manual_seed(seed)
                g = torch.Generator(device=lib.DEV).manual_seed(seed)
                model = mk().to(lib.DEV)
                name = f"charlm_{'sweep_' if a.sweep else ''}{fam}_lr{lr:g}_s{seed}"
                lib.run(name, model, batcher(g, tr), lambda m, b_: F.cross_entropy(m(b_[0]).reshape(-1, V), b_[1].reshape(-1)), evalf,
                        steps=steps, lr=lr, eval_every=250 if not a.sweep else 500, wd=0.01, extra=dict(family=fam, seed=seed, task="charlm", tokens_per_step=B * T))


if __name__ == "__main__":
    main()
