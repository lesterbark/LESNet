"""Benchmark a LESNet training step (forward + backward + Adam) on this GPU.

    python bench.py                          # 256 sets x 32 entities, width 256, depth 5
    python bench.py --N 8 16 32 64 128 256   # sweep the set size
    python bench.py --cin 64 --width 512 --depth 8
    python bench.py --baselines              # parameter-matched MLP and attention stacks
    python bench.py --iters 20 --warmup 8

The data is synthetic: every entity gets `cin` features and a position in the unit square, which
is all the architecture asks for -- there is nothing task-specific here.

Timings are ms/step.  Run with nothing else on the GPU for absolute numbers; ratios stay
meaningful under contention.
"""
import argparse, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import kernels  # noqa: E402
import ref  # noqa: E402
from model import LESNet  # noqa: E402


def synth(B, N, cin, dev, seed=0, masked=0):
    """A batch of B sets, each with N slots, `cin` features per entity and a position.

    `masked` blanks the last few slots of every set, which is the interesting case: N is a ceiling
    rather than a requirement, so a set that uses only part of it should not pay for the rest.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(B, N, cin, generator=g).to(dev)
    coords = (torch.rand(B, N, 2, generator=g) * 2 - 1).to(dev)
    mask = torch.ones(B, N, dtype=torch.bool, device=dev)
    if masked:
        mask[:, -masked:] = False
    return x, coords, mask


# ----------------------------------------------------------------------------------- baselines
class MLPBlock(nn.Module):
    """Per-entity MLP: every entity is processed entirely alone.

    This is the speed floor -- there is no pair term, no softmax, nothing that scales with N^2 --
    and it is also the capability floor, because an entity cannot see any other entity.  It is the
    number to beat on time and the reason LESNet will never quite reach it.
    """
    def __init__(self, cin, width):
        super().__init__()
        self.fc1 = nn.Linear(cin, width)
        self.fc2 = nn.Linear(width, width)

    def forward(self, x, coords=None, mask=None, pf=None):
        return F.leaky_relu(self.fc2(F.leaky_relu(self.fc1(x), 0.1)), 0.1)


class AttnBlock(nn.Module):
    """A transformer block over the set: qkv attention plus an FFN, with residuals and LayerNorm.

    The natural competitor, and the honest one to compare against: same principle (every entity
    looks at every other entity), standard implementation.  Its cost per block is ~4*W^2 against
    LESNet's 2*cin*W + PF*W, which is why the baseline widths here are matched to LESNet's
    *parameter count* rather than to its width -- otherwise the comparison is not a comparison.
    """
    def __init__(self, cin, width):
        super().__init__()
        self.norm1 = nn.LayerNorm(cin)
        self.qkv = nn.Linear(cin, 3 * width)
        self.proj = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.net = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, x, coords=None, mask=None, pf=None):
        B, N, _ = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        keep = mask[:, None, None, :].expand(B, 1, N, N)          # transformers keep self
        h = F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None], attn_mask=keep)
        x = x + self.proj(h[:, 0])
        x = x + self.net(self.norm2(x))
        return x * mask.unsqueeze(-1)


class Stack(nn.Module):
    """`depth` copies of one block: the first widens from cin, the rest stay at `width` -- the same
    shape of model LESNet is, so the comparison is like for like."""
    def __init__(self, make, cin, width, depth, proj_in=False):
        super().__init__()
        # a residual block cannot change width, so a transformer needs an input projection; the
        # MLP's first layer already widens from cin, so it needs none
        self.proj_in = nn.Linear(cin, width) if (proj_in and cin != width) else nn.Identity()
        self.blocks = nn.ModuleList([make(max(cin, width) if proj_in else cin)]
                                    + [make(width) for _ in range(depth - 1)])

    def forward(self, x, coords=None, mask=None, pf=None):
        x = self.proj_in(x)
        for b in self.blocks:
            x = b(x, coords, mask, pf)
        return x


def build_baseline(kind, cin, width, depth):
    if kind == "mlp":
        return Stack(lambda c: MLPBlock(c, width), cin, width, depth)
    if kind == "attn":
        return Stack(lambda c: AttnBlock(c, width), cin, width, depth, proj_in=True)
    raise ValueError(kind)


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def match_width(kind, cin, depth, target, lo=8, hi=2048):
    """Smallest-width ``w`` whose baseline has at most `target` parameters, scanning by bisection.

    Parameter count grows quickly with width here, so a few iterations land within a few percent;
    the exact figure is printed alongside so the comparison can be judged for what it is.
    """
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        if nparams(build_baseline(kind, cin, mid, depth)) <= target:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best


# -------------------------------------------------------------------------------------- timing
def make_step(blocks, out_width, data, dev):
    """One training step (forward + backward + Adam) as a no-argument closure.

    Every model gets the same treatment: the block stack, then a masked mean over the set, then one
    linear head, then MSE against a fixed target.  The head is inside the timed region and counted
    in the parameter total, so nothing is hidden from the comparison.  Building the closure rather
    than timing it here is what lets several models be interleaved against the same contention.
    """
    x, coords, mask = data
    head = nn.Linear(out_width, 1).to(dev)
    target = torch.zeros(x.shape[0], device=dev)
    opt = torch.optim.Adam(list(blocks.parameters()) + list(head.parameters()), lr=1e-4, fused=True)

    def step():
        y = blocks(x, coords, mask)
        loss = ((head(ref.masked_mean(y, mask)) - target) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    return step


def round_robin(fns, iters=15, warmup=4, windows=5, seed=0):
    """ms/step per candidate, min over windows, with the measurement order shuffled each window.

    Timing candidates one after another in fixed order is not a comparison -- whichever runs first
    inherits the previous one's cache state, and any other process on the GPU drifts underneath the
    block.  Interleaving them and keeping the best window gives every candidate the same treatment;
    contention can only ever make a window slower, so the minimum is the honest number.
    """
    import random
    rng = random.Random(seed)
    for _ in range(warmup):
        for f in fns:
            f()
    torch.cuda.synchronize()
    best = [float("inf")] * len(fns)
    for _ in range(windows):
        order = list(range(len(fns)))
        rng.shuffle(order)
        for i in order:
            t = time.perf_counter()
            for _ in range(iters):
                fns[i]()
            torch.cuda.synchronize()
            best[i] = min(best[i], (time.perf_counter() - t) / iters * 1000)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=256, help="batch of sets")
    ap.add_argument("--N", type=int, nargs="*", default=None, help="sweep the number of entities")
    ap.add_argument("--cin", type=int, default=16, help="features per entity")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--masked", type=int, default=0, help="blank this many trailing slots per set")
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--windows", type=int, default=5)
    ap.add_argument("--baselines", action="store_true",
                    help="compare against parameter-matched MLP and attention stacks")
    a = ap.parse_args()

    dev = "cuda"
    print("gpu:", torch.cuda.get_device_name(0))
    kernels.load_extension()
    print("kernels:", kernels.extension_status())
    print(f"pair features: {ref.PAIR_SPEC} = PF{ref.PAIR_FEATS}")

    torch.manual_seed(0)
    net = LESNet(a.cin, width=a.width, depth=a.depth).to(dev)
    target = nparams(net)
    print(f"LESNet cin={a.cin} width={a.width} depth={a.depth}  {target:,} params")
    if a.baselines:
        print("baselines are parameter-matched: the MLP and attention stacks are given whatever "
              "width brings them closest to LESNet's parameter count, since their cost per block "
              "(~4*W^2) is a very different function of width than LESNet's (~2*cin*W + PF*W)")

    for N in (a.N or [32]):
        data = synth(a.B, N, a.cin, dev, masked=a.masked)
        cands = [("LESNet", net, a.width)]
        if a.baselines:
            for kind in ("mlp", "attn"):
                w = match_width(kind, a.cin, a.depth, target)
                cands.append((kind, build_baseline(kind, a.cin, w, a.depth).to(dev), w))
        fns = [make_step(m, w, data, dev) for _, m, w in cands]
        ms = round_robin(fns, a.iters, a.warmup, a.windows)
        print(f"\n  N={N:<5d} B={a.B:<5d} cin={a.cin:<4d} depth={a.depth}  "
              f"[min of {a.windows} windows, shuffled]")
        for (name, m, w), t in zip(cands, ms):
            print(f"    {name:<7s} width={w:<5d} {nparams(m) + w + 1:>10,} params  "
                  f"{t:8.2f} ms/step   {t / ms[0]:5.2f}x LESNet")


if __name__ == "__main__":
    main()
