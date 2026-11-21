"""Break a training step into its parts (forward, backward, optimizer), each timed with
min-of-windows so GPU contention shows up as noise rather than as a wrong ranking.

    python tools/step_bench.py [--adam plain|fused] [--B 256] [--N 32] [--cin 16] [--depth 5]
"""
import argparse, os, sys, time
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import kernels  # noqa
import ref  # noqa
from model import LESNet  # noqa
from bench import synth  # noqa


def timeit(fn, iters=15, warmup=6, windows=6):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(windows):
        t = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t) / iters * 1000)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=256)
    ap.add_argument("--adam", default="plain", choices=["plain", "fused"])
    ap.add_argument("--N", type=int, default=32)
    ap.add_argument("--cin", type=int, default=16)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--width", type=int, default=256)
    a = ap.parse_args()
    dev = "cuda"
    kernels.load_extension()
    x, coords, mask = synth(a.B, a.N, a.cin, dev)
    net = LESNet(a.cin, width=a.width, depth=a.depth).to(dev)
    head = nn.Linear(a.width, 1).to(dev)
    params = list(net.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=1e-4, fused=(a.adam == "fused"))
    target = torch.zeros(a.B, device=dev)

    def forward():
        return head(ref.masked_mean(net(x, coords, mask), mask))

    loss = ((forward() - target) ** 2).mean()
    loss.backward()

    def loss_of():
        return (((forward() - target) ** 2).mean())

    def fwd_bwd():
        opt.zero_grad(set_to_none=True)
        loss_of().backward()

    def step():
        opt.zero_grad(set_to_none=True)
        loss_of().backward()
        opt.step()

    def zero():
        opt.zero_grad(set_to_none=True)

    def optstep():
        opt.step()

    print(f"LESNet cin={a.cin} w={a.width} d={a.depth} B={a.B} N={a.N} adam={a.adam}  "
          f"params {sum(p.numel() for p in params):,}")
    for name, fn in (("forward (whole net)", forward), ("zero_grad", zero),
                     ("forward+backward", fwd_bwd), ("optimizer.step", optstep),
                     ("full step", step)):
        print(f"  {name:20s} {timeit(fn):8.2f} ms")


if __name__ == "__main__":
    main()
