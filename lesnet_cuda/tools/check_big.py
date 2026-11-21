"""Wide + deep sanity check: does the fast LESNet still work at the extreme end of the range
(width up to 2048, depth up to 15), and what does a step cost there?

    python tools/check_big.py [--width 2048] [--depth 15] [--B 8] [--N 32] [--cin 16] [--iters 3]

Reports params, peak GPU memory and ms/step.  It is deliberately a small batch: this is a
robustness check (the width-chunked grid, the 15-deep save buffer, no OOM), not a benchmark.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=2048)
    ap.add_argument("--depth", type=int, default=15)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--N", type=int, default=32)
    ap.add_argument("--cin", type=int, default=16)
    ap.add_argument("--iters", type=int, default=3)
    a = ap.parse_args()
    kernels.require_extension()
    dev = "cuda"
    torch.manual_seed(0)
    x, coords, mask = synth(a.B, a.N, a.cin, dev)
    net = LESNet(a.cin, width=a.width, depth=a.depth).to(dev)
    head = nn.Linear(a.width, 1).to(dev)
    nparam = sum(p.numel() for p in net.parameters()) + sum(p.numel() for p in head.parameters())
    print(f"LESNet cin={a.cin} width={a.width} depth={a.depth} B={a.B} N={a.N}  params {nparam:,}")

    opt = torch.optim.Adam(list(net.parameters()) + list(head.parameters()), lr=1e-4, fused=True)
    target = torch.zeros(a.B, device=dev)
    torch.cuda.reset_peak_memory_stats()
    try:
        for i in range(a.iters + 2):
            if i == 2:
                torch.cuda.synchronize(); t = time.perf_counter()
            y = net(x, coords, mask)
            loss = ((head(ref.masked_mean(y, mask)) - target) ** 2).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t) / a.iters * 1000
        print(f"  OK  {ms:8.2f} ms/step   peak mem {torch.cuda.max_memory_allocated()/2**20:8.1f} MiB"
              f"   loss {loss.item():.4f}   finite {bool(torch.isfinite(y).all())}")
    except RuntimeError as e:
        print(f"  FAIL {type(e).__name__}: {str(e).splitlines()[0]}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
