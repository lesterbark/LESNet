"""Dense vs list layout: ms per forward+backward as the set grows.

    run.bat tools\bench_list.py
    run.bat tools\bench_list.py --N 256 1024 4096 --K 8 16 32 --width 128 --depth 4

Every timing is min-of-windows after warm-up, with cuda events, and the neighbour search is timed
separately from the layers so you can see what each costs.
"""
import argparse, os, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import kernels, neighbors  # noqa: E402
from model import LESNet  # noqa: E402


def timeit(fn, warm=3, windows=5, iters=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(windows):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / iters)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, nargs="+", default=[64, 256, 1024, 4096])
    ap.add_argument("--K", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--B", type=int, default=0, help="batch (0 = size it so B*N is about 16k entities)")
    ap.add_argument("--cin", type=int, default=16)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--dense-max", type=int, default=1024)
    a = ap.parse_args()
    kernels.require_extension()
    dev = "cuda"
    net = LESNet(a.cin, a.width, a.depth).to(dev)
    print(f"gpu: {torch.cuda.get_device_name()}   cin={a.cin} width={a.width} depth={a.depth}")
    print(f"{'N':>6} {'B':>4} | {'dense':>9} | " + " | ".join(f"list K={k:<3d}" for k in a.K) + " | knn brute | knn grid   (ms; layers = fwd+bwd)")
    for N in a.N:
        B = a.B or max(1, 16384 // N)
        g = torch.Generator(device="cpu").manual_seed(0)
        x = torch.randn(B, N, a.cin, generator=g).to(dev)
        coords = torch.rand(B, N, 2, generator=g).to(dev)
        mask = torch.ones(B, N, dtype=torch.bool, device=dev)
        cells = []
        if N <= a.dense_max:
            def dense():
                net.zero_grad(set_to_none=True)
                net(x, coords, mask).square().mean().backward()
            cells.append(f"{timeit(dense):9.2f}")
        else:
            cells.append(f"{'-':>9}")
        for k in a.K:
            nbr = neighbors.knn_neighbors(coords, mask, k=k)

            def lst():
                net.zero_grad(set_to_none=True)
                net(x, coords, mask, nbr=nbr).square().mean().backward()
            cells.append(f"{timeit(lst):9.2f}  ")
        knn = timeit(lambda: neighbors.knn_neighbors(coords, mask, k=16))
        r = 1.5 * (16.0 / (3.14159 * N)) ** 0.5                       # about 36 entities inside the radius
        knn_grid = timeit(lambda: neighbors.knn_neighbors(coords, mask, k=16, radius=r))
        print(f"{N:6d} {B:4d} | " + " | ".join(cells) + f" | {knn:8.2f} | {knn_grid:8.2f}")


if __name__ == "__main__":
    main()
