"""Pick the receiver tile `TI` for the list kernels: time forward and backward for TI in {1,2,4,8}.

    run.bat tools\tune_list.py        (run with the GPU to itself; windows are shuffled)

Shapes are the ones the paper uses: a fluid-sized problem (R=8192, K=24, PF=7) at W=64/128/256.
"""
import os, random, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import kernels  # noqa: E402


def timeit(fn, iters=20):
    fn(); fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    kernels.require_extension()
    dev = "cuda"
    R, K, PF = 8192, 24, 7
    for W in (64, 128, 256):
        g = torch.Generator(device=dev).manual_seed(0)
        a = torch.randn(R, W, device=dev, generator=g) * 0.4
        b = torch.randn(R, W, device=dev, generator=g) * 0.4
        pf = torch.randn(R, K, PF, device=dev, generator=g)
        idx = torch.randint(0, R, (R, K), device=dev, generator=g).int()
        E = torch.randn(PF, W, device=dev, generator=g) * 0.5
        c1 = torch.zeros(W, device=dev)
        gy = torch.randn(R, W, device=dev, generator=g)
        res = {}
        for rnd in range(3):
            order = [1, 2, 4, 8]
            random.shuffle(order)
            for ti in order:
                y, mx, ss = kernels.les_fwd_list(a, b, pf, idx, E, c1, ti)
                f = timeit(lambda: kernels.les_fwd_list(a, b, pf, idx, E, c1, ti))

                def bwd():
                    ga, gb = torch.zeros_like(a), torch.zeros_like(b)
                    gE, gc = torch.zeros_like(E), torch.zeros_like(c1)
                    kernels.les_bwd_list(gy, a, b, pf, idx, E, c1, y, mx, ss, ga, gb, gE, gc, ti)
                bw = timeit(bwd)
                old = res.get(ti, (1e9, 1e9))
                res[ti] = (min(old[0], f), min(old[1], bw))
        print(f"W={W:4d}  R={R} K={K}   " + "   ".join(f"TI={ti}: fwd {res[ti][0]:.3f} bwd {res[ti][1]:.3f} ms" for ti in sorted(res)))


if __name__ == "__main__":
    main()
