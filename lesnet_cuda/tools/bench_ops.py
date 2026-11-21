"""Per-op breakdown of one LESNet layer, timed with min-of-windows.

The torch profiler over-attributes under GPU contention (its own serialisation shows up as
"CUDA time"), so every number here is the *best* of several windows: contention makes a
window slower, never faster, so the minimum is the closest to the uncontended truth.

    python tools/bench_ops.py [--B 256] [--N 32] [--W 256] [--cin 16] [--masked 4]
"""
import argparse, os, sys, time
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import kernels  # noqa
import ref  # noqa
from bench import synth  # noqa


def timeit(fn, iters=20, warmup=8, windows=7):
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
    ap.add_argument("--N", type=int, default=32)
    ap.add_argument("--W", type=int, default=256)
    ap.add_argument("--cin", type=int, default=16)
    ap.add_argument("--ctile", type=int, default=0,
                    help="force the layer's input width C (256 = a steady-state hidden layer)")
    ap.add_argument("--masked", type=int, default=0, help="blank this many trailing slots per set")
    a = ap.parse_args()
    dev = "cuda"
    kernels.load_extension()

    x, coords, mask = synth(a.B, a.N, a.cin, dev, masked=a.masked)
    B, N = mask.shape
    C = a.ctile or a.cin
    W = a.W
    R = B * N

    pf, _ = ref.pair_features(coords, mask)
    xf = torch.randn(R, C, device=dev)
    A = torch.randn(C, W, device=dev)
    Bp = torch.randn(C, W, device=dev)
    E = torch.randn(ref.PAIR_FEATS, W, device=dev)      # follows the active pair-feature set
    c1 = torch.zeros(W, device=dev)
    Wc = torch.cat([A, Bp], 1).contiguous()
    ab = xf @ Wc

    y, mx, ss = kernels.les_fwd(ab, pf, E, c1, mask, N)
    gy = torch.randn(R, W, device=dev)
    gab = torch.zeros(R, 2 * W, device=dev)

    ops = {}

    def bench(name, fn):
        ops[name] = timeit(fn)

    import model as _model
    # every kernel entry point is measured with the PRODUCTION variant, not the legacy default
    fv, bv = _model.LES_FWD_V, _model.LES_BWD_V
    fti, bti = _model.LES_TI, _model.LES_TIB
    # building the relation tensor is per-STEP, not per-layer (all layers reuse the same pf), but it
    # is worth seeing next to the layer it feeds: it is the biggest input in the model.
    bench("pair_features (per step)", lambda: ref.pair_features(coords, mask))
    print(f"  [sizes] entity feats {x.numel()*4/2**20:.2f} MiB   "
          f"pf {pf.numel()*4/2**20:.2f} MiB ({pf.numel()/x.numel():.1f}x)  "
          f"PF={ref.PAIR_FEATS}  masked {a.masked}/{N} slots\n")
    bench("cat(A,B)", lambda: torch.cat([A, Bp], 1))
    bench(f"mm ab=xf@Wc  ({R}x{C}@{C}x{2*W})", lambda: xf @ Wc)
    bench(f"fwd kernel  ({R}x{W})", lambda: kernels.les_fwd(ab, pf, E, c1, mask, N, fti, fv))
    bench(f"zeros gab   ({R}x{2*W})", lambda: torch.zeros(R, 2 * W, device=dev))
    bench("zeros small (gE,gc1)", lambda: (torch.zeros(ref.PAIR_FEATS, W, device=dev),
                                           torch.zeros(W, device=dev)))
    bench("bwd kernel+zeros", lambda: kernels.les_bwd(gy, ab, pf, E, c1, mask, y, mx, ss, N, bti, bv))
    bench(f"mm gWc=xf.t@gab ({C}x{R}@{R}x{2*W})", lambda: xf.t() @ gab)
    if _model.LES_INPUT_GRAD:
        bench(f"mm gx=gab@Wc.t ({R}x{2*W}@{2*W}x{C})", lambda: gab @ Wc.t())
    gWc = xf.t() @ gab
    bench("slice gA/gB .contiguous()", lambda: (gWc[:, :W].contiguous(), gWc[:, W:].contiguous()))
    bench("gy.reshape().contiguous()", lambda: gy.reshape(R, W).contiguous())

    total = sum(ops.values())
    print(f"LESNet layer op breakdown  B={B} N={N} W={W} C={C}  (min-of-windows, ms)")
    for k, v in sorted(ops.items(), key=lambda kv: -kv[1]):
        print(f"  {v:8.3f} ms  {100*v/total:5.1f}%   {k}")
    print(f"  {total:8.3f} ms          TOTAL (sum of parts)")


if __name__ == "__main__":
    main()
