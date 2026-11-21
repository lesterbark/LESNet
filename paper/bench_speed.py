"""Speed and memory vs set size: LESNet (dense and list CUDA kernels) against a naive PyTorch LES
layer, a Transformer, and message-passing GNNs.  Run on an otherwise idle GPU.

    run.bat -u bench_speed.py            ->  results/speed.jsonl

Each row is one (model, N): milliseconds for one forward+backward of the whole network (min over
windows of several iterations, cuda events), and the peak GPU memory it needed.  The batch is sized so
B*N = 8192 entities everywhere, so a flat curve means constant cost per entity.  Width 128, depth 3.
"""
import json
import os
import time

import torch

import models
import neighbors
import tasks

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(R, exist_ok=True)
WIDTH, DEPTH, ENTITIES = 128, 3, 8192
K = 16
RADIUS = 1.5                    # density is 16 entities per unit area, so about 113 inside the radius: K=16 is the binding limit
PF = "pos,axis,polar"


def candidates():
    return [
        ("LES all-to-all (CUDA)", lambda: models.LESTrunk(tasks.CIN, WIDTH, DEPTH, pf_spec=PF), 4096),
        ("LES list K=16 (CUDA)", lambda: models.LESTrunk(tasks.CIN, WIDTH, DEPTH, pf_spec=PF, k=K, radius=RADIUS), 1 << 20),
        ("LES all-to-all (PyTorch)", lambda: models.LESTorchTrunk(tasks.CIN, WIDTH, DEPTH, pf_spec=PF), 512),
        ("Transformer", lambda: models.TransformerTrunk(tasks.CIN, WIDTH, DEPTH), 4096),
        ("Transformer+pf", lambda: models.TransformerTrunk(tasks.CIN, WIDTH, DEPTH, pf_spec=PF), 1024),
        ("GNN all-to-all (PyTorch)", lambda: models.GNNTrunk(tasks.CIN, WIDTH, DEPTH, pf_spec=PF), 256),
        ("GNN gather K=16", lambda: models.GatherGNNTrunk(tasks.CIN, WIDTH, DEPTH, PF, K, RADIUS), 1 << 20),
    ]


def measure(trunk, N, B, iters=4, windows=4):
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(0)
    x = torch.randn(B, N, tasks.CIN, generator=g, device=dev)
    L = (N / 16.0) ** 0.5
    coords = torch.rand(B, N, 2, generator=g, device=dev) * L
    mask = torch.ones(B, N, dtype=torch.bool, device=dev)
    trunk = trunk.to(dev)

    def step():
        trunk.zero_grad(set_to_none=True)
        trunk(x, coords, mask).square().mean().backward()

    for _ in range(2):
        step()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    best = 1e9
    for _ in range(windows):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _i in range(iters):
            step()
        e.record(); torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / iters)
    return best, torch.cuda.max_memory_allocated() / 2 ** 20


def main():
    torch.cuda.set_per_process_memory_fraction(0.85)       # allocations beyond the device budget fail (OOM) instead of spilling into system memory
    path = os.path.join(R, "speed.jsonl")
    open(path, "w").close()
    for name, build, nmax in candidates():
        for N in (16, 32, 64, 128, 256, 512, 1024, 2048, 4096):
            if N > nmax:
                continue
            B = max(1, ENTITIES // N)
            try:
                torch.cuda.empty_cache()
                trunk = build()
                params = models.count_params(trunk)
                ms, mem = measure(trunk, N, B)
                rec = dict(model=name, N=N, B=B, ms=ms, ms_per_entity_us=1000 * ms / (B * N), peak_mb=mem, params=params)
            except torch.cuda.OutOfMemoryError:
                rec = dict(model=name, N=N, B=B, ms=None, peak_mb=None, oom=True)
                torch.cuda.empty_cache()
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(rec, flush=True)


if __name__ == "__main__":
    main()
