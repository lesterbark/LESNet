"""Size robustness: run the layer kernels against a pure-torch oracle over many shapes.

Covers: tiny and large N, N not a multiple of the block tile, W at/over the 1024-thread
limit (the width-chunking path), masked entities, PF==4 (vectorized pair features) and
PF!=4 (scalar path), and a packed gradient check.  Small tensors only, so it is quick.
"""
import os, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kernels  # noqa

SLOPE = 0.1


def torch_layer(ab, pf, E, c1, mask, N):
    """Reference: dense masked softmax-combine layer, matching the kernel exactly."""
    R, S = ab.shape
    W = S // 2
    B = R // N
    a = ab[:, :W].contiguous().view(B, N, W)
    b = ab[:, W:].contiguous().view(B, N, W)
    z = a[:, :, None, :] + b[:, None, :, :] + (pf @ E) + c1
    eye = torch.eye(N, dtype=torch.bool, device=ab.device)
    valid = mask[:, :, None] & mask[:, None, :] & ~eye
    v = torch.nn.functional.leaky_relu(z, SLOPE).masked_fill(~valid[:, :, :, None], -1e30)
    mx = v.max(2, keepdim=True).values
    e = torch.exp(v - mx)
    y = (e * v).sum(2) / e.sum(2).clamp_min(1e-20)
    return y.masked_fill(~valid.any(2)[:, :, None], 0.0)


# Every forward variant the dispatcher exposes: (name, tile size, shared-memory pair tile).  Each one
# advances its own online-softmax state, so each gets its own (y, mx, ss) and its own backward check --
# a variant that is only ever forward-tested can drift silently.  Names are only for failure messages.
JOINT_VARIANTS = [
    ("smem TI=8", 8, True),
    ("smem TI=4", 4, True),
    ("smem TI=1", 1, True),              # TI > N on the small shapes, exercises nt < TI
    ("smem TI=2", 2, True),
    ("global TI=8", 8, False),           # the no-shared-memory load path
    # Uninstantiated tile sizes must snap to a real template AND size the shared tile from the
    # snapped value; a mismatch under-allocates shared memory and corrupts the tile.
    ("TI=3 (snap)", 3, True),
    ("TI=99 (snap)", 99, True),
]
BWD_VARIANTS = [(0, True), (8, True), (2, True), (0, False)]   # (tile, shared pair tile)

def check(B, N, W, PF, mask_kwargs=None, tag=""):
    dev = "cuda"
    torch.manual_seed(hash((B, N, W, PF)) % 10000)
    C = 2 * W
    ab = (torch.randn(B * N, C, device=dev) * 0.3).requires_grad_(True)
    pf = (torch.randn(B, N, N, PF, device=dev) * 0.5).requires_grad_(True)
    E = (torch.randn(PF, W, device=dev) * 0.5).requires_grad_(True)
    c1 = torch.zeros(W, device=dev, requires_grad=True)
    mask = torch.ones(B, N, dtype=torch.bool, device=dev)
    if mask_kwargs:
        frm, to = mask_kwargs
        mask[:, frm:to] = False
    gy = torch.randn(B * N, W, device=dev)

    ref = torch_layer(ab, pf, E, c1, mask, N)
    # the joint kernel now takes the pair-feature width at runtime; PF > MAXPF (=8 in kernels.py)
    # is rejected up front by the extension, so keep the joint variants to the supported range
    variants = JOINT_VARIANTS if PF <= 8 else []

    # oracle gradients: ab -> [ga | gb]; pf gets no kernel gradient (handled outside the kernel)
    g = torch.autograd.grad((ref * gy.reshape(B, N, W)).sum(), [ab, pf, E, c1], allow_unused=True)
    g_ab = (g[0] if g[0] is not None else torch.zeros_like(ab))
    gE_ref = (g[2] if g[2] is not None else torch.zeros_like(E))
    gC_ref = (g[3] if g[3] is not None else torch.zeros_like(c1))
    scale = max(1e-9, g_ab.abs().max().item())

    fwd, worst = 0.0, ""
    bwd_ab = bwd_E = bwd_c = 0.0
    for name, ti, v in variants:
        out, mx, ss = kernels.les_fwd(ab.detach(), pf.detach(), E.detach(), c1.detach(), mask, N, ti, v)
        e = (out.view(B, N, W) - ref).abs().max().item()
        if e > fwd:
            fwd, worst = e, name
        # each forward variant's own (y, mx, ss) state feeds every backward variant
        for bti, bsm in BWD_VARIANTS:
            g2 = kernels.les_bwd(gy, ab.detach(), pf.detach(), E.detach(), c1.detach(),
                                 mask, out, mx, ss, N, bti, bsm)
            bwd_ab = max(bwd_ab, (g2[0] - g_ab).abs().max().item() / scale)
            bwd_E = max(bwd_E, (g2[1] - gE_ref).abs().max().item() / max(1e-9, gE_ref.abs().max().item()))
            bwd_c = max(bwd_c, (g2[2] - gC_ref).abs().max().item() / max(1e-9, gC_ref.abs().max().item()))

    ok = fwd < 2e-4 and bwd_ab < 2e-3 and bwd_E < 2e-3 and bwd_c < 2e-3
    print(f"  {'OK ' if ok else 'FAIL'} B={B:<3d} N={N:<4d} W={W:<5d} PF={PF} {tag:<14s}"
          f" [{len(variants)}f x{len(BWD_VARIANTS)}b var] fwd {fwd:.2e}  gab {bwd_ab:.2e}  "
          f"gE {bwd_E:.2e}  gc1 {bwd_c:.2e}"
          + (f"   worst fwd: {worst}" if worst and fwd > 1e-5 else ""))
    return ok


def torch_list_layer(a, b, pf, E, c1, idx):
    """Reference for the list layout (float32, same maths): a (R,W), b (S,W), pf (R,K,PF), idx (R,K)."""
    bj = b[idx.clamp_min(0).long()]                                   # (R,K,W)
    z = a[:, None, :] + bj + (pf @ E) + c1
    valid = idx >= 0
    v = torch.nn.functional.leaky_relu(z, SLOPE).masked_fill(~valid[..., None], -1e30)
    mx = v.max(1, keepdim=True).values
    e = torch.exp(v - mx)
    y = (e * v).sum(1) / e.sum(1).clamp_min(1e-20)
    return y.masked_fill(~valid.any(1)[:, None], 0.0)


def check_list(R, S, K, W, PF, empty_frac=0.2, packed=False, tag=""):
    """List kernels vs torch over shapes: tile remainders, W over the thread limit, odd PF, empty slots."""
    dev = "cuda"
    torch.manual_seed(hash((R, S, K, W, PF)) % 10000)
    a = (torch.randn(R, W, device=dev) * 0.4).requires_grad_(True)
    b = (torch.randn(S, W, device=dev) * 0.4).requires_grad_(True)
    pf = torch.randn(R, K, PF, device=dev) * 0.5
    E = (torch.randn(PF, W, device=dev) * 0.5).requires_grad_(True)
    c1 = torch.zeros(W, device=dev, requires_grad=True)
    idx = torch.randint(0, S, (R, K), device=dev)
    idx[torch.rand(R, K, device=dev) < empty_frac] = -1
    idx[: max(1, R // 7)] = -1                                       # some receivers with no partner
    idx = idx.int().contiguous()
    gy = torch.randn(R, W, device=dev)
    ref = torch_list_layer(a, b, pf, E, c1, idx)
    g = torch.autograd.grad((ref * gy).sum(), [a, b, E, c1])
    worst = 0.0
    for ti in (0, 1, 2, 8, 3):
        y, mx, ss = kernels.les_fwd_list(a.detach(), b.detach(), pf, idx, E.detach(), c1.detach(), ti)
        ga, gb = torch.zeros_like(a), torch.zeros_like(b)
        gE, gc = torch.zeros_like(E), torch.zeros_like(c1)
        kernels.les_bwd_list(gy, a.detach(), b.detach(), pf, idx, E.detach(), c1.detach(), y, mx, ss,
                             ga, gb, gE, gc, ti)
        errs = [(y - ref).abs().max().item() / max(1e-9, ref.abs().max().item())]
        for got, want in zip((ga, gb, gE, gc), g):
            errs.append((got - want).abs().max().item() / max(1e-9, want.abs().max().item()))
        worst = max(worst, max(errs))
    ok = worst < 2e-3
    print(f"  {'OK ' if ok else 'FAIL'} list R={R:<5d} S={S:<5d} K={K:<3d} W={W:<5d} PF={PF} "
          f"{tag:<14s} worst rel err over fwd/ga/gb/gE/gc1 and 5 tile sizes: {worst:.2e}")
    return ok


def main():
    assert torch.cuda.is_available()
    kernels.load_extension()
    print("extension:", kernels.extension_status())
    results = [
        check_list(1, 1, 1, 32, 4, tag="one receiver"),
        check_list(13, 13, 5, 64, 4, tag="R%TI!=0"),
        check_list(200, 200, 16, 128, 4),
        check_list(200, 50, 33, 100, 4, tag="S<R, ragged W"),
        check_list(64, 64, 12, 1200, 4, tag="W>512"),
        check_list(64, 64, 12, 2048, 4, tag="W=2048"),
        check_list(100, 100, 8, 96, 3, tag="PF=3"),
        check_list(100, 100, 8, 96, 5, tag="PF=5"),
        check_list(100, 100, 8, 96, 6, tag="PF=6"),
        check_list(100, 100, 8, 96, 7, tag="PF=7"),
        check_list(100, 100, 8, 96, 8, tag="PF=8"),
        check_list(100, 100, 8, 96, 1, tag="PF=1"),
        check_list(4096, 4096, 24, 64, 4, tag="large R"),
        check_list(300, 300, 40, 64, 4, empty_frac=0.95, tag="mostly empty"),
        check(2, 2, 32, 4, tag="tiny"),
        check(2, 7, 64, 4, tag="N<TI"),
        check(1, 8, 128, 4),
        check(1, 13, 96, 4, tag="N%TI!=0"),
        check(1, 32, 256, 4, tag="N=32 steady"),
        check(1, 32, 256, 3, tag="PF=3 scalar"),
        check(1, 32, 256, 5, tag="PF=5 odd"),      # odd width: unpacks to neither 4- nor 8-wide
        check(1, 32, 256, 6, tag="PF=6 with dir"),  # pos,axis,dir
        check(1, 32, 256, 7, tag="PF=7 polar"),     # pos,axis,polar -- the bundled distance+angle set
        check(1, 32, 256, 8, tag="PF=8"),
        check(1, 8, 512, 6, tag="PF=6 wide tile"),  # 96 KB tile -> shared staging must fall back
        check(1, 40, 100, 4, (5, 17), tag="masked"),
        check(1, 32, 1024, 4, tag="W=1024"),
        check(1, 32, 1200, 4, tag="W>1024"),
        check(1, 8, 2048, 4, tag="W=2048"),       # the wide end: shared tile falls back to global
        check(1, 128, 256, 4, tag="N=128"),
        check(2, 256, 128, 4, tag="N=256"),
        check(1, 512, 64, 4, tag="N=512"),
    ]
    print("\nPASS" if all(results) else "\nFAIL")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
