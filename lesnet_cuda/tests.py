"""Correctness checks: fast CUDA LESNet vs the pure-torch reference (values AND gradients).

The reference runs in float64 as a strict oracle; the fast path is float32, so the diffs we
see are the kernel's own rounding error (~1e-5 relative), not an algorithmic mismatch.

Four things are checked:

  1. values, parameter gradients and the input gradient against the float64 oracle, over shapes
     that exercise the awkward cases (N under the block tile, ragged N and W, cin != width)
  2. masking -- a masked slot must contribute nothing at all, including when its coordinates are
     deliberately absurd
  3. permutation equivariance -- the layer is a set operation, so reordering the entities must
     reorder the answer and change nothing else.  This is a property of the architecture, not of
     any particular tensor shape, so it is worth asserting directly.
  4. the pf escape hatch (handing the layers a pre-built relation tensor) and the guard that
     stops a learnable one from being passed in and silently never trained

Run it once per pair-feature spec to cover them all:

    for s in "pos,axis" "pos,axis,dir" "pos,axis,polar"; do LES_PAIR_FEATS=$s python tests.py; done
"""
import sys
import torch

import kernels
import neighbors
import ref
from model import LESNet


def rel(u, v):
    return (u - v).abs().max().item() / (v.abs().max().item() + 1e-12)


def make_inputs(B, N, cin, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    mask = torch.rand(B, N, generator=g) > 0.25
    for b in range(B):                                   # never leave a row with no entities
        mask[b, torch.randperm(N, generator=g)[0]] = True
    # masked slots get deliberately absurd coordinates: a masked entity must contribute nothing,
    # so if its position leaks into a live pair's features the check will see it.
    coords = torch.randn(B, N, 2, generator=g)
    coords[~mask] = torch.randn(int((~mask).sum()), 2, generator=g) * 1e3
    # .to(dev) *then* requires_grad_: doing it the other way round leaves a non-leaf whose .grad is
    # never populated, and the input-gradient check would silently read None.
    x = (torch.randn(B, N, cin, generator=g) * 0.4).to("cuda").requires_grad_(True)
    return x, coords, mask


def check(B, N, cin, W, depth, seed, tag, k=None, radius=None):
    """Fast LESNet vs the float64 oracle on arbitrary entities (values and gradients).

    With `k` the partners are each entity's k nearest (within `radius` if given) -- the list layout."""
    dev = "cuda"
    x, coords, mask = make_inputs(B, N, cin, seed)
    fc, fm = coords.to(dev), mask.to(dev)
    nbr = None if k is None else neighbors.knn_neighbors(fc, fm, k=k, radius=radius)

    fnet = LESNet(cin, width=W, depth=depth).to(dev)
    rnet = ref.RefLESNet(cin, width=W, depth=depth).to(dev).double()
    rnet.load_state_dict(fnet.state_dict())

    fy = fnet(x, fc, fm, nbr=nbr)
    # the input gradient is part of the contract for a general block -- a LESNet used inside a
    # bigger model must pass gradient back into x -- so it is checked against the oracle, not
    # merely asserted to exist.  Hence a second, detached float64 copy that accumulates grad.
    xr = x.double().detach().requires_grad_(True)
    ry = rnet(xr, fc, fm, nbr=nbr)
    df = rel(fy.double().detach(), ry.detach())

    rnet.zero_grad(); fnet.zero_grad()
    w = fm.unsqueeze(-1).float()                          # masked slots must not move the loss
    (ry ** 2 * w.double()).sum().backward()
    (fy.double() ** 2 * w).sum().backward()
    worst, worst_name = 0.0, ""
    for (n, pr), (_, pfast) in zip(rnet.named_parameters(), fnet.named_parameters()):
        r = rel(pfast.grad.double(), pr.grad)
        if r > worst:
            worst, worst_name = r, n
    dgx = rel(x.grad.double(), xr.grad)

    ok = df < 2e-3 and worst < 5e-3 and dgx < 5e-3
    print(f"  {tag:<20} fwd {df:.2e}  param-grad {worst:.2e} ({worst_name})  input-grad {dgx:.2e}"
          f"  {'ok' if ok else 'FAIL'}")
    return ok


def check_equivariance(B, N, cin, W, depth, seed):
    """Reordering the entities must reorder the output and change nothing else.

    This is the property that makes it a set architecture.  Every slot is a real entity here (no
    mask holes), so a permutation of the input really is just a relabelling of the same set.
    """
    dev = "cuda"
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.randn(B, N, cin, generator=g) * 0.4).to(dev)
    coords = torch.randn(B, N, 2, generator=g).to(dev)
    mask = torch.ones(B, N, dtype=torch.bool, device=dev)
    net = LESNet(cin, width=W, depth=depth).to(dev)

    y = net(x, coords, mask)
    perm = torch.stack([torch.randperm(N, generator=g) for _ in range(B)]).to(dev)   # (B,N)
    bi = torch.arange(B, device=dev)[:, None]
    yp = net(x[bi, perm], coords[bi, perm], mask)
    d = rel(yp.detach(), y.detach()[bi, perm])            # yp[i] should equal y[perm[i]]
    ok = d < 2e-5
    print(f"  {'permutation':<20} rel diff {d:.2e}  {'ok' if ok else 'FAIL'}")
    return ok


def check_pf_path(B, N, cin, W, depth, seed):
    """A pre-built relation tensor must be equivalent to the one built from coordinates, and a
    learnable one must be refused rather than silently ignored."""
    dev = "cuda"
    x, coords, mask = make_inputs(B, N, cin, seed)
    fc, fm = coords.to(dev), mask.to(dev)
    net = LESNet(cin, width=W, depth=depth).to(dev)

    y = net(x, fc, fm)
    pf = ref.pair_features(fc, fm)[0]
    y2 = net(x, None, fm, pf=pf)
    d = rel(y2.detach(), y.detach())

    try:
        net(x, None, fm, pf=pf.detach().requires_grad_(True))
        guard = False
    except ValueError:
        guard = True
    ok = d < 1e-6 and guard
    print(f"  {'pf= escape hatch':<20} rel diff {d:.2e}  learnable-pf guard "
          f"{'fires' if guard else 'MISSING'}  {'ok' if ok else 'FAIL'}")
    return ok


def check_input_grad_flag(B, N, cin, W, depth, seed):
    """`input_grad=False` must skip only the gradient into x.

    It is a speed knob, so the forward and every parameter gradient have to come out identical to
    the default -- otherwise turning it on silently changes the model rather than just saving a
    matmul.
    """
    dev = "cuda"
    x, coords, mask = make_inputs(B, N, cin, seed)
    fc, fm = coords.to(dev), mask.to(dev)
    a = LESNet(cin, width=W, depth=depth, input_grad=True).to(dev)
    b = LESNet(cin, width=W, depth=depth, input_grad=False).to(dev)
    b.load_state_dict(a.state_dict())

    xa = x.detach().clone().requires_grad_(True)
    xb = x.detach().clone().requires_grad_(True)
    ya, yb = a(xa, fc, fm), b(xb, fc, fm)
    df = rel(ya.detach(), yb.detach())
    ya.sum().backward(); yb.sum().backward()
    worst = max(rel(p1.grad, p2.grad) for (_, p1), (_, p2)
                in zip(a.named_parameters(), b.named_parameters()))
    skipped = xb.grad is None                          # the entire point of the knob
    ok = df < 1e-6 and worst < 1e-6 and skipped
    print(f"  {'input_grad=False':<20} fwd {df:.2e}  param-grad {worst:.2e}  input-grad "
          f"{'skipped' if skipped else 'STILL COMPUTED'}  {'ok' if ok else 'FAIL'}")
    return ok


def check_ragged_batch(B, N, cin, W, depth, seed):
    """A set with 3 entities and one with 30 must share a batch and get identical answers.

    This is the whole point of `mask` being an input: N is a ceiling, not a requirement. So the
    test is not "it runs" -- it is that a row with c real entities, padded into an N-slot batch,
    produces exactly what it produces on its own with N=c. If padding leaked in anywhere (a
    denominator, a stale tile, an offset), this is where it shows.
    """
    dev = "cuda"
    g = torch.Generator(device="cpu").manual_seed(seed)
    counts = [N, 1, max(1, N // 2), N - 1, 2]        # deliberately ragged, including the extremes
    B = len(counts)
    x = torch.randn(B, N, cin, generator=g).to(dev)
    coords = torch.randn(B, N, 2, generator=g).to(dev)
    mask = torch.zeros(B, N, dtype=torch.bool, device=dev)
    for b, c in enumerate(counts):
        mask[b, :c] = True

    net = LESNet(cin, width=W, depth=depth).to(dev)
    padded = net(x, coords, mask).detach()

    worst = 0.0
    for b, c in enumerate(counts):
        alone = net(x[b:b + 1, :c], coords[b:b + 1, :c],
                    torch.ones(1, c, dtype=torch.bool, device=dev)).detach()
        worst = max(worst, rel(padded[b, :c], alone[0]))
        if c < N and padded[b, c:].abs().max().item() != 0.0:
            worst = float("inf")                     # a padded slot must stay exactly zero
    ok = worst < 1e-5
    print(f"  {'ragged N per row':<20} counts {counts}  vs run-alone rel diff {worst:.2e}  "
          f"{'ok' if ok else 'FAIL'}")
    return ok


def check_feature_depth(W, depth, dev="cuda"):
    """`cin` is free: the kernels never see it, only the two projections do.

    The layer's CUDA takes `ab` (the already-projected features) and never the entity feature
    vector itself, so C is not a kernel parameter at all -- 1, 80 or 4096 cost the same kernel.
    This checks a few depths that have nothing in common, against the oracle.
    """
    ok = True
    for cin in (1, 3, 80, 129):
        ok &= check(2, 5, cin, W, depth, seed=200 + cin, tag=f"N=5 cin={cin}")
    return ok


def check_list_equals_dense(B, N, cin, W, depth, seed):
    """The list layout with every other entity listed IS the dense layer (same kernel maths, a
    different walk).  This pins the two paths to each other, not just each to the oracle."""
    dev = "cuda"
    x, coords, mask = make_inputs(B, N, cin, seed)
    fc, fm = coords.to(dev), mask.to(dev)
    net = LESNet(cin, width=W, depth=depth).to(dev)
    nbr = neighbors.knn_neighbors(fc, fm, k=N - 1)
    d = rel(net(x, fc, fm, nbr=nbr).detach(), net(x, fc, fm).detach())
    ok = d < 2e-5
    print(f"  {'list(K=N-1)==dense':<20} rel diff {d:.2e}  {'ok' if ok else 'FAIL'}")
    return ok


def check_list_equivariance(B, N, cin, W, depth, seed, k):
    """kNN lists are built from geometry, so reordering the entities must still just reorder y."""
    dev = "cuda"
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.randn(B, N, cin, generator=g) * 0.4).to(dev)
    coords = torch.randn(B, N, 2, generator=g).to(dev)
    net = LESNet(cin, width=W, depth=depth).to(dev)
    y = net.local(x, coords, None, k=k)
    perm = torch.stack([torch.randperm(N, generator=g) for _ in range(B)]).to(dev)
    bi = torch.arange(B, device=dev)[:, None]
    yp = net.local(x[bi, perm], coords[bi, perm], None, k=k)
    d = rel(yp.detach(), y.detach()[bi, perm])
    ok = d < 2e-5
    print(f"  {'list permutation':<20} k={k} rel diff {d:.2e}  {'ok' if ok else 'FAIL'}")
    return ok


def check_rect(Rn, Sn, K, cin, W, seed):
    """Receivers != senders: one layer, raw kernel entry point, against the oracle layer."""
    dev = "cuda"
    g = torch.Generator(device="cpu").manual_seed(seed)
    xr = (torch.randn(Rn, cin, generator=g) * 0.5).to(dev).requires_grad_(True)
    xs = (torch.randn(Sn, cin, generator=g) * 0.5).to(dev).requires_grad_(True)
    idx = torch.randint(-1, Sn, (Rn, K), generator=g).to(dev)          # includes empty slots
    idx[0] = -1                                                       # one receiver with no partners
    pf = torch.randn(Rn, K, ref.PAIR_FEATS, generator=g).to(dev)
    lay = ref.RefSoftmaxLayer(cin, W).to(dev)
    A, B_, E, c1 = (p.detach().clone().float().requires_grad_(True)
                    for p in (lay.A, lay.B, lay.E, lay.c1))
    y = torch.nn.functional.leaky_relu(
        kernels.les_layer_list(xr, xs, idx.int().contiguous(), pf.contiguous(), A, B_, E, c1), 0.1)
    lay = lay.double()
    xr2, xs2 = xr.detach().double().requires_grad_(True), xs.detach().double().requires_grad_(True)
    y2 = lay.forward_list(xr2[None], pf.double()[None], idx[None], xs=xs2[None])[0]
    gy = torch.randn(Rn, W, generator=g).to(dev)
    (y * gy).sum().backward()
    (y2 * gy.double()).sum().backward()
    d = rel(y.detach().double(), y2.detach())
    gs = [rel(A.grad.double(), lay.A.grad), rel(B_.grad.double(), lay.B.grad),
          rel(E.grad.double(), lay.E.grad), rel(c1.grad.double(), lay.c1.grad),
          rel(xr.grad.double(), xr2.grad), rel(xs.grad.double(), xs2.grad)]
    ok = d < 2e-5 and max(gs) < 5e-4
    print(f"  {f'rect R{Rn} S{Sn} K{K}':<20} fwd {d:.2e}  grads A/B/E/c/xr/xs max {max(gs):.2e}  "
          f"{'ok' if ok else 'FAIL'}")
    return ok


def check_grid_knn(B, N, k, radius, frac, seed):
    """The grid search must return exactly what an exact brute-force search returns (same partners, same
    nearest-first order), including masked entities, tiny radii (receivers with no partner) and k near N."""
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed)
    coords = torch.rand(B, N, 2, generator=g, device=dev) * 3.0
    mask = torch.rand(B, N, generator=g, device=dev) < frac
    ref = neighbors.knn_neighbors(coords, mask, k=k, radius=radius, use_grid=False)
    got = neighbors.knn_neighbors(coords, mask, k=k, radius=radius, use_grid=True)
    ok = ref.shape == got.shape and bool((ref == got).all())
    print(f"  {f'grid kNN B{B} N{N} k{k} r{radius}':<28} identical to brute force: {ok}  "
          f"(mean partners {(got >= 0).sum(-1).float().mean():.1f})")
    return ok


def check_multiplicity_invariance(B, N, cin, W, seed, k=6, m=3):
    """Repeating every partner m times must not change the output (the soft-max weights renormalise).

    This is the formal reason a slice cannot count -- stated and used in the paper -- so it is asserted
    on the kernels, not just argued."""
    dev = "cuda"
    x, coords, mask = make_inputs(B, N, cin, seed)
    fc, fm = coords.to(dev), mask.to(dev)
    net = LESNet(cin, width=W, depth=1).to(dev)
    nbr = neighbors.knn_neighbors(fc, fm, k=k)
    nbr_m = torch.cat([nbr] * m, -1)
    pf = ref.pair_features_nbr(fc, nbr)
    pf_m = torch.cat([pf] * m, 2)
    y1 = net(x.detach(), fc, fm, pf=pf, nbr=nbr).detach()
    y2 = net(x.detach(), fc, fm, pf=pf_m, nbr=nbr_m).detach()
    d = rel(y2, y1)
    ok = d < 2e-6
    print(f"  {'multiplicity invariance':<22} each partner x{m}: rel diff {d:.2e}  {'ok' if ok else 'FAIL'}")
    return ok


def check_spec_parsing():
    """The feature set is a config knob, so a typo in it must be an error rather than a silently
    narrower model -- and a bundle must expand to its parts in a fixed order."""
    good = ref.pair_parts("pos,axis,polar") == ["pos", "axis", "dist", "dir"]
    dedup = ref.pair_parts("pos,pos,axis") == ["pos", "axis"]
    try:
        ref.pair_parts("pos,bogus")
        raised = False
    except ValueError:
        raised = True
    ok = good and dedup and raised
    print(f"  {'spec parsing':<20} bundle {'expands' if good else 'WRONG'}  dupes "
          f"{'dropped' if dedup else 'WRONG'}  typo {'rejected' if raised else 'ACCEPTED'}  "
          f"{'ok' if ok else 'FAIL'}")
    return ok


def main():
    assert torch.cuda.is_available()
    kernels.load_extension(verbose=True)
    print("extension:", kernels.extension_status())
    print(f"pair features: {ref.PAIR_SPEC} = PF{ref.PAIR_FEATS}")

    print("\n-- fast vs float64 oracle --")
    ok = True
    for i, (b, n, c, w, d) in enumerate([
        (4, 16, 8, 64, 3),                 # small and square
        (3, 7, 5, 32, 1),                  # N below the block tile, single layer
        (2, 33, 12, 128, 4),               # N just over 32, W a multiple of the tile
        (2, 50, 6, 100, 2),                # ragged N and a ragged W
        (2, 64, 6, 256, 5),                # a steady-state layer: cin != width
        (4, 5, 80, 256, 3),                # a handful of entities, each 80 features deep
    ]):
        ok &= check(b, n, c, w, d, seed=100 + i, tag=f"B{b} N{n} C{c} W{w} D{d}")

    print("\n-- list layout (k nearest / radius / receivers != senders) vs the oracle --")
    for i, (b, n, c, w, d, k, rad) in enumerate([
        (3, 24, 8, 64, 3, 6, None),        # plain kNN
        (2, 40, 6, 128, 2, 12, 0.9),       # kNN limited by a radius: some receivers get few partners
        (2, 33, 6, 100, 2, 32, None),      # K = N-1 and a ragged W
        (4, 9, 5, 32, 1, 3, 0.6),          # small radius: several receivers with NO partners at all
        (2, 64, 6, 256, 4, 16, None),      # steady state
    ]):
        ok &= check(b, n, c, w, d, seed=300 + i, tag=f"list B{b} N{n} K{k}", k=k, radius=rad)
    ok &= check_list_equals_dense(3, 24, 10, 128, 3, seed=61)
    ok &= check_list_equivariance(3, 40, 10, 128, 3, seed=63, k=8)
    for i, (b, n, k, r, fr) in enumerate([(2, 300, 8, 0.4, 0.8), (3, 1000, 24, 0.3, 0.9), (1, 64, 5, 1.5, 1.0),
                                          (4, 500, 16, 0.05, 0.7), (2, 2000, 32, 0.25, 0.95), (1, 40, 64, 2.0, 1.0)]):
        ok &= check_grid_knn(b, n, k, r, fr, seed=400 + i)
    ok &= check_multiplicity_invariance(3, 24, 8, 64, seed=81)
    ok &= check_rect(50, 17, 9, 6, 64, seed=71)
    ok &= check_rect(7, 200, 30, 11, 100, seed=72)

    print("\n-- variable feature depth (cin) --")
    ok &= check_feature_depth(192, 3)

    print("\n-- variable entity count (N is per row, not per batch) --")
    ok &= check_ragged_batch(2, 8, 12, 128, 3, seed=41)

    print("\n-- architecture properties --")
    ok &= check_equivariance(3, 24, 10, 128, 3, seed=17)
    ok &= check_pf_path(3, 24, 10, 128, 3, seed=23)
    ok &= check_input_grad_flag(3, 24, 10, 128, 3, seed=31)
    ok &= check_spec_parsing()

    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
