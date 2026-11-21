"""Pure-PyTorch reference for LESNet: the ground truth the CUDA kernels are checked against.

LESNet is a general set network.  Its input is a set of entities -- not a grid, not a sequence,
not a board -- and one layer, for entity i and slice k, is

    v_ij  = leaky( A x_i + B x_j + E pf_ij + c1 )     # linear pair filter over the relation vector
    w_ij  = softmax over partners j of v_ij           # a separate distribution per slice
    out_i = leaky( sum_j w_ij * v_ij )                # the soft-max combine

`pf_ij` is whatever relation describes entity j *from* entity i.  By default it is the relative
geometry that `pair_features` builds from each entity's two coordinates, but any (B,N,N,PF)
tensor will do -- a graph, a retrieval pass, a learned score.

Two things about the shape of the input are deliberate, because they are what makes this a set
architecture rather than a sequence one:

    x      (B,N,C)   one feature vector per entity, in any order -- the layer is permutation
                     equivariant, so the same set in a different order gives the same answer
    mask   (B,N)     which slots actually hold an entity.  N is a ceiling, not a requirement:
                     a batch with 3 entities and one with 30 both run, no padding waste.

This is the ground truth.  The fast CUDA path must match it bit-for-bit-ish (we check forward
values and all gradients).  It is only used on small tensors, because the reference materialises
the (B,N,N,W) pair tensor that the kernels avoid.
"""
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

SLOPE = 0.1
act = lambda x: F.leaky_relu(x, SLOPE)

# ------------------------------------------------------------------------------- pair-feature set
# WHICH relations matter is a per-task decision, so the width of the pair-feature vector is a
# choice here rather than a constant, and the kernels take it at runtime.  Pieces of the vector:
#
#   pos     dx, dy            relative position to the partner (signed)
#   axis    |dx|, |dy|        per-axis distance
#   dist    r                 radial distance, sqrt(dx^2 + dy^2)
#   dir     dx/r, dy/r        normalised direction (cos, sin)
#
# Why `dir` is genuinely new rather than a restatement of `pos`: a slice only ever makes one
# straight cut through the numbers it is handed.  With `pos` alone, "north-east of me" and
# "north-east of me AND far away" are different cuts, so the two can never be separated.  Adding
# the normalised direction lets a slice weight dx/dy off entirely and cut on pure direction, which
# makes "roughly in this direction, at any distance" ONE slice.
#
# Coordinates arrive in whatever units the caller supplies, so `pos` and `dist` carry those units
# too -- pre-scale them if the numbers are large.  `dir` is scale-free.  See pair_features().
#
# Why cos/sin and not an angle in radians: a single ordered number implies an order that isn't
# there, one straight cut can't isolate a middle value of it, and it jumps at +/-pi.
PARTS = {"pos": 2, "axis": 2, "dist": 1, "dir": 2}

# Bundles.  Distance and direction are one idea wearing two hats and are always turned on together:
# a distance with no direction says "something is near me" but not which way, a direction with no
# distance says "something lies that way" but not how far, and neither is a relation a single slice
# can make a clean cut through.  Giving both at once is what makes "that way, at any distance" (or
# "nearby, whichever way") collapse into one slice, which is the entire payoff.  `polar` expands to
# the pair so the two cannot be switched on independently by accident.
BUNDLES = {"polar": ("dist", "dir")}
DEFAULT_PAIR_SPEC = "pos,axis"


def pair_parts(spec):
    """Spec string -> ordered list of raw parts, bundles expanded, duplicates dropped.

    An unrecognised name is an error, not something to quietly drop.  A typo shrinks the relation
    vector, which shrinks what the net can see, and the only symptom would be a slightly worse
    model -- the one failure mode that is genuinely hard to notice from the outside.
    """
    out, bad = [], []
    for tok in (t.strip() for t in spec.split(",")):
        if not tok:
            continue
        if tok in BUNDLES:
            for p in BUNDLES[tok]:
                if p not in out:
                    out.append(p)
        elif tok in PARTS:
            if tok not in out:
                out.append(tok)
        else:
            bad.append(tok)
    if bad:
        raise ValueError(
            f"unknown pair-feature name(s) {bad} in LES_PAIR_FEATS={spec!r}.\n"
            f"  parts:   {', '.join(sorted(PARTS))}\n"
            f"  bundles: {', '.join(sorted(BUNDLES))}   (polar = {' + '.join(BUNDLES['polar'])})")
    return out


PAIR_SPEC = os.environ.get("LES_PAIR_FEATS", DEFAULT_PAIR_SPEC)
PAIR_PARTS = pair_parts(PAIR_SPEC)
if not PAIR_PARTS:                          # never leave the net with no pair information at all
    PAIR_PARTS, PAIR_SPEC = pair_parts(DEFAULT_PAIR_SPEC), DEFAULT_PAIR_SPEC
PAIR_FEATS = sum(PARTS[p] for p in PAIR_PARTS)


# ------------------------------------------------------------------------------- input building
def pair_mask(mask):
    """(B,N) entity mask -> (B,N,N) ordered-pair validity (both present, and not self)."""
    N = mask.shape[1]
    eye = torch.eye(N, dtype=torch.bool, device=mask.device)
    return mask[:, :, None] & mask[:, None, :] & ~eye


def pair_features(coords, mask, spec=None):
    """(B,N,2) coordinates + (B,N) mask -> pf (B,N,N,PAIR_FEATS) and valid (B,N,N).

    This is the general way into the architecture.  LESNet is a set network: a layer needs the
    entity features, and one relation vector per ordered pair.  Whatever your entities are --
    pixels, tokens, objects, graph nodes, particles -- if each one has two coordinates, this turns
    them into the pair features the layers consume.  `pf[b,i,j]` describes entity j *from* entity i.

    It is a pure torch function, so it composes: slice, scale, warp or replace it as your task
    needs.  You can equally feed the layers a `pf` you built some other way (learned, from a graph,
    from a retrieval pass) -- the layers only care about the width.

    Coordinates are taken in whatever units you pass, since the layer's weights read dx/dy
    directly: pre-scale them yourself if the numbers are large.  `dir` is scale-free; `dist`
    carries the units you supplied.

    Interpolation/rounding: `dx,dy` are coordinate differences, so they are already translation
    invariant -- nothing here depends on the origin.  Two entities at the same point give r=0, so
    r is clamped and `dir` is defined as zero rather than 0/0.

    Note this materialises every ordered pair, i.e. N rows per entity, when the filter only ever
    consumes one pair at a time.  That is a layout choice, not a requirement: see the note in
    TRAVEL_GUIDE.md -- the same numbers can be derived inside the kernel from the same two
    coordinates per entity, which removes this tensor entirely.
    """
    parts = PAIR_PARTS if spec is None else pair_parts(spec)
    if sum(PARTS[p] for p in parts) != PAIR_FEATS:
        raise ValueError(
            f"pair-feature spec {spec!r} is {sum(PARTS[p] for p in parts)} wide but the layers were "
            f"built for {PAIR_FEATS} (LES_PAIR_FEATS={PAIR_SPEC!r}).  The width sizes E, so it is "
            f"fixed when the model is constructed -- set the env var before importing, or rebuild.")
    if not coords.is_floating_point():
        coords = coords.float()
    cx, cy = coords[..., 0], coords[..., 1]
    dx = cx[:, :, None] - cx[:, None, :]
    dy = cy[:, :, None] - cy[:, None, :]
    return _stack_pair_parts(parts, dx, dy), pair_mask(mask)


def _stack_pair_parts(parts, dx, dy):
    # clamped so two entities sharing a spot cannot produce 0/0 -> NaN, which would poison the whole
    # layer rather than just that one pair.  Clamping the norm means `dir` of a coincident pair is
    # (0,0), a real value, instead of NaN.
    r = torch.sqrt(dx * dx + dy * dy).clamp_min(1e-6)
    pieces = {"pos": [dx, dy], "axis": [dx.abs(), dy.abs()], "dist": [r], "dir": [dx / r, dy / r]}
    return torch.stack([t for p in parts for t in pieces[p]], -1)


def pair_features_nbr(coords, nbr, coords_s=None):
    """Pair features for the list layout: (B,R,2) receiver coords + (B,R,K) local sender indices
    (-1 = none) -> pf (B,R,K,PAIR_FEATS).  Same definition as pair_features(), row by row: entry
    [b,i,kk] describes sender nbr[b,i,kk] *from* receiver i.  Empty slots hold the features of
    sender 0 (finite, and ignored because the layer skips them)."""
    cs = coords if coords_s is None else coords_s
    if not coords.is_floating_point():
        coords, cs = coords.float(), cs.float()
    B, R, K = nbr.shape
    cj = torch.gather(cs, 1, nbr.clamp_min(0).reshape(B, R * K, 1).expand(-1, -1, cs.shape[-1]))
    cj = cj.view(B, R, K, -1)
    dx = coords[:, :, None, 0] - cj[..., 0]
    dy = coords[:, :, None, 1] - cj[..., 1]
    return _stack_pair_parts(PAIR_PARTS, dx, dy)



# ---------------------------------------------------------------------------- reference layer
class RefSoftmaxLayer(nn.Module):
    def __init__(self, cin, width):
        super().__init__()
        self.A = nn.Parameter(torch.randn(cin, width) * math.sqrt(2.0 / (2 * cin)))
        self.B = nn.Parameter(torch.randn(cin, width) * math.sqrt(2.0 / (2 * cin)))
        self.E = nn.Parameter(torch.randn(PAIR_FEATS, width) * 0.5)
        self.c1 = nn.Parameter(torch.zeros(width))

    def forward(self, x, pf, valid):
        Bn, N, C = x.shape
        W = self.A.shape[1]
        a = x @ self.A                                              # (B,N,W)
        b = x @ self.B
        z = a[:, :, None, :] + b[:, None, :, :] + (pf @ self.E)     # (B,N,N,W)
        z = z + self.c1
        v = act(z)
        v = v.masked_fill(~valid[:, :, :, None], -1e30)
        mx = v.max(dim=2, keepdim=True).values
        e = torch.exp(v - mx)                                       # invalid -> 0
        den = e.sum(2, keepdim=True)
        y = (e * v).sum(2) / den.squeeze(2).clamp_min(1e-20)
        y = y.masked_fill(~valid.any(2)[:, :, None], 0.0)           # entities with no partners
        return act(y)

    def forward_list(self, x, pf, nbr, xs=None):
        """List layout: pf (B,R,K,PF), nbr (B,R,K) local sender indices (-1 = none).  `xs` are the
        senders' features when they are a different set from the receivers."""
        Bn = x.shape[0]
        xs = x if xs is None else xs
        a = x @ self.A
        b = xs @ self.B
        bj = b[torch.arange(Bn, device=x.device)[:, None, None], nbr.clamp_min(0)]   # (B,R,K,W)
        z = a[:, :, None, :] + bj + (pf @ self.E) + self.c1
        v = act(z)
        valid = nbr >= 0
        v = v.masked_fill(~valid[..., None], -1e30)
        mx = v.max(dim=2, keepdim=True).values
        e = torch.exp(v - mx)
        den = e.sum(2, keepdim=True)
        y = (e * v).sum(2) / den.squeeze(2).clamp_min(1e-20)
        y = y.masked_fill(~valid.any(2)[:, :, None], 0.0)
        return act(y)


class RefLESNet(nn.Module):
    """Reference LESNet: an arbitrary set of entities in, one vector per entity out.

    forward(x, coords, mask) -> (B, N, width)

        x      (B,N,C)   whatever you have per entity
        coords (B,N,2)   coordinates, used to build the pair features
        mask   (B,N)     bool, which slots hold a real entity -- variable N is native, pad freely
        pf     (B,N,N,PF)  optional: a pre-built relation tensor, in which case `coords` is unused

    There is no task head here on purpose.  The output is (B,N,width); attach whatever your task
    needs -- a linear per entity, a masked mean for a set-level label, or nothing at all if this
    block sits inside a bigger model.
    """
    kind = "entity"

    def __init__(self, cin, width=256, depth=5):
        super().__init__()
        dims = [cin] + [width] * depth
        self.layers = nn.ModuleList(RefSoftmaxLayer(dims[k], width) for k in range(depth))

    def forward(self, x, coords, mask, pf=None, nbr=None):
        dty = self.layers[0].A.dtype
        if nbr is not None:                       # list layout: nbr (B,N,K) local indices, -1 = none
            alive = torch.gather(mask, 1, nbr.clamp_min(0).flatten(1)).view_as(nbr)
            nbr = torch.where(alive & mask[:, :, None], nbr, torch.full_like(nbr, -1))
            if pf is None:
                pf = pair_features_nbr(coords, nbr)
            x, pf = x.to(dty), pf.to(dty)
            for layer in self.layers:
                x = layer.forward_list(x, pf, nbr)
            return x
        if pf is None:
            pf, valid = pair_features(coords, mask)
        else:
            valid = pair_mask(mask)
        x = x.to(dty)
        pf = pf.to(dty)
        for layer in self.layers:
            x = layer(x, pf, valid)
        return x


def masked_mean(x, mask, dim=1):
    """Pool a set down to one vector: mean over the entities that are actually present.

    Only useful as a readout for a set-level target.  Non-entity slots are zero in x (the layer
    zeroes them), but they are excluded from the divisor too, so a batch with 3 entities and one
    with 30 both average over what they have rather than over N.
    """
    m = mask.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(dim) / m.sum(dim).clamp_min(1)


def count_params(m):
    return sum(p.numel() for p in m.parameters())
