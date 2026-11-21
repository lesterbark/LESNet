"""Partner selection for the list layout: who each entity looks at.

LESNet does not need all-to-all.  Every entity is computed, but each one may be paired with only its
`k` nearest others, or only those within a `radius`.  The filter and the soft-max combine are
unchanged; the partner set is a compute choice, and distance never wires anything inside the layer.

    idx = knn_neighbors(coords, mask, k=16, radius=0.1)      # (B,N,K) local indices, -1 = none
    y   = net(x, coords, mask, nbr=idx)                      # LESNet list path

Two exact searches produce the same lists (nearest first, -1 padded):

* **grid** (used whenever a radius is given, in 2-D, with k <= 64): the entities are sorted into a uniform
  grid whose cell edge is the radius and a CUDA kernel scans the 3x3 cells around each receiver.  O(N)
  for bounded density.
* **brute force** (no radius, or fallback): a (B,R,S) distance matrix per row chunk and a top-k.  O(N^2).

The layer only consumes the (B,N,K) index tensor, so anything else that can make one works as well.
"""
import torch

import kernels

MAX_GRID_K = 64
MAX_CELLS = 1 << 24


def _grid_knn(coords, mask, k, radius):
    B, N, _ = coords.shape
    dev = coords.device
    c = coords.float()
    inf = float("inf")
    cm = mask[..., None]
    xmin = torch.where(cm, c, torch.full_like(c, inf)).amin(1)
    xmax = torch.where(cm, c, torch.full_like(c, -inf)).amax(1)
    ext = torch.nan_to_num(xmax - xmin, nan=0.0, posinf=0.0, neginf=0.0)       # rows with no entity: 0
    G = (ext / radius).floor().long() + 1
    Gx, Gy = int(G[:, 0].max()), int(G[:, 1].max())
    if B * Gx * Gy > MAX_CELLS:
        return None                                                            # sparse cloud: brute force instead
    ncell = Gx * Gy
    cel = ((c - torch.nan_to_num(xmin, posinf=0.0)[:, None]) / radius).floor().long().clamp_min(0)
    b = torch.arange(B, device=dev)[:, None]
    key = b * ncell + cel[..., 1].clamp_max(Gy - 1) * Gx + cel[..., 0].clamp_max(Gx - 1)
    key = torch.where(mask, key, torch.full_like(key, B * ncell)).reshape(-1)
    order = torch.argsort(key)
    cell_start = torch.searchsorted(key[order], torch.arange(B * ncell + 1, device=dev))
    out = kernels.les_knn(c.reshape(-1, 2).contiguous(), order.int(), cell_start.int(), key.int(),
                          mask.reshape(-1).contiguous(), k, N, Gx, Gy, float(radius)).view(B, N, k).long()
    return torch.where(out >= 0, out - b[..., None] * N, out)


def knn_neighbors(coords, mask=None, k=16, radius=None, coords_s=None, mask_s=None,
                  exclude_self=None, chunk=4096, use_grid=True):
    """k nearest valid senders for every receiver, optionally limited to `radius`.

    coords   (B,R,D) receiver positions        mask   (B,R) bool, receivers that exist (default all)
    coords_s (B,S,D) sender positions (default: the receivers themselves, i.e. one set)
    mask_s   (B,S)   bool, senders that exist (default: `mask` when one set, else all)
    exclude_self     drop the receiver itself; defaults to True when there is one set

    Returns idx (B,R,K) int64: a local sender index, or -1 for "no partner" (fewer than k valid
    senders, farther than `radius`, or a receiver that does not exist).  Ordered nearest first.
    """
    B, R, D = coords.shape
    one_set = coords_s is None
    cs = coords if one_set else coords_s
    S = cs.shape[1]
    if exclude_self is None:
        exclude_self = one_set
    if mask is None:
        mask = torch.ones(B, R, dtype=torch.bool, device=coords.device)
    if mask_s is None:
        mask_s = mask if one_set else torch.ones(B, S, dtype=torch.bool, device=coords.device)
    k = min(k, S - 1 if exclude_self else S)
    if use_grid and radius is not None and one_set and D == 2 and exclude_self and k <= MAX_GRID_K \
            and coords.is_cuda and kernels.load_extension():
        out = _grid_knn(coords, mask, k, radius)
        if out is not None:
            return out
    r2 = None if radius is None else float(radius) ** 2
    out = []
    for r0 in range(0, R, chunk):
        c = coords[:, r0:r0 + chunk]
        # exact differences, not the matrix-multiply form of cdist (which is only approximate and
        # moves points across a radius that they sit on)
        d2 = torch.cdist(c.float(), cs.float(), compute_mode="donot_use_mm_for_euclid_dist").pow_(2)
        d2.masked_fill_(~mask_s[:, None, :], float("inf"))
        if exclude_self:
            ar = torch.arange(r0, min(r0 + chunk, R), device=coords.device)
            d2[:, torch.arange(len(ar), device=coords.device), ar] = float("inf")
        dist, idx = torch.topk(d2, k, dim=-1, largest=False)
        ok = torch.isfinite(dist)
        if r2 is not None:
            ok &= dist <= r2
        ok &= mask[:, r0:r0 + chunk, None]
        out.append(torch.where(ok, idx, torch.full_like(idx, -1)))
    return torch.cat(out, 1)


def to_rows(idx, S, mask_s=None):
    """(B,R,K) local sender indices (-1 none) -> (B*R, K) int32 global sender rows for the kernels.

    A sender that is masked out is dropped here too, so a hand-built list cannot point at an entity
    that does not exist."""
    B = idx.shape[0]
    if mask_s is not None:
        alive = torch.gather(mask_s, 1, idx.clamp_min(0).flatten(1)).view_as(idx)
        idx = torch.where(alive, idx, torch.full_like(idx, -1))
    off = (torch.arange(B, device=idx.device) * S).view(B, 1, 1)
    rows = torch.where(idx >= 0, idx + off, torch.full_like(idx, -1))
    return rows.reshape(-1, idx.shape[-1]).to(torch.int32).contiguous()


def neighbor_stats(idx):
    """Mean / min / max number of real partners per receiver -- worth printing once per task."""
    n = (idx >= 0).sum(-1).float()
    return n.mean().item(), n.min().item(), n.max().item()
