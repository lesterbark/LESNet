"""Synthetic entity tasks with exact, known targets, generated on the GPU each step.

Every task draws a world of n entities (n varies per row, padded to nmax with a mask) in a square
whose side grows as sqrt(n), so the DENSITY is constant (about 16 entities per unit area) no matter
how many entities there are.  That is what makes the out-of-distribution-N test clean for local tasks:
the same radius contains the same number of neighbours at n = 16 and n = 96.

Entity features x (cin = 5): one-hot type (A = the single "ball", B, C) and a 2-d velocity.
Coordinates are separate; models that need absolute coordinates as features get them concatenated.

    kind        output                 metric
    entity_cls  class per entity       accuracy
    entity_reg  vector per entity      R^2 over valid entities
    set_cls     class per set          accuracy

The tasks are chosen to separate what the architectures can and cannot do:

    nn_type        type of my nearest neighbour                   local, relational, argmin-like
    count_r        how many entities are within r of me           COUNTING (softmax is not a counter)
    rank_ball      how many entities are closer to the ball       rank (a count of a pairwise comparison)
    closest_team   which team owns the entity nearest the ball    global, argmin-like, set-level
    exist_pair     is any B within r of any C                     global existence, max-like, set-level
    local_vel_avg  mean velocity of entities within r of me       local smoothing (SPH-like average)
    nbody_acc      softened gravity from every other entity       SUPERPOSITION (a sum over all pairs)
"""
import math
import torch

CIN = 5
DENSITY_N = 16.0           # entities per unit area (side L = sqrt(n / DENSITY_N))
R_COUNT = 0.25
R_PAIR = 0.07
R_VEL = 0.3
NBODY_EPS = 0.2           # softening: keeps the force finite (heavy tails otherwise dominate R^2)

TASKS = {
    "nn_type": dict(kind="entity_cls", out=3),
    "count_r": dict(kind="entity_reg", out=1),
    "rank_ball": dict(kind="entity_reg", out=1),
    "closest_team": dict(kind="set_cls", out=2),
    "exist_pair": dict(kind="set_cls", out=2),
    "local_vel_avg": dict(kind="entity_reg", out=2),
    "nbody_acc": dict(kind="entity_reg", out=2),
}
LOCAL_TASKS = ("nn_type", "count_r", "local_vel_avg")      # N-invariant by construction


def make_world(B, n_lo, n_hi, nmax, gen, device):
    n = torch.randint(n_lo, n_hi + 1, (B,), generator=gen, device=device)
    ar = torch.arange(nmax, device=device)
    mask = ar[None, :] < n[:, None]
    L = torch.sqrt(n.float() / DENSITY_N)[:, None, None]
    coords = torch.rand(B, nmax, 2, generator=gen, device=device) * L
    typ = torch.randint(1, 3, (B, nmax), generator=gen, device=device)      # B=1 or C=2
    ball = (torch.rand(B, nmax, generator=gen, device=device) * mask).argmax(1)   # one ball per row
    typ[torch.arange(B, device=device), ball] = 0
    vel = torch.randn(B, nmax, 2, generator=gen, device=device)
    onehot = torch.nn.functional.one_hot(typ, 3).float()
    x = torch.cat([onehot, vel], -1) * mask[..., None]
    coords = coords * mask[..., None]
    return dict(x=x, coords=coords, mask=mask, typ=typ, vel=vel, ball=ball, n=n)


def _dist(w):
    c, m = w["coords"], w["mask"]
    D = torch.cdist(c, c)
    eye = torch.eye(c.shape[1], dtype=torch.bool, device=c.device)
    D = D.masked_fill(~(m[:, :, None] & m[:, None, :]) | eye, float("inf"))
    return D


def targets(task, w):
    """-> y with shape (B,N) long / (B,N,out) float for entity tasks, (B,) long for set tasks."""
    D, mask, typ = _dist(w), w["mask"], w["typ"]
    B, N = mask.shape
    bi = torch.arange(B, device=mask.device)
    if task == "nn_type":
        nn_idx = D.argmin(-1)
        return torch.gather(typ, 1, nn_idx)
    if task == "count_r":
        return ((D < R_COUNT).sum(-1).float() / 4.0)[..., None]
    if task == "rank_ball":
        dball = D[bi, w["ball"]]                                   # (B,N) distance of each to the ball
        closer = (dball[:, None, :] < dball[:, :, None]) & torch.isfinite(dball[:, None, :])
        closer &= mask[:, None, :]
        rank = closer.sum(-1).float()
        rank = rank.masked_fill(torch.arange(N, device=mask.device)[None] == w["ball"][:, None], 0.0)
        return (rank / 6.0)[..., None]
    if task == "closest_team":
        dball = D[bi, w["ball"]].masked_fill(typ == 0, float("inf"))
        j = dball.argmin(-1)
        return (typ[bi, j] - 1).long()                              # 0 = team B, 1 = team C
    if task == "exist_pair":
        isb, isc = (typ == 1) & mask, (typ == 2) & mask
        near = (D < R_PAIR) & isb[:, :, None] & isc[:, None, :]
        return near.flatten(1).any(1).long()
    if task == "local_vel_avg":
        near = (D < R_VEL).float()
        cnt = near.sum(-1, keepdim=True)
        avg = (near @ w["vel"]) / cnt.clamp_min(1.0)
        return avg
    if task == "nbody_acc":
        c = w["coords"]
        d = c[:, None, :, :] - c[:, :, None, :]                      # (B,i,j,2): from i towards j
        r2 = (d ** 2).sum(-1) + NBODY_EPS ** 2
        m = torch.where(typ == 0, 4.0, 1.0) * mask
        valid = (mask[:, :, None] & mask[:, None, :]) & ~torch.eye(N, dtype=torch.bool, device=mask.device)
        f = d * (m[:, None, :] * valid / r2.pow(1.5))[..., None]
        return f.sum(2) / 4.0
    raise KeyError(task)


def sample(task, B, n_lo, n_hi, nmax, gen, device="cuda"):
    w = make_world(B, n_lo, n_hi, nmax, gen, device)
    w["y"] = targets(task, w)
    return w


def majority_baseline(task, w):
    """Accuracy of always answering the most common class (classification tasks)."""
    kind = TASKS[task]["kind"]
    y = w["y"]
    if kind == "set_cls":
        return max((y == c).float().mean().item() for c in range(TASKS[task]["out"]))
    if kind == "entity_cls":
        m = w["mask"]
        return max(((y == c) & m).sum().item() / m.sum().item() for c in range(TASKS[task]["out"]))
    return float("nan")


def score(task, pred, w):
    """Higher is better: accuracy for classification, R^2 for regression (over valid entities)."""
    kind, m, y = TASKS[task]["kind"], w["mask"], w["y"]
    if kind == "set_cls":
        return (pred.argmax(-1) == y).float().mean().item()
    if kind == "entity_cls":
        return ((pred.argmax(-1) == y) & m).sum().item() / m.sum().item()
    mm = m[..., None].float()
    err = (((pred - y) ** 2) * mm).sum()
    mean = (y * mm).sum((0, 1)) / mm.sum()
    tot = (((y - mean) ** 2) * mm).sum()
    return 1.0 - (err / tot.clamp_min(1e-12)).item()


def loss_fn(task, pred, w):
    kind, m, y = TASKS[task]["kind"], w["mask"], w["y"]
    if kind == "set_cls":
        return torch.nn.functional.cross_entropy(pred, y)
    if kind == "entity_cls":
        l = torch.nn.functional.cross_entropy(pred.transpose(1, 2), y, reduction="none")
        return (l * m).sum() / m.sum()
    mm = m[..., None].float()
    return (((pred - y) ** 2) * mm).sum() / (mm.sum() * y.shape[-1])
