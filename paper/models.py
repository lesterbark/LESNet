"""Models for the paper: LESNet (fast CUDA and a flexible PyTorch twin for ablations) and baselines.

Every model is  net(x, coords, mask) -> (B,N,out) for entity tasks or (B,out) for set tasks, with the
same readout head, so only the trunk differs:

    les          LESNet, CUDA kernels (dense all-to-all, or k-nearest lists with k=...)
    les_torch    the same maths in plain PyTorch, with switches for the ablations (combine, activation,
                 hidden pair layer, separate score channel, pair features, k-nearest masks)
    mlp          flatten the set, MLP (fixed N, order dependent): the "no structure" floor
    deepsets     per-entity MLP + mean-pool context (no pair interaction)
    transformer  pre-LN encoder, 4 heads;  transformer_pf adds a learned bias from the pair features
    gnn          message passing on the full graph: message = MLP(h_i, h_j, pf_ij) (one hidden layer),
                 mean aggregation;  gnn_knn restricts the graph to each entity's k nearest

All trunks see [x, coords] as node input except LESNet-with-pair-features, which sees x only (the
relation reaches it through pf).  `fit_width` picks the width that matches a parameter budget.
"""
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lesnet_cuda"))
import kernels  # noqa: E402
import neighbors  # noqa: E402
import ref  # noqa: E402
from tasks import CIN, TASKS  # noqa: E402

SLOPE = 0.1
ACTS = {
    "leaky": lambda x: F.leaky_relu(x, SLOPE),
    "relu": F.relu,
    "none": lambda x: x,
    "tanh": torch.tanh,
    "gelu": F.gelu,
}


# ------------------------------------------------------------------------------ pair features
def parts_of(spec):
    return [] if spec in ("none", "", None) else ref.pair_parts(spec)


def pf_width(spec):
    p = parts_of(spec)
    return max(1, sum(ref.PARTS[q] for q in p))


def pf_dense(coords, spec):
    B, N, _ = coords.shape
    parts = parts_of(spec)
    if not parts:
        return torch.zeros(B, N, N, 1, device=coords.device)
    cx, cy = coords[..., 0], coords[..., 1]
    return ref._stack_pair_parts(parts, cx[:, :, None] - cx[:, None, :], cy[:, :, None] - cy[:, None, :])


def pf_list(coords, nbr, spec):
    B, N, K = nbr.shape
    parts = parts_of(spec)
    if not parts:
        return torch.zeros(B, N, K, 1, device=coords.device)
    cj = torch.gather(coords, 1, nbr.clamp_min(0).reshape(B, N * K, 1).expand(-1, -1, 2)).view(B, N, K, 2)
    return ref._stack_pair_parts(parts, coords[:, :, None, 0] - cj[..., 0], coords[:, :, None, 1] - cj[..., 1])


def knn_mask(coords, mask, k):
    """(B,N,N) bool: partner j is among i's k nearest valid others."""
    nbr = neighbors.knn_neighbors(coords, mask, k=k)
    B, N, K = nbr.shape
    m = torch.zeros(B, N, N + 1, dtype=torch.bool, device=coords.device)
    m.scatter_(2, torch.where(nbr >= 0, nbr, torch.full_like(nbr, N)), True)
    return m[:, :, :N]


# ------------------------------------------------------------------------------ LESNet (CUDA)
class LESTrunk(nn.Module):
    """LESNet layers on the CUDA kernels.  `k=None` is all-to-all, `k=int` the k-nearest list layout."""
    def __init__(self, cin, width, depth, pf_spec="pos,axis", k=None, radius=None, reinject=False):
        super().__init__()
        kernels.require_extension()
        self.pf_spec, self.k, self.radius = pf_spec, k, radius
        self.reinject = reinject                         # the input entity variables are concatenated onto every later layer's input
        self.abs_coords = not parts_of(pf_spec)          # no relation given -> geometry must come in as features
        cin = cin + (2 if self.abs_coords else 0)
        PF = pf_width(pf_spec)
        dims = [cin] + [width + (cin if reinject else 0)] * (depth - 1)
        self.A = nn.ParameterList(); self.B = nn.ParameterList()
        self.E = nn.ParameterList(); self.c = nn.ParameterList()
        for l in range(depth):
            self.A.append(nn.Parameter(torch.randn(dims[l], width) * math.sqrt(2.0 / (2 * dims[l]))))
            self.B.append(nn.Parameter(torch.randn(dims[l], width) * math.sqrt(2.0 / (2 * dims[l]))))
            self.E.append(nn.Parameter(torch.randn(PF, width) * 0.5))
            self.c.append(nn.Parameter(torch.zeros(width)))
        self.width = width

    def forward(self, x, coords, mask):
        B, N, _ = x.shape
        if self.abs_coords:
            x = torch.cat([x, coords], -1)
        if self.k is None:
            pf = pf_dense(coords, self.pf_spec).contiguous()
            x0 = x
            for l in range(len(self.A)):
                if self.reinject and l > 0:
                    x = torch.cat([x, x0], -1)
                x = F.leaky_relu(kernels.les_layer(x, mask, pf, self.A[l], self.B[l], self.E[l], self.c[l], N,
                                                   need_gx=(l > 0)), SLOPE)
            return x
        nbr = neighbors.knn_neighbors(coords, mask, k=self.k, radius=self.radius)
        pf = pf_list(coords, nbr, self.pf_spec).reshape(B * N, nbr.shape[-1], -1).contiguous()
        rows = neighbors.to_rows(nbr, N, mask_s=mask)
        h = x.reshape(B * N, -1)
        h0 = h
        for l in range(len(self.A)):
            if self.reinject and l > 0:
                h = torch.cat([h, h0], -1)
            h = F.leaky_relu(kernels.les_layer_list(h, h, rows, pf, self.A[l], self.B[l], self.E[l], self.c[l],
                                                    need_gx=(l > 0)), SLOPE)
        return h.view(B, N, -1)


# ------------------------------------------------------------------------------ LESNet (PyTorch twin)
class TorchLESLayer(nn.Module):
    """The LES layer in plain PyTorch with every ablation switch.  O(N^2 W) memory: small N only.

    combine: softmax (the LESNet default) | sum | mean | max | scalar (sum * learned alpha) | lse
    sep:     a separate per-slice score channel instead of using v as its own score (attention-style)
    hidden:  a Linear(W, W) + activation on every pair before the combine (the filter's hidden layer)
    """
    def __init__(self, cin, width, PF, combine="softmax", act="leaky", hidden=False, sep=False, use_pf=True, init=1.0):
        super().__init__()
        self.A = nn.Parameter(torch.randn(cin, width) * init * math.sqrt(2.0 / (2 * cin)))
        self.B = nn.Parameter(torch.randn(cin, width) * init * math.sqrt(2.0 / (2 * cin)))
        self.E = nn.Parameter(torch.randn(PF, width) * init * 0.5) if use_pf else None
        self.c = nn.Parameter(torch.zeros(width))
        self.combine, self.act, self.hidden, self.sep = combine, ACTS[act], hidden, sep
        if hidden:
            self.H = nn.Linear(width, width)
        if sep:
            self.A2 = nn.Parameter(torch.randn(cin, width) * math.sqrt(2.0 / (2 * cin)))
            self.B2 = nn.Parameter(torch.randn(cin, width) * math.sqrt(2.0 / (2 * cin)))
            self.E2 = nn.Parameter(torch.randn(PF, width) * 0.5) if use_pf else None
        if combine == "scalar":
            self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, pf, valid):
        z = (x @ self.A)[:, :, None, :] + (x @ self.B)[:, None, :, :] + self.c
        if self.E is not None:
            z = z + pf @ self.E
        v = self.act(z)
        if self.hidden:
            v = self.act(self.H(v))
        vm = valid[..., None]
        neg = torch.finfo(v.dtype).min / 4
        if self.combine == "softmax":
            if self.sep:
                s = (x @ self.A2)[:, :, None, :] + (x @ self.B2)[:, None, :, :]
                if self.E2 is not None:
                    s = s + pf @ self.E2
                s = s.masked_fill(~vm, neg)
            else:
                s = v.masked_fill(~vm, neg)
            w = torch.softmax(s, 2) * vm
            y = (w * v).sum(2)
        elif self.combine in ("sum", "mean", "scalar"):
            y = (v * vm).sum(2)
            if self.combine == "mean":
                y = y / vm.sum(2).clamp_min(1)
            elif self.combine == "scalar":
                y = y * self.alpha
        elif self.combine == "max":
            y = v.masked_fill(~vm, neg).max(2).values
        elif self.combine == "lse":
            y = torch.logsumexp(v.masked_fill(~vm, neg), 2)
        else:
            raise KeyError(self.combine)
        y = torch.where(valid.any(2)[..., None], y, torch.zeros_like(y))
        return self.act(y)


class LESTorchTrunk(nn.Module):
    def __init__(self, cin, width, depth, pf_spec="pos,axis", k=None, **layer_kw):
        super().__init__()
        self.pf_spec, self.k = pf_spec, k
        self.abs_coords = not parts_of(pf_spec)
        cin = cin + (2 if self.abs_coords else 0)
        PF = pf_width(pf_spec)
        dims = [cin] + [width] * depth
        self.layers = nn.ModuleList(
            TorchLESLayer(dims[l], width, PF, use_pf=bool(parts_of(pf_spec)), **layer_kw) for l in range(depth))
        self.width = width

    def forward(self, x, coords, mask):
        if self.abs_coords:
            x = torch.cat([x, coords], -1)
        eye = torch.eye(x.shape[1], dtype=torch.bool, device=x.device)
        valid = mask[:, :, None] & mask[:, None, :] & ~eye
        if self.k is not None:
            valid = valid & knn_mask(coords, mask, self.k)
        pf = pf_dense(coords, self.pf_spec)
        for layer in self.layers:
            x = layer(x, pf, valid)
        return x


# ------------------------------------------------------------------------------ baselines
class MLPFlatTrunk(nn.Module):
    """Flatten the whole set into one vector.  Needs a fixed N and treats slots as ordered."""
    def __init__(self, cin, width, depth, nmax):
        super().__init__()
        self.nmax, self.width = nmax, width
        dims = [nmax * (cin + 2)] + [width] * (depth - 1) + [nmax * width]
        self.net = nn.Sequential(*[m for i in range(len(dims) - 1)
                                   for m in (nn.Linear(dims[i], dims[i + 1]), nn.LeakyReLU(SLOPE))])

    def forward(self, x, coords, mask):
        B, N, _ = x.shape
        if N != self.nmax:
            raise ValueError("flat MLP needs the training N")
        h = torch.cat([x, coords], -1).reshape(B, -1)
        return self.net(h).view(B, N, self.width) * mask[..., None]


class DeepSetsTrunk(nn.Module):
    def __init__(self, cin, width, depth):
        super().__init__()
        self.inp = nn.Linear(cin + 2, width)
        self.blocks = nn.ModuleList(nn.Linear(2 * width, width) for _ in range(depth))
        self.width = width

    def forward(self, x, coords, mask):
        m = mask[..., None].float()
        h = F.leaky_relu(self.inp(torch.cat([x, coords], -1)), SLOPE) * m
        for blk in self.blocks:
            pooled = h.sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp_min(1)
            h = F.leaky_relu(blk(torch.cat([h, pooled.expand_as(h)], -1)), SLOPE) * m
        return h


class TransformerTrunk(nn.Module):
    def __init__(self, cin, width, depth, heads=4, pf_spec=None):
        super().__init__()
        assert width % heads == 0
        self.heads, self.pf_spec, self.width = heads, pf_spec, width
        self.inp = nn.Linear(cin + 2, width)
        self.norm1 = nn.ModuleList(nn.LayerNorm(width) for _ in range(depth))
        self.qkv = nn.ModuleList(nn.Linear(width, 3 * width) for _ in range(depth))
        self.proj = nn.ModuleList(nn.Linear(width, width) for _ in range(depth))
        self.norm2 = nn.ModuleList(nn.LayerNorm(width) for _ in range(depth))
        self.ffn = nn.ModuleList(nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))
                                 for _ in range(depth))
        if pf_spec:
            self.bias = nn.ModuleList(nn.Linear(pf_width(pf_spec), heads) for _ in range(depth))

    def forward(self, x, coords, mask):
        B, N, _ = x.shape
        h = self.inp(torch.cat([x, coords], -1))
        keep = (mask[:, None, :] & mask[:, :, None] | torch.eye(N, dtype=torch.bool, device=x.device))
        keep = keep[:, None]                                                  # (B,1,N,N); a transformer keeps self
        pf = pf_dense(coords, self.pf_spec) if self.pf_spec else None
        for l in range(len(self.qkv)):
            q, k, v = self.qkv[l](self.norm1[l](h)).chunk(3, -1)
            sp = lambda t: t.view(B, N, self.heads, -1).transpose(1, 2)
            bias = torch.zeros(B, 1, N, N, device=x.device)
            if pf is not None:
                bias = self.bias[l](pf).permute(0, 3, 1, 2)                   # (B,H,N,N)
            attn_mask = bias.masked_fill(~keep, float("-inf"))
            o = F.scaled_dot_product_attention(sp(q), sp(k), sp(v), attn_mask=attn_mask)
            h = h + self.proj[l](o.transpose(1, 2).reshape(B, N, -1))
            h = h + self.ffn[l](self.norm2[l](h))
        return h * mask[..., None]


class GNNTrunk(nn.Module):
    """Message passing, full graph or k-nearest.  message = Linear(act(A h_i + B h_j + E pf_ij)) -- the
    pair step is the same as LESNet's plus the hidden layer LESNet leaves out; aggregation is the mean."""
    def __init__(self, cin, width, depth, pf_spec="pos,axis", k=None):
        super().__init__()
        self.pf_spec, self.k, self.width = pf_spec, k, width
        PF = pf_width(pf_spec)
        self.inp = nn.Linear(cin + 2, width)
        self.A = nn.ModuleList(nn.Linear(width, width) for _ in range(depth))
        self.B = nn.ModuleList(nn.Linear(width, width, bias=False) for _ in range(depth))
        self.E = nn.ModuleList(nn.Linear(PF, width, bias=False) for _ in range(depth))
        self.M = nn.ModuleList(nn.Linear(width, width) for _ in range(depth))
        self.U = nn.ModuleList(nn.Linear(2 * width, width) for _ in range(depth))

    def forward(self, x, coords, mask):
        eye = torch.eye(x.shape[1], dtype=torch.bool, device=x.device)
        valid = mask[:, :, None] & mask[:, None, :] & ~eye
        if self.k is not None:
            valid = valid & knn_mask(coords, mask, self.k)
        pf = pf_dense(coords, self.pf_spec)
        vm = valid[..., None].float()
        cnt = vm.sum(2).clamp_min(1)
        h = F.leaky_relu(self.inp(torch.cat([x, coords], -1)), SLOPE) * mask[..., None]
        for l in range(len(self.A)):
            z = self.A[l](h)[:, :, None, :] + self.B[l](h)[:, None, :, :] + self.E[l](pf)
            msg = F.leaky_relu(self.M[l](F.leaky_relu(z, SLOPE)), SLOPE)
            agg = (msg * vm).sum(2) / cnt
            h = h + F.leaky_relu(self.U[l](torch.cat([h, agg], -1)), SLOPE)
            h = h * mask[..., None]
        return h


class GenTrunk(nn.Module):
    """Softmax-aggregation message passing in the style of DeeperGCN (GENConv): message m_ij = relu(B h_j + E pf_ij) + eps,
    aggregated per channel by sum_j softmax_j(beta * m_ij) m_ij with a learnable beta, then a two-layer node update on h_i + agg."""
    def __init__(self, cin, width, depth, pf_spec="pos,axis", k=None):
        super().__init__()
        self.pf_spec, self.k, self.width = pf_spec, k, width
        PF = pf_width(pf_spec)
        self.inp = nn.Linear(cin + 2, width)
        self.B = nn.ModuleList(nn.Linear(width, width) for _ in range(depth))
        self.E = nn.ModuleList(nn.Linear(PF, width, bias=False) for _ in range(depth))
        self.beta = nn.Parameter(torch.ones(depth))
        self.U1 = nn.ModuleList(nn.Linear(width, width) for _ in range(depth))
        self.U2 = nn.ModuleList(nn.Linear(width, width) for _ in range(depth))

    def forward(self, x, coords, mask):
        eye = torch.eye(x.shape[1], dtype=torch.bool, device=x.device)
        valid = mask[:, :, None] & mask[:, None, :] & ~eye
        if self.k is not None:
            valid = valid & knn_mask(coords, mask, self.k)
        pf = pf_dense(coords, self.pf_spec)
        vm = valid[..., None]
        h = F.leaky_relu(self.inp(torch.cat([x, coords], -1)), SLOPE) * mask[..., None]
        for l in range(len(self.B)):
            m = F.relu(self.B[l](h)[:, None, :, :] + self.E[l](pf)) + 1e-7
            s = (self.beta[l] * m).masked_fill(~vm, float("-inf"))
            w = torch.softmax(s, dim=2)
            w = torch.nan_to_num(w, nan=0.0)                           # receivers with no valid partner
            agg = (w * m).sum(2)
            h = h + self.U2[l](F.leaky_relu(self.U1[l](h + agg), SLOPE))
            h = h * mask[..., None]
        return h


class GatherGNNTrunk(nn.Module):
    """GNS-style message passing on the same neighbour lists: message = Linear(act(A h_i + B h_j + E pf)),
    mean aggregation, residual node update.  The pair step carries the hidden layer LESNet leaves out."""
    def __init__(self, cin, width, depth, pf_spec, k, radius, residual=True):
        super().__init__()
        self.k, self.radius, self.pf_spec, self.width = k, radius, pf_spec, width
        self.residual = residual                                          # False: h = update(h, agg), no skip path
        PF = pf_width(pf_spec)
        self.inp = nn.Linear(cin, width)
        self.A = nn.ModuleList(nn.Linear(width, width) for _ in range(depth))
        self.B = nn.ModuleList(nn.Linear(width, width, bias=False) for _ in range(depth))
        self.E = nn.ModuleList(nn.Linear(PF, width, bias=False) for _ in range(depth))
        self.M = nn.ModuleList(nn.Linear(width, width) for _ in range(depth))
        self.U = nn.ModuleList(nn.Linear(2 * width, width) for _ in range(depth))

    def forward(self, x, coords, mask):
        Bn, N, _ = x.shape
        nbr = neighbors.knn_neighbors(coords, mask, k=self.k, radius=self.radius)
        valid = (nbr >= 0)[..., None].float()
        cnt = valid.sum(2).clamp_min(1.0)
        pf = pf_list(coords, nbr, self.pf_spec)
        bi = torch.arange(Bn, device=x.device)[:, None, None]
        idx = nbr.clamp_min(0)
        h = F.leaky_relu(self.inp(x), 0.1) * mask[..., None]
        for l in range(len(self.A)):
            hj = self.B[l](h)[bi, idx]                                   # (B,N,K,W)
            z = self.A[l](h)[:, :, None, :] + hj + self.E[l](pf)
            msg = F.leaky_relu(self.M[l](F.leaky_relu(z, 0.1)), 0.1)
            agg = (msg * valid).sum(2) / cnt
            upd = F.leaky_relu(self.U[l](torch.cat([h, agg], -1)), 0.1)
            h = ((h + upd) if self.residual else upd) * mask[..., None]
        return h




class GatherGenTrunk(nn.Module):
    """GENConv-style softmax aggregation on the neighbour lists (see GenTrunk): message relu(B h_j + E pf) + eps, per-channel
    softmax(beta * message) weighted sum over the K partners, residual two-layer node update.  The pair stage is recomputed in the
    backward pass (checkpointing), because storing every (B,N,K,W) intermediate does not fit a 6 GB card at the jet batch size."""
    def __init__(self, cin, width, depth, pf_spec, k, radius):
        super().__init__()
        self.k, self.radius, self.pf_spec, self.width = k, radius, pf_spec, width
        PF = pf_width(pf_spec)
        self.inp = nn.Linear(cin, width)
        self.B = nn.ModuleList(nn.Linear(width, width) for _ in range(depth))
        self.E = nn.ModuleList(nn.Linear(PF, width, bias=False) for _ in range(depth))
        self.beta = nn.Parameter(torch.ones(depth))
        self.U1 = nn.ModuleList(nn.Linear(width, width) for _ in range(depth))
        self.U2 = nn.ModuleList(nn.Linear(width, width) for _ in range(depth))

    def _pair(self, l, h, pf, idx, valid):
        Bn = h.shape[0]
        bi = torch.arange(Bn, device=h.device)[:, None, None]
        m = F.relu(self.B[l](h)[bi, idx] + self.E[l](pf)).clamp_min(1e-7)                 # (B,N,K,W)
        w = torch.softmax((self.beta[l] * m).masked_fill(~valid, -1e30), dim=2)
        return (w * m).sum(2)

    def forward(self, x, coords, mask):
        from torch.utils.checkpoint import checkpoint
        nbr = neighbors.knn_neighbors(coords, mask, k=self.k, radius=self.radius)
        valid = (nbr >= 0)[..., None]
        pf = pf_list(coords, nbr, self.pf_spec)
        idx = nbr.clamp_min(0)
        h = F.leaky_relu(self.inp(x), 0.1) * mask[..., None]
        for l in range(len(self.B)):
            if torch.is_grad_enabled():
                agg = checkpoint(self._pair, l, h, pf, idx, valid, use_reentrant=False)
            else:
                agg = self._pair(l, h, pf, idx, valid)
            h = (h + self.U2[l](F.leaky_relu(self.U1[l](h + agg), 0.1))) * mask[..., None]
        return h


# ------------------------------------------------------------------------------ assembled network
class Net(nn.Module):
    def __init__(self, trunk, task, head_hidden=True):
        super().__init__()
        kind, out = TASKS[task]["kind"], TASKS[task]["out"]
        W = trunk.width
        self.trunk, self.kind = trunk, kind
        if kind == "set_cls":
            self.head = nn.Sequential(nn.Linear(2 * W, W), nn.LeakyReLU(SLOPE), nn.Linear(W, out))
        else:
            self.head = nn.Linear(W, out)

    def forward(self, x, coords, mask):
        h = self.trunk(x, coords, mask)
        if self.kind == "set_cls":
            m = mask[..., None].float()
            mean = (h * m).sum(1) / m.sum(1).clamp_min(1)
            mx = h.masked_fill(~mask[..., None], torch.finfo(h.dtype).min / 4).max(1).values
            return self.head(torch.cat([mean, mx], -1))
        return self.head(h)


def build_trunk(spec, width, depth, nmax):
    fam = spec["family"]
    kw = {k: v for k, v in spec.items() if k not in ("family", "label")}
    if fam == "les":
        return LESTrunk(CIN, width, depth, **kw)
    if fam == "les_torch":
        return LESTorchTrunk(CIN, width, depth, **kw)
    if fam == "mlp":
        return MLPFlatTrunk(CIN, width, depth, nmax)
    if fam == "deepsets":
        return DeepSetsTrunk(CIN, width, depth)
    if fam == "transformer":
        return TransformerTrunk(CIN, width, depth, **kw)
    if fam == "gnn":
        return GNNTrunk(CIN, width, depth, **kw)
    if fam == "gen":
        return GenTrunk(CIN, width, depth, **kw)
    raise KeyError(fam)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


_width_cache = {}


def fit_width(spec, depth, nmax, task, target):
    """Width (multiple of 4, of 8 for transformers) whose full network is closest to `target` params."""
    key = (repr(sorted(spec.items())), depth, nmax, task, target)
    if key in _width_cache:
        return _width_cache[key]
    step = 8 if spec["family"] == "transformer" else 4
    def params(w):
        return count_params(Net(build_trunk(spec, w, depth, nmax), task))
    lo, hi = 1, 256                      # width in units of `step`; parameter count is monotone in width
    while params(hi * step) < target and hi < 4096:
        hi *= 2
    while lo < hi:                       # smallest width whose count reaches the target
        mid = (lo + hi) // 2
        if params(mid * step) < target:
            lo = mid + 1
        else:
            hi = mid
    cands = [w for w in (lo - 1, lo) if w >= 1]
    bw = min(cands, key=lambda w: abs(params(w * step) - target)) * step
    _width_cache[key] = bw
    return bw
