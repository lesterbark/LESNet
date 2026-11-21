"""LESNet on the fast CUDA path: a general set architecture.

    net = LESNet(cin, width=256, depth=5)
    y   = net(x, coords, mask)                  # all-to-all:  (B,N,cin) -> (B,N,width)
    y   = net.local(x, coords, mask, k=16)      # k nearest partners per entity
    y   = net.local(x, coords, mask, k=32, radius=0.1)

The architecture is the one in ref.py (linear pair filter, per-slice soft-max combine, leaky at both
ends).  Only the layer implementation differs: these are fused CUDA kernels (kernels.py).
Nothing here knows anything about what your entities are.
"""
import math
import os
import torch
import torch.nn as nn

import kernels
import neighbors
import ref
from ref import PAIR_FEATS, pair_features

SLOPE = 0.1

# Tile sizes (entities one block walks together), 0 = the kernel's measured default.
LES_TI = int(os.environ.get("LES_TI", "0"))
LES_TIB = int(os.environ.get("LES_TIB", "0"))
# Compute the gradient that flows back into the layer's input.  On by default because a general
# block normally has something trainable upstream of it; turning it off saves one matmul and is only
# right when the input really is raw data.
LES_INPUT_GRAD = os.environ.get("LES_INPUT_GRAD", "1") != "0"


class FastSoftmaxLayer(nn.Module):
    def __init__(self, cin, width):
        super().__init__()
        self.A = nn.Parameter(torch.randn(cin, width) * math.sqrt(2.0 / (2 * cin)))
        self.B = nn.Parameter(torch.randn(cin, width) * math.sqrt(2.0 / (2 * cin)))
        self.E = nn.Parameter(torch.randn(PAIR_FEATS, width) * 0.5)
        self.c1 = nn.Parameter(torch.zeros(width))
        self.width = width

    def forward(self, x, mask, pf, N, need_gx=True):
        y = kernels.les_layer(x, mask, pf, self.A, self.B, self.E, self.c1, N,
                              TI=LES_TI, TIB=LES_TIB, need_gx=need_gx)
        return torch.nn.functional.leaky_relu(y, SLOPE)      # the layer's final activation

    def forward_list(self, x, rows, pf, need_gx=True):
        """x (R,C) flattened receivers/senders, rows (R,K) int32 sender rows, pf (R,K,PF)."""
        y = kernels.les_layer_list(x, x, rows, pf, self.A, self.B, self.E, self.c1,
                                   TI=LES_TI, need_gx=need_gx)
        return torch.nn.functional.leaky_relu(y, SLOPE)


class LESNet(nn.Module):
    """LESNet: a set of entities in, one vector per entity out.

        net = LESNet(cin=64, width=256, depth=5)
        y   = net(x, coords, mask)          # (B,N,64) -> (B,N,256)

        x      (B,N,cin)   one feature vector per entity, in any order
        coords (B,N,2)     coordinates, used to build the pair features
        mask   (B,N)       bool, which slots hold a real entity.  Optional if every slot is real.
        nbr    (B,N,K)     optional partner lists (local indices, -1 = none): the entity pairs with
                           these K others instead of all of them.  `net.local(...)` builds them.

    There is no task head: the output is (B,N,width), and what you attach to it is your business.
    A masked mean for a set-level target, a linear for per-entity targets, or nothing at all if
    this block sits inside a bigger model.  Stacking blocks is just calling them in a row.

    `pf` may also be handed in pre-built -- `net(x, None, mask, pf=...)` -- when your relation
    vector comes from somewhere other than coordinates.  It is (B,N,N,PAIR_FEATS) all-to-all, or
    (B,N,K,PAIR_FEATS) with `nbr`.

    `input_grad=False` skips the gradient that flows back into `x`.  That is free speed when `x`
    is raw data with nothing trainable upstream, and wrong the moment it isn't; the default is the
    safe one, so nothing is silently left untrained.
    """
    kind = "entity"

    def __init__(self, cin, width=256, depth=5, input_grad=None):
        super().__init__()
        self.input_grad = LES_INPUT_GRAD if input_grad is None else input_grad
        dims = [cin] + [width] * depth
        self.layers = nn.ModuleList(FastSoftmaxLayer(dims[k], width) for k in range(depth))
        kernels.require_extension()

    @staticmethod
    def _check_pf(pf):
        if pf.requires_grad:
            # The kernels return a gradient for x and for A/B/E/c1 -- but not for pf: the relation
            # tensor is data.  A pf wired to parameters would train nothing and say nothing, which is
            # the worst failure mode, so refuse instead.
            raise ValueError(
                "pf requires grad, but the kernels do not compute a gradient for the relation "
                "tensor, so whatever built it would silently never update.  Pass pf as fixed "
                "data (e.g. from pair_features), or detach it.")

    def forward(self, x, coords=None, mask=None, pf=None, nbr=None):
        B, N, _ = x.shape
        if mask is None:
            mask = torch.ones((B, N), dtype=torch.bool, device=x.device)
        if nbr is not None:
            return self._forward_list(x, coords, mask, pf, nbr)
        if pf is None:
            if coords is None:
                raise ValueError(
                    f"this model was built with pair features {PAIR_FEATS} wide "
                    f"(LES_PAIR_FEATS), so it needs coords (B,N,2) to build them; pass a "
                    f"pre-built pf=... instead if you have no coordinates.")
            pf, _ = pair_features(coords, mask)
        else:
            self._check_pf(pf)
        for k, layer in enumerate(self.layers):
            x = layer(x, mask, pf, N, need_gx=(self.input_grad or k > 0))
        return x

    def _forward_list(self, x, coords, mask, pf, nbr):
        B, N, C = x.shape
        K = nbr.shape[-1]
        if pf is None:
            if coords is None:
                raise ValueError("list layout needs coords (B,N,2) or a pre-built pf (B,N,K,PF)")
            pf = ref.pair_features_nbr(coords, nbr)
        else:
            self._check_pf(pf)
        rows = neighbors.to_rows(nbr, N, mask_s=mask)
        rows = torch.where(mask.reshape(-1, 1), rows, torch.full_like(rows, -1))   # masked receivers
        pf = pf.reshape(B * N, K, -1).float().contiguous()
        h = x.reshape(B * N, C)
        for k, layer in enumerate(self.layers):
            h = layer.forward_list(h, rows, pf, need_gx=(self.input_grad or k > 0))
        return h.view(B, N, -1)

    def local(self, x, coords, mask=None, k=16, radius=None):
        """Each entity pairs with its k nearest others, within `radius` if given (units of coords)."""
        nbr = neighbors.knn_neighbors(coords, mask, k=k, radius=radius)
        return self.forward(x, coords, mask, nbr=nbr)
