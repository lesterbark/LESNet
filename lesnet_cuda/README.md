# lesnet_cuda

LESNet on CUDA: the kernels, a model, a float64 oracle, tests and kernel tools.

```
kernels.py      the CUDA extension (all-to-all and list layouts, forward and backward, autograd wrappers)
model.py        LESNet(cin, width, depth): net(x, coords, mask) and net.local(x, coords, mask, k, radius)
neighbors.py    k-nearest and radius partner lists (exact grid search)
ref.py          float64 PyTorch oracle (small tensors only) and the pair-feature definitions
tests.py        values and every gradient against the oracle, masking, equivariance, list equals all-to-all, receivers different from senders
tests_sizes.py  odd shapes: N up to 512, widths up to 2048, relation widths 1 to 8, tiles that do not divide N
bench.py        training-step time against parameter-matched MLP and attention stacks
tools/          build_ext.py, bench_list.py, bench_ops.py, step_bench.py, tune_list.py and others
```

## Use

```python
from model import LESNet

net = LESNet(cin=16, width=128, depth=4).cuda()

y = net(x, coords, mask)                              # all-to-all:  (B, N, 16) -> (B, N, 128)
y = net.local(x, coords, mask, k=24)                  # the 24 nearest partners of every entity
y = net.local(x, coords, mask, k=32, radius=0.07)     # at most 32, none farther than 0.07
```

`x (B, N, cin)` are the entity features, `coords (B, N, 2)` the positions (the layer sees only relative positions through
the pair features) and `mask (B, N)` marks which slots hold an entity (optional). The output has no task head: attach a
linear layer per entity, or `ref.masked_mean` for a set-level target.

Partner lists of your own (a graph, a retrieval pass): `net(x, coords, mask, nbr=idx)` with `idx (B, N, K)` local indices
and `-1` for none. Relations of your own: `net(x, None, mask, pf=pf)` with `pf (B, N, N, P)` (all-to-all) or `(B, N, K, P)`
(lists). The relation is treated as data and receives no gradient.

Lower level: `kernels.les_layer(x, mask, pf, A, B, E, c1, N)` (all-to-all) and
`kernels.les_layer_list(xr, xs, nbr, pf, A, B, E, c1)` (receivers `xr`, senders `xs`; the rows of `nbr` index `xs`).

## Run

```
python tests.py ; python tests_sizes.py        # correctness; both print PASS
python bench.py --baselines                    # step time against MLP and attention stacks
python tools/bench_list.py                     # all-to-all against list layout
```

The test and benchmark scripts expect PyTorch 2.4.1+cu124, `nvcc` and a host compiler; see the top-level README.
