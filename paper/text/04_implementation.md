## 4. Implementation

### 4.1 A fused kernel that never stores the pair tensor

A direct implementation materialises `z`, `v` and the softmax weights, each of shape (batch, `N`, `N`, `W`): 68.7 GB (64 GiB) per
tensor for one set with `N=4096, W=1024` in fp32. The
kernel instead walks each receiver's partners once and keeps a running softmax per slice, the
*online* recurrence `m' = max(m, v)`, `ℓ' = ℓ·e^{m−m'} + e^{v−m'}`, `u' = u·e^{m−m'} + v·e^{v−m'}`, and writes
only the combine `s_i = u/ℓ` (before the outer leaky ReLU) together with `m` and `ℓ`, which the backward pass needs. One thread owns one slice, so
partners are read as coalesced rows of `b`. Exactly one of the two rescaling factors is `e^0`, so one
`__expf(-|m-v|)` serves both.

There are two partner layouts in one file (`lesnet_cuda/kernels.py`):

* **all-to-all.** One block walks `TI` receivers of one set against all partners; the partner loop is outer
  and the receiver loop inner, so each partner row `b_j` is loaded once for the whole tile, and the tile's
  pair features are staged in shared memory whenever the tile fits within 48 KB (`TI · N · P · 4 ≤ 48 KB`, falling back to global loads for larger `N`). The default tile sizes are `TI=8` forward and `TI=4` backward.
* **list.** A receiver names its `K` partner rows (`-1` = none). The index is the same for every thread of
  the block, so the branch is warp-uniform; the partner row is a coalesced gather. One block handles one receiver (`TI=1`). The backward scatters
  `∇b` with atomic adds, which coalesce across a warp's adjacent slices.

The packed projection `ab = x·[A | B]` is one matrix multiply, so a layer is three matrix multiplies in total (one forward, two backward) plus one kernel in each direction. The relation width `P` is a runtime value up to 8; widths 4, 6 and 8 use
vector loads in the forward pass, and widths 4 and 8 use `float4` loads in the backward pass.

### 4.2 Verification

Full multi-layer networks (both all-to-all and list layouts, including receivers distinct from senders) are checked against a float64 PyTorch oracle that materialises every tensor, verifying forward values and every parameter and input gradient (`lesnet_cuda/tests.py`). A second kernel-level shape suite (`lesnet_cuda/tests_sizes.py`) checks the raw kernels against a float32 PyTorch reference across ragged `N` and `W`, masked entities, empty receivers, relation widths `P ∈ {1, 3, 4, 5, 6, 7, 8}`, widths above the thread-block limit, and `N` up to 512 (all-to-all) and 4096 (list).

{{table:correctness|Worst-case agreement of the fused CUDA layers with the PyTorch reference (`tests.py`: relative error against a float64 oracle; `tests_sizes.py`: all-to-all forward is max absolute error against a float32 reference, gradients are relative; list layout reports one worst relative error across forward and all gradients in both columns).}}

Forward values and every gradient agree with the float64 oracle to within 2×10⁻⁶ relative, and with the float32 shape suite to under 2×10⁻⁶ ({{ref:correctness}}). The properties of §3.2 and §3.4 are asserted as well: permutation equivariance (5e-7), padded rows reproducing each row run alone (4e-7), the list layout equal to the all-to-all layout when every other entity is listed (3e-7), the grid neighbour search returning exactly the brute-force lists, and invariance to repeating every partner three times (1e-7).

### 4.3 Neighbour search

`neighbors.knn_neighbors` returns each entity's `K` nearest valid partners, nearest first, optionally
only those within a radius, with masks for absent entities and no self-pairs. Two exact searches sit
behind it. **Brute force**: a (batch, `R`, `S`) distance matrix per row chunk and a top-`K`; `O(N²)`. **Grid** (used whenever a radius is given): the
entities are sorted into a uniform grid whose cell edge is the radius (a sort and a `searchsorted` in
PyTorch) and a CUDA kernel, one thread per receiver, scans the 3×3 cells around its own, keeping the `K`
nearest candidates in a small sorted array; `O(N)` at bounded density. The grid search is tested to return
*exactly* the lists a float64 brute-force search returns (including masked entities, tiny radii and `K` close
to `N`). The layer only consumes the (batch, `N`, `K`) index tensor, however it was made.

### 4.4 Numerics and hardware

Computation is in fp32 with `--use_fast_math`. The kernels are written for CUDA cores and do not use tensor cores. All measurements in this paper are on one NVIDIA GTX 1660 Ti (Turing, sm_75, 24
SMs, 6 GB) with an Intel i7-10750H, PyTorch 2.4.1 / CUDA 12.4 and nvcc 12.9. The
backward uses floating-point atomics, so gradients are reproducible only up to summation order.
