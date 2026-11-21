## 3. The LESNet layer

### 3.1 Definition

A layer maps a set of up to `N` entities with features `x_i ∈ R^C` to `y_i ∈ R^W`. For every ordered pair `(i, j)`
with `j` in the partner set `𝒫(i)` it is given a *relation vector* `pf_ij ∈ R^P` describing entity `j` as
seen from entity `i`. The parameters are `A, B ∈ R^{C×W}`, `E ∈ R^{P×W}` and `c ∈ R^W`. For each slice
`k = 1..W`, with `a_i = x_i A` and `b_j = x_j B`:

```
z_ij[k] = (x_i A)[k] + (x_j B)[k] + (pf_ij E)[k] + c[k]       pair filter (one linear map)
v_ij[k] = leaky(z_ij[k])                                      slope 0.1
w_ij[k] = exp(v_ij[k]) / Σ_{j' ∈ 𝒫(i)} exp(v_ij'[k])          softmax over the partners, per slice
y_i[k]  = leaky( Σ_{j ∈ 𝒫(i)} w_ij[k] · v_ij[k] )             softmax combine
```

{{fig:architecture.png|One LESNet slice. (a) A receiver (red) and its partners (blue). (b) The pair value is one linear map of the receiver's features, the partner's features and the relation, followed by a leaky ReLU. (c) The weights are the softmax of the values over the partners; the slice output is the weighted sum of the values passed through a leaky ReLU.}}

An entity with no valid partner outputs zero. In the all-to-all layout, `𝒫(i)` is the set of all valid entities other than `i`; custom partner lists may include `i` itself when a task calls for it (§6.3, §6.4). A network is a stack of layers; the output has shape `(N, W)` and feeds a task head (a linear layer per entity, or a pooled readout for a set-level target). `x_i A` and `x_j B` are computed once per entity (`O(NCW)`); per `(i, j, k)` the pair step sums them, adds the relation term (`P` multiply-adds) and applies one activation.

### 3.2 Properties

**Permutation equivariance.** `𝒫(i)` is a set and both the softmax and the sum run over it, so
reordering the entities reorders the output and changes nothing else. (With neighbour lists this holds
when the lists are built from geometry, up to ties.)

**Variable set size.** `N` is the maximum set size. A boolean mask marks which slots hold an
entity; masked slots are never partners, never receive and stay exactly zero, so the rows of one batch may hold different
numbers of entities.

**Cost.** All-to-all: `O(N²W)` per layer. With neighbour lists of length `K`: `O(NKW)`. Memory is
`O(NW)` with the fused kernel and neighbour lists (plus `O(NKP)` for the relation vectors), against `O(N²W)` for a direct implementation. The all-to-all layout additionally holds the relation tensor, `O(N²P)`.

**Translation invariance.** The pair filter depends on positions only through `pf`. If `pf` is built from coordinate differences and the entity features exclude absolute coordinates, the layer is invariant to translating all positions.

### 3.3 Relation to attention, and what a slice can select

Multi-head attention computes, per head, `Σ_j softmax_j(s_ij) u_j` with a scalar score `s_ij = q_i·k_j` and
a value `u_j` that depends on the partner only. A LESNet slice is the same recipe with a head of
dimension one in which the score and the value are one quantity, `v_ij[k]`, and that quantity
depends on the pair. There are `W` such heads per layer.

Because leaky ReLU is strictly increasing, the ranking of the partners `j` by `v_ij[k]` is the ranking by
`z_ij[k]`, and `(x_i A)[k]` is the same for every `j`:

> For a fixed receiver `i`, the order in which slice `k` ranks its partners is the order of
> `(x_j B)[k] + (pf_ij E)[k]`. It does not depend on `x_i`.

The receiver's own features set how sharply the softmax concentrates (they shift `z` across the
leaky ReLU's kink) and how large the output is. Selection is driven by the partner's features
and by the relation. Without a relation vector, every receiver ranks its partners identically within a
slice (by `x_j B` alone), so a single slice cannot rank its partners by their relation to the receiver from
absolute coordinates. This is the role of `pf`; §5.3 measures the effect of removing it.

### 3.4 What one layer computes: the sharp and the diffuse limit

Let `z_ij = (x_i A)[k] + (x_j B)[k] + (pf_ij E)[k] + c[k]` be the pair filter of one slice. Leaky ReLU is positively homogeneous, so scaling the parameters of the slice by `λ > 0` scales `v` by `λ`. The limits `λ → ∞` and `λ → 0` have closed forms.

**Sharp limit.** As `λ → ∞` the softmax weights concentrate on the partner with the largest `v`, and since leaky ReLU is increasing the largest `v` belongs to the largest `z`:

```
y_i[k] / λ  →  φ( (x_i A)[k] + c[k] + max_{j ∈ 𝒫(i)} [ (x_j B)[k] + (pf_ij E)[k] ] ),     φ = leaky ∘ leaky.
```

The slice is a max-plus operator over the partners: a maximum of terms that are linear in the partner's features and in the relation, shifted by a term that depends on the receiver alone. A maximum is not a sum: the limit cannot accumulate contributions from several partners. The entity features `x_i` and `x_j` also never multiply each other inside the pair filter, which is a sum of a receiver term, a partner term and a relation term.

**Diffuse limit.** As `λ → 0` the weights become uniform and the combine `s_i[k] = Σ_j w_ij[k] v_ij[k]` expands as the mean `(1/|𝒫(i)|) Σ_j v_ij[k]` of the leaky pair values plus their variance over the partners (of order `λ²`, while the mean is of order `λ`), up to `O(λ³)`. Passing through the outer activation gives `y_i[k] / λ → leaky((1/|𝒫(i)|) Σ_j v_ij[k])`. The trained scale of a slice's parameters places it between the two limits, and a layer can contain slices of both kinds.

**Multiplicity.** At every scale the combine is `E_μ[e^v v] / E_μ[e^v]`, where `μ` is the empirical distribution of the partners' `(x_j, pf_ij)`, so the output depends on the partner multiset only through its normalised distribution. In particular, if every partner is repeated `m` times,

```
Σ_j m·e^{v_j} v_j / Σ_j m·e^{v_j}  =  Σ_j e^{v_j} v_j / Σ_j e^{v_j},
```

so `y_i[k]` is unchanged. A slice reports which kinds of partners are present and with what share; the number of partners is not an input. For lists of exactly `K` partners a share determines a count; for all-to-all sets `|𝒫(i)|` varies with the set size, so a layer is not a counting operator in general (counting targets are measured in §5.1).

### 3.5 Gradients

Let `s_i[k] = Σ_j w_ij[k] v_ij[k]` be the combine, so that `y_i = leaky(s_i)`, and let `g_i` be the gradient of the loss with respect to `s_i`. For the pre-activation, `dz_ij = g_i · w_ij (1 + v_ij − s_i) · leaky'(z_ij)`, and
`∇a_i = Σ_j dz_ij`, `∇b_j = Σ_i dz_ij`, `∇E = Σ_ij pf_ij ⊗ dz_ij`, `∇c = Σ_ij dz_ij`. The kernel's backward
pass computes this directly, recomputing `z_ij` instead of storing it. The relation `pf` is treated as data and
receives no gradient. Because every valid partner has positive weight and `leaky'` is non-zero everywhere (taking the subgradient `0.1` at `z_ij = 0`), whenever `g_i ≠ 0` every partner receives a non-zero gradient except where `1 + v_ij = s_i`. Note that `1 + v_ij − s_i` changes sign when `v_ij < s_i − 1`, since increasing a low-scoring partner's value raises its softmax weight and pulls the weighted average `s_i` down.

### 3.6 Relation features

`pf_ij` is built from coordinate differences (`dx = x_i − x_j`, `dy = y_i − y_j`, with `r = max(√(dx² + dy²), 10⁻⁶)` so coincident entities give zero direction rather than `0/0`) and is a configuration choice: `pos` = `(dx, dy)`,
`axis` = `(|dx|, |dy|)`, `dist` = `r`, `dir` = `(dx/r, dy/r)`. Direction and distance are bundled as `polar` because either alone leaves one-slice questions unanswerable: distance alone does not identify the
direction of a partner, and direction alone does not identify its distance. Direction is given as
`(cos, sin)`, so there is no wrap-around and no implied order. Any tensor of
relations can be used instead (a graph's edge attributes, an offset in a sequence). The kernels accept up to eight relation components.

### 3.7 Neighbour lists

For each receiver the list holds up to `K` partners: its `K` nearest valid entities, optionally only
those within a radius. The filter and the combine are unchanged; the list changes the partner set and the cost, and the layer uses distance only if `pf` contains it. Lists may also point at a different
set of entities than the receivers (senders that never receive, such as fixed scenery).
