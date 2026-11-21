## 5. Experiments on synthetic entity tasks

### 5.0 Protocol

**Worlds.** Each example is a set of `n` entities (`n` drawn uniformly from 8-24, padded to 24 with a
mask) in a square whose side grows as `√n`, so the density stays at about 16 entities per unit area for
any `n`. An entity has a one-hot type (A, B or C; exactly one entity, the "ball", is type A), a 2-d
velocity feature and a position. Local tasks therefore mean the same thing at `n = 16` and `n = 96`, which
is what makes the out-of-distribution-size test (train on 8-24, test on exactly 48 and 96) unambiguous.

**Tasks** (targets are exact functions of the world; data are generated on the GPU, fresh every step):

| task | target | kind | probes |
|---|---|---|---|
| `nn_type` | type of my nearest other entity | per-entity, 3 classes | local, relational, argmin-like |
| `count_r` | number of entities within 0.25 of me | per-entity regression | **counting** |
| `rank_ball` | number of entities closer to the ball than I am | per-entity regression | rank (a count of comparisons) |
| `closest_team` | team (B or C) of the entity nearest the ball | set-level, 2 classes | global argmin |
| `exist_pair` | is any B within 0.07 of any C | set-level, 2 classes | global existence (max-like) |
| `local_vel_avg` | mean velocity of entities within 0.3 of me | per-entity regression (2-d) | local smoothing, as in SPH |
| `nbody_acc` | softened gravity from all other entities | per-entity regression (2-d) | **superposition**: a sum over all pairs |

Classification is scored by accuracy, regression by R² over valid entities (higher is better for both). Regression targets `count_r`, `rank_ball` and `nbody_acc` are linearly scaled by fixed constants (1/4, 1/6 and 1/4) for training stability, which does not change R²; for `local_vel_avg`, entities with no partner within 0.3 have target zero.
For `nbody_acc` the acceleration of entity `i` is `Σ_j m_j (r_j − r_i) / (|r_j − r_i|² + ε²)^{3/2}` with softening `ε = 0.2`, the ball having mass 4 and every other entity mass 1. At the density of the worlds (16 entities per unit area, median nearest-neighbour distance 0.13) an entity has on average 1.6 partners within `ε`. The forces of the 1, 2, 3, 5 and 8 nearest partners alone explain R² = 0.16, 0.33, 0.47, 0.68 and 0.85 of the target, and a linear function of the entity's position and the centroid of the set explains 0.68.

**Models.** All baselines receive `[x, coords]` as node input, whereas LESNet receives `x` alone so that positions enter only through the relation; every model that takes a relation receives the same seven-component vector `pf = (pos, axis, polar)`:

* **LESNet**: the layer on the CUDA kernels, all-to-all; **LESNet, 8 nearest**: the same with 8 nearest partners.
* **MLP-flat**: the set flattened into one vector (fixed `n`, order-dependent), the no-structure floor.
* **DeepSets**: per-entity MLP with a pooled context, no pair interactions.
* **Transformer, no relation**: pre-LN encoder, 4 heads; **Transformer + relation**: the same with a learned attention bias
  computed from `pf`.
* **Graph network (GNN)**: message passing on the full graph, `message = act(Linear(act(A h_i + B h_j + E pf_ij)))` with a
  residual node update (`h + act(U([h, agg]))`) and mean aggregation. **Graph network, 8 nearest** restricts it to 8 neighbours.
* **GENConv**: softmax aggregation message passing in the style of DeeperGCN (Li et al., 2020), the closest prior aggregator to LESNet: `message = relu(B h_j + E pf_ij) + 10⁻⁷`, a per-channel softmax weighting of the messages with one learned inverse temperature per layer, and a two-layer residual node update.

**Training and evaluation.** Every model has about 150k parameters (width fitted per family by search), depth 3, the
same head, and the same optimiser: AdamW (Loshchilov and Hutter, 2019; weight decay 0), cosine schedule with 100-step linear warmup, gradient-norm clipping at 1.0, batch 128 and 3000 steps, seeing the same data stream for a
given seed. The learning rate is tuned once per family on three tasks (`nn_type`, `closest_team`, `local_vel_avg`; one seed; grids and results in {{ref:lr_tune}}, Appendix A) and fixed for all tasks; variants inherit their family's value. Each cell reports the mean over 3 seeds with a 95% confidence interval from the
t distribution (2 degrees of freedom). We call two models separated only when their marginal 95% intervals do not overlap (with no correction for multiple comparisons); because the t multiplier at 2 degrees of freedom is 4.30, this rule is conservative, and overlapping intervals mean a difference is not resolved at three seeds rather than proving equality. Evaluation sets are fixed (2048 examples; 1024 for the larger-`n` tests); every family's learning rate is chosen with seed 0 on the same 2048-example sets.
