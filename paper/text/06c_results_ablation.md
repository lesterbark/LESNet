### 5.3 Ablations: the relation vector and the nonlinearity

These ablations use the PyTorch reference implementation of the layer (same mathematics, no fused kernel) on sets of 8-16 entities, with 50k parameters, 2000 steps, two or three seeds and a learning rate chosen per variant from a short grid. Absolute scores are therefore lower than in §5.1 (`nn_type` 0.926 against 0.979), and the unmodified layer is a separate run in each table. {{ref:ablate}} uses two seeds (t multiplier 12.7) and {{ref:noact}} three.

**The relation vector.** By §3.3, without `pf_ij` every receiver ranks its partners identically within a slice. The ablation removes the relation from all three layers (the filter keeps `a_i + b_j`, and this variant receives the absolute coordinates as input features instead) and compares the four-component relation (relative position and axis) with one that adds distance and direction.

{{table:ablate|Score with no relation vector, with relative position and axis, and with position, axis, distance and direction (mean ± 95% interval over two seeds, t distribution with 1 degree of freedom; learning rate 1e-2 for all variants; the parameter count is the mean over the three tasks).}}

Removing the relation lowers `closest_team` from 0.957 to 0.616 (clearly separated); on `nn_type` (0.926 vs 0.807) and `count_r` (0.800 vs 0.783) the wide two-seed intervals overlap. Because later layers can still combine a receiver's own coordinates with partner summaries from earlier layers, these scores measure how much of each task the full stack can recover without `pf`, not the one-slice property itself. Adding distance and direction raises `nn_type` from 0.926 to 0.958 (separated); on `count_r` (0.865) and `closest_team` (0.983) the gains lie within the two-seed intervals.

**The nonlinearity.** The layer applies a leaky ReLU to the pair values and to the combined output. Replacing both with the identity (learning rate chosen from 3e-3 and 1e-2; 1e-2 has the higher mean for both variants on every task, with overlapping intervals on `count_r` and `closest_team`):

{{table:noact|Score with the leaky ReLU and with an identity in its place (mean ± 95% interval over three seeds, t distribution with 2 degrees of freedom).}}

The activation matters on the two tasks whose target is a graded function of the neighbourhood: `count_r` drops from 0.797 to 0.713 and `local_vel_avg` from 0.792 to 0.733. On the two selection tasks (`nn_type`, `closest_team`) removing the leaky ReLU makes no resolved difference (0.926 vs 0.926; 0.948 vs 0.945).

### 5.4 Locality, capacity and learning rate

**Partners per entity.** Varying the number `K` of nearest partners (two seeds; learning rate 1e-2 rather than the 3e-2 of §5.1, so scores are below {{ref:main}}):

{{fig:ksweep.png|LESNet score against the number of nearest partners K ("all": all-to-all), mean over two seeds with 95% intervals (t, 1 degree of freedom); learning rate 1e-2.}}

`K = 1` solves `nn_type` exactly (the nearest entity is the only partner) and is of little use on the other two tasks. For the radius-based tasks (about 3 to 4.5 entities inside the radius) the
mean score peaks at `K` = 4 to 8 (`count_r` 0.918 at `K` = 4 and 0.923 at `K` = 8, `local_vel_avg` 0.901 at `K` = 4) and dips toward all-to-all (0.891 and 0.866). On `local_vel_avg`, `K` = 4 separates from all-to-all while `K` = 4, 8 and 16 overlap one another; on `count_r`, the two-seed intervals for `K` = 4, 8, 16 and all-to-all all overlap.

**Width and depth.** Across single-seed runs at learning rate 3e-2, `nn_type` stays flat at 0.97–0.98 for every width and depth tried, and `closest_team` ranges between 0.96 and 0.996. On `local_vel_avg` (the one task with headroom), deeper stacks score higher (width 128: 0.831, 0.870, 0.889 at depth 1, 2, 4) while width makes little difference (depth 1: 0.821 to 0.834 from 32 to 256 slices).

**Parameter scale.** From about 30k to 600k parameters (two seeds), `nn_type` is 0.978 to 0.980 for LESNet, 0.941 to 0.961 for the Transformer with the relation and 0.991 to 0.995 for the graph network.

**Learning rate.** The mean score over the three tuning tasks (one seed) against the learning rate is in {{ref:lr_tune}} (Appendix A).

LESNet's single-seed score rises up to 3e-2 (0.870 at 1e-3, 0.947 at 3e-2) and drops at 1e-1 (0.777); its chosen rate is three to ten times that of the baselines (GENConv and DeepSets peak at 1e-2, the graph network at 3e-3, and the Transformer degrades above 3e-3). The rate is chosen with one seed, and the chosen rate leads the next best by 0.005 for LESNet and 0.003 for the graph network.

To see whether the low-rate deficit comes from the soft maximum collapsing onto one partner, the weights were measured at the end of training of the PyTorch reference implementation (three layers of width 96, 1500 steps, all-to-all) at four learning rates (one seed). The normalised entropy of the weights is the entropy of `w` over the valid partners divided by the logarithm of their number (1 for uniform weights).

The weights stay close to uniform on average at every rate ({{ref:softmax_probe}}, Appendix A: normalised entropy 0.89 to 0.99, mean largest weight 0.09 to 0.21), and in these single runs a higher learning rate goes with more uniform weights and a higher score. On average, then, the low-rate deficit is not a collapse of the soft maximum onto one partner, and the trained layers lie nearer the diffuse limit of §3.4 than the sharp one.
