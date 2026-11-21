### 7.5 Results

We evaluate 0.5-second rollouts from the native state across the 12 held-out trajectories (186–976 particles), using 6 start frames per trajectory (72 rollouts per model) at `M = 4` (8.3 ms per learned step).

{{fig:fluid_strips_v2.png|One held-out test scene: native SPH (top) and the 105k LESNet and 107k graph-network simulators started from the same state under the same controls (training seed 0). Colour: particle speed.}}

**Accuracy.** After 0.5 s, both 105k LESNet and the 107k graph network achieve Chamfer distances of 0.52 to 0.55 particle spacings and position errors of 3.4 to 3.8 particle spacings across four training seeds each.

{{table:fluid_cluster_diff|Difference LESNet minus graph network of the same training seed, mean ± 95% interval over the 12 trajectories.}}

The intervals in {{ref:fluid_cluster_diff}} capture variation over trajectories rather than training seeds. Averaged over the four training seeds, none of the differences separates from zero: Chamfer distance 0.008 ± 0.024 spacings, position error 0.20 ± 0.33 spacings and density error −2.2 ± 3.6 percentage points. Within individual seeds ({{ref:fluid_cluster_diff}}, intervals over trajectories), Chamfer distance overlaps in all four seeds; LESNet has higher position error in one seed and overlaps in three; and its density error is lower in three seeds and higher in one. At 407k parameters (one seed), LESNet reaches the lowest mean Chamfer distance (0.51) and density error (5.1%) in {{ref:fluid_cluster}}, though its trajectory-level intervals overlap every 105k LESNet seed.

{{table:fluid_seeds|Four training seeds of each 105k model: median kinetic-energy ratio at 0.5 s, one-step validation loss, quiet-scene calm-down speed and median calm-down kinetic-energy ratio at 1 s (started at t = 2.3 s).}}

The graph network reaches a lower one-step validation loss in every seed. Conversely, LESNet's median kinetic-energy ratio at 0.5 s across the 12 test trajectories is closer to 1 in all four seed pairs (0.91–1.04 vs 0.79–0.90), though the four-seed intervals overlap for both the ratio (0.97 ± 0.10 vs 0.83 ± 0.08) and its distance from 1 (0.05 ± 0.06 vs 0.17 ± 0.08).

**Fast and slow particles.** Binning particles by native speed on the active scenes after 0.1 s and 0.25 s shows where the two simulators diverge:

{{fig:fluid_speed_bins.png|(a, b) Mean speed of the learned rollout over the native mean speed, for particles binned by their native speed, after 0.1 s (a) and 0.25 s (b) on the active test scenes. (c) Kinetic energy of the learned rollout over the native one. Lines: mean over the four training seeds; bars and bands: range over seeds; dotted: native (1).}}

For particles moving at 0.3 to 1 m/s, the learned-to-native speed ratio is 0.87 to 0.97 for LESNet against 0.79 to 0.91 for the graph network across the four seeds and two horizons; LESNet is higher in all eight seed-horizon pairs, though the four-seed intervals overlap at both horizons. Between 1 and 2 m/s the two ranges overlap (0.93–1.04 vs 0.94–1.03). On the active scenes ({{ref:fluid_speed_bins.png}}c), LESNet's seed-mean kinetic energy stays above native for most of the rollout while the graph network stays below (with overlapping four-seed intervals at 0.35 s and 0.5 s): at 0.35 s the ratios span 0.94–1.10 vs 0.84–0.93, and at 0.5 s 1.03–1.19 vs 0.96–1.01. For the slowest particles (below 0.3 m/s), the graph network tracks native speed (0.98–1.02 at 0.1 s; 1.02–1.10 at 0.25 s) while LESNet overshoots (1.03–1.08 and 1.17–1.36). Both models underestimate the fastest particles (above 3 m/s, 4 and 213 particles in the two bins, not plotted; ratios 0.82–0.94).

**Calm-down test.** Rolling out a nearly settled fluid under straight-down gravity gives the results in {{ref:fluid_rest}} and {{ref:fluid_seeds}}:

{{table:fluid_rest|Calm-down test on the eight settle and quiet test trajectories (training seed 0, single values): mean particle speed over the final 0.75 s of a 3 s rollout from the end of each trajectory, change in mean fluid height over those 3 s, and Chamfer distance after a 1 s rollout started at t = 2.3 s.}}

Starting from the end of the quiet scenes, where the native fluid moves at 0.065 m/s, 105k LESNet ends at 0.050 to 0.104 m/s across its four seeds and the graph network at 0.048 to 0.150 m/s ({{ref:fluid_seeds}}), with overlapping ranges; 407k LESNet ends at 0.058 m/s. Across all eight settle and quiet scenes, the seed-0 models finish at 0.111 m/s for LESNet (native starting speed 0.109 m/s) and 0.074 m/s for the graph network, with 1 s median kinetic-energy ratios (from `t = 2.3 s`) of 2.29 and 0.86; across all four seeds, final speeds span 0.057–0.111 and 0.057–0.190 m/s with overlapping intervals. Mean fluid height shifts by −4.3 mm (105k LESNet), −3.0 mm (407k LESNet) and −9.7 mm (graph network) for seed 0 ({{ref:fluid_rest}}), and by −2.0 to −4.3 mm vs −0.3 to −13.5 mm across the four seeds (overlapping intervals).

**More particles than any training scene.** We also test on four held-out scenes containing 1620–2162 particles, roughly twice the largest training scenes (at most about 1000 particles).

Chamfer distances (0.50–0.51) remain comparable to the in-distribution test scenes. For the seed-0 models, density error is 5.6% for 105k LESNet, 26.6% for the graph network (4.7 times higher) and 5.4% for 407k LESNet, while all three overshoot native kinetic energy (median ratios 2.90, 2.09 and 2.28).


