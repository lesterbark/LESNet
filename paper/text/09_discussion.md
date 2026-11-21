## 9. Discussion

### 9.1 What the evidence shows

On variable-size sets with geometric relations, the graph network and GENConv outperform LESNet at equal parameters and steps, both on top-quark jets (where the AUC gap to the graph network narrows from 0.0019 at 2000 steps to 0.0010 at 11 800) and on four and five of the seven synthetic tasks (§5.1, §6.1). The gaps are not uniform: they are largest on the graded local targets `count_r` and `local_vel_avg` (0.073 and 0.100 R² behind the graph network) and vanish within error bars on the two set-level selection tasks, `closest_team` and `exist_pair`. Against a Transformer given the same relation vector, the outcome depends on the task: LESNet wins on three synthetic tasks and loses on one, while on jets it lags at 1000 steps and matches within error bars at 2000 and 11 800 steps.

Restricting each entity to its 8 nearest partners lifts LESNet on `count_r`, `local_vel_avg` and `exist_pair`, hurts it on the two tasks requiring distant partners, and holds its score at 96 entities within 0.011 of the training range while every all-to-all model drops to mean R² ≤ 0.64 on the regression tasks (§5.1, §5.2). Removing the relation vector drops `closest_team` from 0.957 to 0.616 (§5.3).

As a learned fluid simulator, LESNet matches a parameter-matched graph-network simulator in 0.5 s rollout error across four training seeds, even though the graph network achieves lower one-step validation loss in every seed. The clearest qualitative difference lies in rollout energy: LESNet's median kinetic-energy ratio at 0.5 s (0.91–1.04 vs 0.79–0.90) and mid-speed particle tracking (0.3–1 m/s) lie closer to the native solver in every seed pair, though the four-seed intervals overlap (§7.5).

### 9.2 Where the layer is behind

LESNet trails domain-specific baselines on the three benchmarks outside its target regime (a non-geometric graph, a pixel grid and a character sequence; §6.3–§6.5), as well as the graph network and GENConv on jets and most synthetic tasks (§5.1, §6.1).

Three properties from §3 characterise what a single slice can compute. Receiver and partner features enter additively inside the pair filter and never multiply, unlike a convolution (which applies a separate weight matrix per offset) or dot-product attention (which forms `q_i·k_j`). Within a slice, partner ranking depends on the receiver only through the relation `pf`. And the softmax combine sees only the normalised partner distribution, so multiplicity is invisible. GENConv shares all three properties inside its aggregation: its message `relu(B h_j + E pf_ij)` is additive with no receiver term, and its softmax aggregator (like the graph network's mean) ignores multiplicity. Those properties alone therefore do not account for the gap between LESNet and GENConv; the remaining architectural differences listed in §9.3 (receiver term, ReLU vs leaky ReLU, learned temperature, node-update MLP and residual path) were not ablated individually.

### 9.3 Limitations

* **Budgets and seeds.** The main jet comparison uses 0.42 and 0.85 passes over the data (5 in the longer runs, with three seeds except one for GENConv, and 10 with one seed) against 20 for published baselines. The MNIST point-set models had not fully plateaued (accuracy still rose over the last two epochs, separating from zero over three seeds for DeepSets), and LESNet on CIFAR-10 was still improving (68.5% in a single 200-epoch run vs 63.0% at 40). Several secondary sweeps use one or two seeds (§10).
* **Learning-rate grids.** Learning rates were tuned per family on short, single-seed grids.
* **GENConv coverage.** GENConv was evaluated on the synthetic tasks and top-quark jets, not on MNIST point sets, citation graphs or the fluid simulator.
* **Relation width and gradients.** The CUDA kernels cap the relation vector at eight components: the full 2-D set of §3.6 (`pos, axis, dist, dir`) needs seven, whereas its 3-D counterpart needs ten. Because `pf` receives no gradient, the relation cannot be learned end-to-end.
* **Bundled baseline differences.** The graph network differs from LESNet in four ways at once (a second linear layer inside the message, mean aggregation, a node-update network and a residual path), and GENConv in five (no receiver term, ReLU, a learned temperature, a node-update network and a residual path); no ablation isolates each piece. On jets and MNIST point sets, the Transformer attends over all pairs while the other pair models use 16 nearest neighbours.
* **Fluid scope.** The fluid benchmark is two-dimensional, single-phase and confined to a unit box, with 0.5 s rollouts (3 s for calm-down) against a custom cell-list SPH reference solver.

### 9.4 Future work

* **An additional aggregation path.** The forward pass already accumulates `Σ e^v v` and `Σ e^v` per slice. A running sum `Σ v` of the pair values could be accumulated in the same pass. A layer that emits the softmax combine together with a sum would have a counting path, and together with a mean an averaging path. Its effect could be measured on `count_r`, the citation graphs and the pixel grid.
* **Pair-step variants.** Other activation functions, other combine functions and further layers inside the pair step are untested.
* **Larger relation vectors and learned relations.** Raising the relation limit of the kernels to cover three-dimensional geometry, and passing gradients to the relation so that it can be learned.
* **Other hardware.** Implementations of the pair step for tensor cores and reduced precision, and cost measurements on hardware generations other than the one used here, with layer widths chosen to match the architecture of each.
* **Training to convergence.** Full-length training on the jet benchmark with the published protocol and parameter counts, and the same for the other benchmarks.
* **Fluids.** Three dimensions, several fluids and boundaries, training on rollouts rather than single steps, comparison against a position-based-fluids solver, and a comparison with all simulators at the same time step.
* **Theory.** The sharp limit of §3.4 describes a max-plus operator on the partner set. The function class of a stack of layers, and the effect of depth on it, are open.

### 9.5 Conclusion

This paper tested how far a minimal pair layer (one linear map of receiver, partner and relation, a leaky ReLU, and a per-slice soft maximum) can go on sets of interacting entities. Each slice interpolates between a max-plus operator and a leaky-rectified mean, ranks partners without multiplicative receiver-partner terms, and is invariant to partner multiplicity. Fused online-softmax CUDA kernels eliminate the `O(N²W)` pair tensor, keeping step time and peak memory well below PyTorch message-passing baselines on a Turing GPU; on jet tagging this gives competitive accuracy at equal compute, and on 2-D fluid simulation LESNet matches a graph-network simulator in rollout error. At matched steps and parameters, however, omitting a hidden message layer, residual node update and counting path leaves clear gaps on graded synthetic targets, jet tagging and non-geometric domains.
