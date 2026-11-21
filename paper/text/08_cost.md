## 8. Cost on the test hardware

All costs were measured on one NVIDIA GTX 1660 Ti (mobile; Turing, 6 GB, no tensor cores) in fp32. The LESNet kernels are fused CUDA code and the baselines are PyTorch implementations. The kernels were not built with tensor cores in mind, so other hardware may give different ratios.

Costs are given as ratios to LESNet, except in §8.1, which gives absolute times and memory for this card. Step costs and memory were measured on an otherwise idle card as the minimum over several timing windows of a full training step (forward, backward and optimiser) at the batch size used in training, except in §8.1, which times one forward and backward pass of the layer stack.

### 8.1 Cost against set size

Here we fix width to 128 and depth to 3, sizing the batch so that every set size `N` processes the same 8192 entities. `LES all-to-all (PyTorch)` is a direct implementation of the same mathematics; `GNN gather` and `LES list` share the same grid neighbour search with `K = 16`. At this width LESNet has 70k parameters, the Transformers 398k and the graph networks 250k; parameter-matched costs for the benchmark models appear in §8.2.

{{fig:speed.png|Time (left) and peak memory (right) of one forward and backward pass of a three-layer, width-128 stack against set size N, with the batch sized so that every point processes 8192 entities; both axes logarithmic. Dashed: direct PyTorch implementations; OOM: out of memory at the next size; the Transformer with the relation was measured up to N = 1024. With neighbour lists (K = 16) the time of LESNet stays between 7.4 and 8.5 ms.}}

* The fused all-to-all kernel takes 5.0 ms at 16 entities against 44.5 ms for the direct PyTorch implementation (9 times faster), and 11.2 ms against 169.8 ms at 64 entities (15 times). The direct implementation and the all-to-all graph network in PyTorch are out of memory from 128 entities; the fused kernel runs at 4096 entities (781 ms).
* At 16 and 32 entities the all-to-all kernel is the fastest model measured (5.0 and 7.2 ms); from 64 entities the list kernel is, at 7.4 to 8.5 ms and 109 MB for every N from 16 to 4096. The Transformer takes 17.0 ms at 16 entities and 237 ms at 4096, 27.8 times the list kernel.
* The graph network on the same neighbour lists takes 38 to 43 ms and 796 to 844 MB, which is 5.0 to 5.7 times the time of the list kernel from 32 entities on and 7.3 to 7.7 times its memory.

**Arithmetic per pair.** Counting a multiply-add as two floating-point operations, a LESNet layer costs `4NCW + NKW(2P + 20)`, where about 20 operations per pair and slice cover the adds, the activation and the online softmax, which is about 34 operations per (pair, slice) for `P = 7`. A dense attention block costs `16NW² + 4N²W`, which is 4 operations per (pair, slice) plus the projections (with a feed-forward layer of width 2W). The message function of the graph network adds `2W²` per pair for its second linear layer.

### 8.2 Cost of the benchmark models

Step cost and peak memory of the models of §6 are in {{ref:cost}}.

{{table:cost|Step cost and peak memory of the benchmark models relative to LESNet, at the batch size of each benchmark.}}

On jets the graph network and GENConv are the most expensive models ({{ref:cost}}). GENConv recomputes its pair stage in the backward pass (checkpointing), which lowers its memory and raises its time; without it the model does not fit the 6 GB card at the batch size used. At the widths used, the message function of the graph network (a second linear map of width 92 on every pair) takes 18 500 operations per pair and layer against 5 300 for LESNet at width 156, a ratio of 3.5, and the graph network materialises a message tensor of shape (batch, entities, neighbours, width) in every layer, which the fused LESNet kernel does not.

### 8.3 Accuracy at equal cost on jets

The jet learning curves plotted against relative compute ({{ref:toptag.png}}, right) place LESNet runs of 1700, 3200 and 6400 steps at 1.02, 1.04 and 1.04 times the cost of 1000 Transformer steps, 1000 graph-network steps and 2000 graph-network steps. At the cost of 1000 Transformer steps LESNet reaches AUC 0.9808 and background rejection 662 at 30% signal efficiency (Transformer: 0.9802, 645). At the cost of 1000 graph-network steps it reaches 0.9824 and 811 (graph network: 0.9816, 701), and at the cost of 2000 graph-network steps 0.9834 and 918 (graph network: 0.9830, 885). One GENConv step costs 5.7 LESNet steps, so the LESNet run of 6400 steps costs 1.11 times 1000 GENConv steps and reaches 0.9834 and 918 (GENConv: 0.9810 and 644), and the run of 11 800 steps costs 1.03 times 2000 GENConv steps and reaches 0.9840 ± 0.0001 and 981 ± 105 over three seeds (GENConv: 0.9827 ± 0.0002 and 804 ± 21; both intervals separate). The GENConv cost includes the recomputation. The runs of 1700, 3200 and 6400 steps have one seed each and are reported without an interval; the baselines have three seeds.

### 8.4 The learned fluid simulator against the native solver

Cost is the time to advance one simulated second at a given number of particles, relative to LESNet with 8.3 ms steps at 500 particles.

{{fig:fluid_scaling.png|Cost of advancing the fluid by one simulated second against particle count, relative to the 105k LESNet at 500 particles (both axes logarithmic). The learned simulators take 8.3 ms steps and the native solver 0.13 ms steps. Shaded: more particles than the unit box holds at the solver's spacing, so only the learned models run.}}

A learned step of 8.3 ms replaces 64 solver steps, so the ratios compare simulators with different time steps, not the cost of one step.

At 500 to 2000 particles LESNet with 8.3 ms steps advances the fluid 1.9 to 2.0 times faster than the cell-list solver and the graph network 1.0 to 1.4 times faster (cell-list cost divided by learned cost); against the all-pairs solver the ratios are 2.9 to 6.7 for LESNet. Per simulated second at equal step length LESNet costs 0.68, 0.69 and 0.53 times as much as the graph network at 500, 1000 and 2000 particles, and at these particle counts the 407k-parameter LESNet costs the same as the 105k one within 6%.

The cost of LESNet with 105k parameters is flat from 500 to 4000 particles. Beyond about 2400 particles the unit box cannot hold the particles at the solver's spacing, so only learned simulators are compared there. From 4000 to 8000 particles the cost of LESNet rises by a factor of 2.1, in proportion to the particle count, and from 4000 to 16 000 particles by a factor of 6.6 for 4 times the particles.
