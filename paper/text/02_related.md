## 2. Related work

**Interaction and relation networks.** Interaction Networks (Battaglia et al., 2016) and Relation
Networks (Santoro et al., 2017) apply a shared function to every pair of objects and *sum* the results;
the pair function is a multilayer perceptron. LESNet shares the all-pairs, shared-filter structure but
(i) replaces the pair MLP with one linear map and an activation, (ii) replaces the sum by a per-slice
soft maximum, and (iii) feeds the relation `pf_ij` in explicitly.

**Message passing on graphs.** Message-passing neural networks (Gilmer et al., 2017) on a complete
graph with edge features contain the layer as a special case. Softmax aggregation of messages appears
in DeeperGCN (Li et al., 2020) and attention-weighted aggregation in graph attention networks
(Veličković et al., 2018; Brody et al., 2022); principal neighbourhood aggregation (Corso et al., 2020)
combines several aggregators. The softmax aggregator of DeeperGCN weights each channel of a message by a softmax of that same channel, so its message is also its own score. LESNet forms the message from receiver, partner and relation with one linear map and a leaky ReLU, and has no node-update network.

**Set functions.** Deep Sets (Zaheer et al., 2017) and PointNet (Qi et al., 2017) process entities
independently and pool; they have no pairwise term. The Set Transformer (Lee et al., 2019) adds attention.
We use a Deep Sets variant with a pooled context as a baseline.

**Attention.** In a Transformer (Vaswani et al., 2017) the weight of partner `j` for entity `i` is a
softmax over scalar scores `q_i·k_j`, and the output reads the partner's value `v_j`, which does not depend
on `i`. §3.3 shows each LESNet slice is an attention head whose value is pair-dependent and doubles as
its score. Fused attention kernels that never store the score matrix (Dao et al., 2022) are the closest
engineering precedent for §4; the online-softmax recurrence we use is that of Milakov and Gimelshein (2018).

Attention that does not rest on the query-key product has been studied: Synthesizer
(Tay et al., 2021) generates the attention weights from each token alone, or learns them as free parameters. In LESNet the weights depend on the pair through an additive filter and the value is pair-dependent.

**Point sets and jets.** Dynamic Graph CNN (Wang et al., 2019) builds a nearest-neighbour graph and applies an
edge function with a hidden layer to each neighbour pair; ParticleNet (Qu and Gouskos, 2020) applies it to particle
jets and is a standard strong model on the top-tagging benchmark used in §6.1, with Energy Flow Networks / Particle Flow
Networks (Komiske et al., 2019), a Deep Sets variant, as the usual lightweight baseline. Our graph-network baseline is
in this family: nearest neighbours, a pair function with a hidden layer, mean aggregation.

**Real-data baselines.** Convolutional networks are the model for regular grids (LeCun et al., 1998; CIFAR-10: Krizhevsky, 2009); graph convolutional and graph attention networks (Kipf and Welling, 2017; Veličković et al., 2018) are the
standard models for node classification on citation graphs (the Planetoid split: Yang et al., 2016); small autoregressive Transformers (Radford et al., 2019) are the standard models for character-level language modelling.

**Learned physics.** Graph-network simulators (Sanchez-Gonzalez et al., 2020), DPI-Net (Li et al., 2019)
and continuous-convolution fluid networks (Ummenhofer et al., 2020) learn particle dynamics from
simulation. SchNet (Schütt et al., 2017) and equivariant graph networks (Satorras et al., 2021) pass
pairwise distances or relative coordinates into the pair step, as `pf_ij` does here. Our fluid
experiment follows the graph-network-simulator recipe (noise-injected one-step training, autoregressive
rollout) and uses smoothed-particle hydrodynamics (Monaghan, 1992) as the reference solver.
