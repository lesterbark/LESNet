## 1. Introduction

Many learning problems are about a *set of interacting entities* rather than a grid or a sequence: the
players and ball of a game or the particles of a fluid. The
target often depends on *relationships* rather than on one entity alone, such as which other entity is nearest or the mean velocity of a particle's neighbours. We call this the **entity regime**: tens to a few thousand entities, each with a feature vector
and a position, no fixed order, and a number that changes from one example to the next.

Established models address parts of that regime.

* **Convolutional networks** apply a learned filter at every position of a regular grid, within a local window.
* **Graph networks** let entities exchange messages along edges, usually chosen by distance, and aggregate the messages (by a mean, for example) before a node update. The graph is built before the network runs.
* **Transformers** let every entity attend to every other. Attention is a scalar score per pair and head followed by a read of the partner's own value; the relation between two entities (their relative position, say) enters through position encodings or an attention bias, and the cost is quadratic in the number of entities unless the attention is restricted.

In a graph network the pair step is a multilayer perceptron; in a Transformer the relation enters only the score, and the value read from a partner does not depend on the receiver. This paper studies a pair step at the other extreme: a single linear map of receiver, partner and relation, followed by a per-channel soft maximum over the partners.

**LESNet** (Leaky Entity Softmax Network) is a layer of this kind. One layer does three
things for every entity `i` and every slice `k` of its output:

1. a **linear pair filter**: `z_ij = a_i + b_j + pf_ij E + c`, where `a_i` and `b_j` are linear in the two
   entities' features and `pf_ij` is the *relation* between them (relative position, distance,
   direction), supplied as data;
2. a **leaky ReLU** (Maas et al., 2013), giving one value `v_ij` per partner `j`;
3. a **softmax combine**: `y_i = leaky(Σ_j softmax_j(v_ij) · v_ij)`, a soft maximum over partners,
   computed separately for every slice.

The pair step is a single linear map and an activation; there is no query/key/value split, no positional encoding and no
normalisation layer. Each slice learns one linear test on `(x_i, x_j, pf_ij)`
and then weights the partners by how strongly they pass it. The width (the number of slices) is the size parameter of a layer. Because the partners of an entity are a list, the same layer runs all-to-all or
on each entity's `K` nearest neighbours within a radius.

**Contributions.**

* The layer and its analysis (§3): each slice acts as a one-dimensional attention head whose score and value are the same pair-dependent quantity; within a slice the receiver's features do not change the partner ranking except through the relation; the output depends on the partners only through their normalised distribution; and a slice has closed-form sharp (max-plus) and diffuse (rectified mean) limits.
* A fused CUDA implementation (§4) for both partner layouts. It never materialises the pair tensor, is verified against a PyTorch oracle in values and every gradient, and comes with an exact grid neighbour search.
* A controlled study on seven synthetic relational tasks with exact targets (§5): nine models at matched parameter count and three seeds, ablations of the relation vector and the nonlinearity, locality, capacity, and sets larger than those seen in training.
* Real-data benchmarks (§6): top-quark jet tagging and MNIST as a point set (variable-size sets with geometric relations), citation-graph node classification (a given graph, no geometry), CIFAR-10 with pixels as entities against a convolutional network, and character-level language modelling against a small GPT.
* A learned particle-fluid simulator compared with the SPH solver that generated its data and with a graph-network simulator (§7), including a calm-down test and an analysis by particle speed.
* Cost measurements for the layer and comparison models, reported as ratios on one GPU (§8).

Across these benchmarks, the results map out both the reach and the limits of a single-linear pair filter. On geometric sets, LESNet trails a message-passing graph network and GENConv at matched parameters and steps (most visibly on graded local averages and counts), while holding its own against a relation-biased Transformer on several tasks and matching a graph-network fluid simulator in rollout error. On graphs without geometry, regular pixel grids and character sequences, specialised baselines stay clearly ahead.
