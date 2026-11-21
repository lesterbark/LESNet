## 6. Real-data benchmarks

This section applies the layer to real data: two benchmarks in the regime it is designed for (variable-size sets with geometric relations: top-quark jets and MNIST as a point set) and three outside it (a given graph without geometry, a regular grid, a sequence). Unless stated otherwise, models are matched in parameter count, the learning rate is chosen per model on held-out data, all models see the same data stream, and results are over several seeds. The right-hand panel of each learning-curve figure plots relative compute: optimiser steps times the step-cost ratio to LESNet measured in §8.

### 6.1 Top-quark jet tagging

The **Top Quark Tagging Reference Dataset** (Kasieczka et al., 2019; 1.21M training, 403k validation and 404k test jets, half top-quark and half light-quark/gluon jets) gives each jet as an unordered set of up
to 200 particles (we keep the 100 leading; 49 on average), each with a four-momentum. The task is binary classification. The relation between two particles is geometric (their distance in the
pseudorapidity-azimuth plane), there is no grid and the number of entities varies: LESNet's regime. The standard models are nearest-neighbour graph networks (ParticleNet; Qu and Gouskos, 2020) and deep sets (the Particle Flow Network, PFN; Komiske et al., 2019).

**Inputs.** Following ParticleNet, each constituent has the node features (log pT, log E, log pT/pT_jet, log E/E_jet, ΔR to the jet axis), standardised, and the coordinates (Δη, Δφ) relative to the jet axis, scaled by 10 so that
typical neighbours are about one unit apart. LESNet, the graph network and GENConv receive coordinates only through the relation; the Transformer and DeepSets also receive them as node features (the same split is used in §6.2). Pair features are the coordinate offset, its axis components and its polar form (`pos,axis,polar`).

**Models.** Every model has about 150k parameters (width fitted per family), depth 3 and the same readout head (mean and max pooling over the jet, one hidden layer, one logit). LESNet, the graph network and GENConv (§5.0, using `clamp_min(1e-7)` on the message and gradient checkpointing on the pair stage) operate on each particle's 16 nearest neighbours in (Δη, Δφ); the Transformer applies a learned attention bias from `pf` over all pairs; DeepSets uses a pooled context without pair interactions.

**Protocol.** One learning rate per family is chosen by validation AUC after 400 steps from {1e-3, 3e-3, 1e-2, 3e-2} (starting from {3e-3, 1e-2, 3e-2} for LESNet and DeepSets, where 1e-2 was interior); 1e-2 is selected for every family except the Transformer (3e-3). Each model is then trained for 1000 and 2000 steps at batch size 512 (0.42 and 0.85 passes over the training jets) with
AdamW (weight decay 1e-4), a cosine schedule and the same data stream across families. Results are reported on the full test set: accuracy, AUC and background rejection 1/ε_B at signal efficiencies of 50% and 30%. The main training budget is 24 to 47 times smaller than that of the published references (ParticleNet: 20 epochs, 366k parameters), whose numbers are quoted from Qu and Gouskos (2020) and Komiske et al. (2019) as external reference points.
