## 7. Application: a learned fluid simulator

A particle fluid has many entities, local interactions and a given relation (relative position). We train LESNet to advance a two-dimensional fluid of
about 150 to 1000 particles and compare it with the solver that generated its data, with a graph-network simulator (Sanchez-Gonzalez et al., 2020) as the learned baseline. Cost is compared in §8.4.

### 7.1 The native solver

The reference is weakly compressible smoothed-particle hydrodynamics (Monaghan, 1992), written as CUDA kernels. Density is the kernel-weighted
sum over neighbours (cubic spline, support 2h with h = 1.3 particle spacings), pressure follows the Tait equation of state `p = max(0, B((ρ/ρ0)^7 − 1))` with
numerical sound speed 40 m/s (about ten times the fastest flow), and artificial viscosity (Monaghan form, α = 0.08) damps acoustic noise. Walls are three layers of
static particles whose pressure mirrors the fluid's. Time stepping is symplectic Euler at the CFL limit, `dt = 0.13 ms`; the unit box, gravity 9.81 m/s² and a
particle spacing of 2 cm fix the scale. There are two implementations of the pair loop, which agree to float precision (positions to 5·10⁻⁶ m after 400 solver steps):
every fluid particle visiting all particles, and a cell list (a uniform grid with cells of side 2h plus a 3 cm skin, rebuilt every 8 solver steps, walls
pre-sorted by cell, threads following the sorted order). Cost comparisons (§8.4) are against the cheaper of the two at each particle count.

### 7.2 Data and controls

Each trajectory is a random scene (one to three blocks of fluid of random size and position, some falling) simulated for 3.5 s under two *controls*: **gravity**,
which every 0.2-0.4 s may rotate by up to ±1 rad, and a **finger**, a radial force field of radius 0.10-0.20 m that pushes or pulls for 0.1-0.35 s at a random place.
Both are inputs of the learned simulator (a per-particle external acceleration). Frames are stored every 16
solver steps (2.08 ms). The data cover three kinds of trajectory, so that the models see the fluid moving and at rest: **active** (controls throughout), **settle**
(controls for the first 35% of the time, then straight gravity, so the fluid sloshes and calms) and **quiet** (no finger and straight gravity throughout, including
flat pools that take a drop). The mean particle speed at the end is 0.62, 0.23 and 0.07 m/s for the three kinds; the native fluid's residual speed at rest is
about 0.065 m/s. There are 160 training, 8 validation and 12 test trajectories (4 of each kind), from fixed seeds. During training each sample is reflected left-right with
probability ½ (positions, velocities, gravity and the finger's position), a symmetry of the physics.

### 7.3 The learned simulators

A learned step maps the state at frame `t` to the state `M = 4` frames later (`Δt` = 8.3 ms). Each particle receives 10 input features: its velocity, the velocity change since the previous stored frame (2.08 ms) in training and since the previous learned step (8.3 ms) in rollouts, the external acceleration, and the distance to each of the four walls (clamped at 10 cm, in place of the solver's static wall particles). Pair features are the relative position, its absolute components, distance and direction to each of
the particle's `K = 24` nearest neighbours within 7 cm (`pos,axis,polar`). The network predicts an acceleration `a`, and the state is advanced by
`v' = v + Δt·a`, `x' = x + Δt·v'`, with the same box-boundary clamp in the integrator as the native solver.

* **LESNet**: four layers of width 128 on the neighbour-list kernels, 105k parameters (also width 256, 407k parameters).
* **Graph network** (the graph-network-simulator recipe): the same neighbour lists and inputs, message passing with a two-layer message function, mean aggregation, a node-update
  MLP and a residual path in every layer; width 72, 107k parameters.

Training is single-step regression of the normalised acceleration with Gaussian noise added to the input velocities (standard deviation about 0.04 m/s) and positions (standard deviation about 0.3 mm), so that rollouts that leave the
data manifold are pulled back. The learning rate is chosen per model from {1e-3, 3e-3, 1e-2} by the one-step validation loss after 5000 steps (3e-3 for both; the rollout metrics were not used for selection); the final models are trained for 30 000
steps with a cosine schedule, each on a batch of 8 whole scenes taken at random frames of random training trajectories. The 105k models are trained with four seeds.

### 7.4 Metrics

From the native state at several frames of each held-out trajectory, the learned simulator is rolled out for 0.5 simulated seconds under the same gravity and finger
schedule and compared with the native frames: the **Chamfer distance** between the two particle sets in particle spacings (permutation-free, appropriate for a chaotic fluid), the **per-particle position error**, the **density error** of the final state (native SPH estimator) and the **kinetic-energy ratio**
(learned over native, summarised by the median across rollouts). Because Chamfer distance is nearly blind to a uniform shimmer of one spacing, two further measurements are made. The **calm-down test** runs two evaluations on the eight held-out settle and quiet trajectories: first, starting from the end of each trajectory (second-to-last frame) under straight-down gravity alone, it rolls out for 3 s and records the mean particle speed over the final 0.75 s and the drift of the mean fluid height; second, starting at `t = 2.3 s` (so that 1 s of native continuation remains), it rolls out for 1 s and compares Chamfer distance and the median kinetic-energy ratio against the native fluid. The **speed analysis** bins particles by their native speed after 0.1 s and 0.25 s of rollout on the active scenes and reports the ratio of learned to native
speed in each bin. Because rollouts from the same trajectory share a scene and are not independent, confidence intervals on means are computed across the 12 trajectory-level means (each averaging 6 rollouts) using a t interval.
