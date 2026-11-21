# fluid: native SPH against a learned LESNet simulator

A two-dimensional particle fluid (150 to 1000 particles in a unit box). `sph.py` is the reference solver (weakly
compressible SPH as CUDA kernels). `learned.py` and `train_sim.py` train LESNet, and a graph-network baseline on the same
inputs, to reproduce its dynamics `M` frames at a time. Details and results are in section 7 of `../paper/PAPER.md`.

## Pipeline

```
python gen_data.py         # the four large scenes (1500-2200 particles) of the large-scene test -> data/big_*.npz
python gen_data.py --v2    # native trajectories -> data2/*.npy: 160 training, 8 validation and 12 test trajectories of 3.5 s,
                           # in three kinds (active controls; controls then calm gravity; no controls), memory-mapped
python train_sim.py --kind les|gnn --M 4 --steps 40000 --data data2 --mirror --name les_M4_d2
                           # resumable; checkpoints in runs/<name>/; validation loss every 500 steps
python run_v2.py           # the whole campaign: learning-rate selection, then the long runs
python evaluate.py --runs ... --data data2      # rollouts against the native solver and timing -> out/fluid_eval*.json
python evaluate.py --mode rest ...              # the calm-down test
python eval_v2.py                               # all evaluations of the campaign (add --timing on an idle GPU)
python scaling.py --runs ...                    # speed against particle count, against the cell-list and the all-pairs solver
python figures.py --runs les_M4_d2 gnn72_M4_d2 --eval fluid_eval_v2.json --data data2 --tag _v2 --traj 1 --t0 60
```

`M` is the number of frames (2.08 ms each) per learned step, so `M = 4` gives 8.3 ms steps. Run names are
`les_M4_d2` (LESNet, 105k parameters), `les256_M4_d2` (LESNet, width 256, 407k) and `gnn72_M4_d2` (graph network, 107k);
`_s1`, `_s2` and `_s3` are further training seeds.

## Interactive viewer

```
python interactive.py --runs les_M4_d2 gnn72_M4_d2            # native solver and learned models side by side
python interactive.py --runs les_M4_d2 --no-native            # learned only
python interactive.py --record out/demo.gif --seconds 3 --runs les_M4_d2 gnn72_M4_d2
```

| input | effect |
|---|---|
| left mouse drag | push the fluid away from the cursor |
| right mouse drag | pull it in |
| arrow keys, `g` | rotate gravity, reset it |
| `1` to `5` | scene presets (dam break, two blocks, pool and drop, tall column, calm pool); `r` restarts |
| `space` | pause |
| `+`, `-` | finger strength |
| `f` | toggle the real-time cap |
| `q`, `Esc` | quit |

## Files

`sph.py` native solver (`method="cell"`, the default, uses a cell list; `"brute"` uses all pairs; they agree to float
precision) and scene helpers; `gen_data.py` data and control schedules; `learned.py` models, dataset and rollout;
`train_sim.py`; `evaluate.py`; `eval_v2.py`; `scaling.py`; `analysis_speed.py` (speed-bin analysis); `figures.py`;
`interactive.py` viewer. `out/*.json` holds the evaluation results and `runs/*/train_log.json` the training logs read by
`../paper/aggregate.py`.
