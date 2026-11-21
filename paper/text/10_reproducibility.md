## 10. Reproducibility

**Hardware and software.** NVIDIA GTX 1660 Ti (mobile; Turing, sm_75, 24 SMs, 6 GB, no tensor cores), Intel Core i7-10750H, 16 GB RAM, Windows 11. Python 3.11.9, PyTorch 2.4.1+cu124, nvcc 12.9, MSVC 2022. Everything is fp32. The benchmark data are the Top Quark Tagging
Reference Dataset (Zenodo, DOI 10.5281/zenodo.2603256), MNIST and CIFAR-10 (via torchvision), the Planetoid citation graphs (Cora, CiteSeer) and Tiny Shakespeare; the jet files are read with h5py and hdf5plugin.

**Code.** The code, the scripts that produce every table and figure, and the raw result files are released with this paper. Paths below are relative to the repository root.

| what | where |
|---|---|
| CUDA kernels (all-to-all and list layouts, grid neighbour search), autograd wrappers | `lesnet_cuda/kernels.py`, `neighbors.py` |
| LESNet model, float64 oracle, correctness and size-robustness tests | `lesnet_cuda/model.py`, `ref.py`, `tests.py`, `tests_sizes.py` |
| synthetic tasks, models, baselines, training loop and every synthetic sweep | `paper/tasks.py`, `models.py`, `train.py`, `sweeps.py`, `bench_speed.py` |
| jets: data conversion, one run, the campaign, GENConv, longer runs, learning-rate grids | `paper/toptag_prep.py`, `toptag.py`, `run_toptag.py`, `gen_jets.py`, `long_jets.py`, `long_jets2.py`, `long_jets3.py`, `sweep_ext.py`, `bench_toptag.py` |
| MNIST point sets, citation graphs, CIFAR-10, character language model | `paper/bench/{pointset,cora,cifar,charlm,timing,lib}.py` |
| fluid: native SPH (all pairs and cell list), data, learned simulators, evaluation, viewer | `fluid/` (`sph.py`, `gen_data.py`, `learned.py`, `train_sim.py`, `run_v2.py`, `evaluate.py`, `eval_v2.py`, `analysis_speed.py`, `scaling.py`, `interactive.py`) |
| softmax-weight probe, `nbody_acc` decomposition, operation counts | `paper/saturation_probe.py`, `nbody_check.py`, `flops.py` |
| tables and figures from the raw results, and the paper itself | `paper/aggregate.py`, `aggregate_bench.py`, `fig_architecture.py`, `fluid/figures.py`, `build_paper.py` |

**Raw results** are JSON lines (`paper/results/*.jsonl`, one line per run) and JSON files (`paper/results/bench/`, `paper/results/toptag_v2/`, `fluid/out/`) with the configuration, learning curve and wall time, together with captured console output of the correctness tests. `aggregate.py` and `build_paper.py` regenerate every table and figure of the paper from these files, except the two fluid figures that show trajectories (`fluid_strips_v2.png` and `fluid_speed_bins.png`), which are rendered from the trained simulators by `fluid/figures.py` and `fluid/analysis_speed.py`. The jet data are fetched from Zenodo and converted by `toptag_prep.py`; MNIST and CIFAR-10 are fetched by torchvision.

**Seeds.** Learning rates are tuned with one seed (seed 0). Synthetic experiments use seeds 0-2 (relation ablation and locality sweep: 0-1; capacity sweep: 0; parameter scale: 0-1), the benchmarks three seeds (graphs: ten), jets three seeds at 1000, 2000 and 11 800 steps (one at 11 800 for GENConv) and one at other budgets, the 105k fluid models four seeds and the 407k fluid model one. A seed fixes initialisation and the data stream. Floating-point atomics in the backward pass make
repeated runs agree only up to summation order.


