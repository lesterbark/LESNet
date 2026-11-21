# paper/

The LESNet paper and everything needed to regenerate its tables and figures.

| file | what |
|---|---|
| `PAPER.md`, `PAPER.html` | the paper (the HTML is self-contained, figures inlined); built by `build_paper.py` |
| `text/*.md` | the prose, one file per section; numbers come from `results/` through `{{table:...}}` placeholders |
| `tables_manual/` | tables written by hand (the benchmark summary) |
| `tasks.py` | the seven synthetic entity tasks with exact targets |
| `models.py` | LESNet (CUDA kernels and a PyTorch reference implementation), MLP, DeepSets, Transformer, graph network (`gnn`), GENConv (`gen`) |
| `train.py` | one training run, written as one JSON record |
| `sweeps.py` | the synthetic-task sweeps: `lr_tune main ablate capacity ksweep scale noact` and the extended learning-rate and GENConv grids |
| `toptag_prep.py`, `toptag.py`, `run_toptag.py` | top-quark tagging: data conversion, one run, the campaign (learning-rate sweep and the 1000- and 2000-step runs) |
| `gen_jets.py`, `sweep_ext.py`, `long_jets.py`, `long_jets2.py`, `long_jets3.py`, `ensure_jets.py`, `rerun_jets.py`, `tf_long.py` | the GENConv jet runs, the extended learning-rate grids and the longer jet runs |
| `bench/` (`pointset.py`, `cora.py`, `cifar.py`, `charlm.py`, `timing.py`, `lib.py`), `aggregate_bench.py` | the real-data benchmarks: one script per task, the shared training loop and the LESNet list stack |
| `bench_speed.py`, `bench_toptag.py`, `flops.py` | step time and memory against set size and for the benchmark models; analytic operation counts |
| `saturation_probe.py`, `nbody_check.py` | the softmax-weight measurements and the decomposition of the `nbody_acc` target |
| `aggregate.py` | raw results to `results/tables.md` and `figures/*.png` |
| `fig_architecture.py` | the architecture figure |
| `results/` | raw results: one JSON line per run (`*.jsonl`), one JSON file per run in `results/bench/` and `results/toptag_v2/` |

```
python -u sweeps.py main        # resumable: running it again only fills in what is missing
python aggregate.py
python build_paper.py
```
