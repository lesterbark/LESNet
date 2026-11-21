<!-- table:main -->
| task | LESNet | LESNet, 8 nearest | Transformer + relation | Transformer, no relation | graph network | graph network, 8 nearest | GENConv | DeepSets | MLP-flat |
|---|---|---|---|---|---|---|---|---|---|
| `nn_type` (majority 0.47) | 0.979 ± 0.003 | 0.982 ± 0.003 | 0.963 ± 0.015 | 0.957 ± 0.003 | 0.995 ± 0.001 | **0.996 ± 0.001** | 0.989 ± 0.004 | 0.761 ± 0.014 | 0.563 ± 0.002 |
| `count_r` | 0.916 ± 0.010 | 0.951 ± 0.002 | 0.877 ± 0.022 | 0.895 ± 0.015 | 0.989 ± 0.000 | **0.991 ± 0.001** | 0.969 ± 0.009 | 0.816 ± 0.007 | 0.009 ± 0.000 |
| `rank_ball` | 0.992 ± 0.001 | 0.960 ± 0.003 | 0.996 ± 0.001 | 0.996 ± 0.000 | **0.999 ± 0.000** | 0.964 ± 0.008 | 0.994 ± 0.001 | 0.981 ± 0.006 | 0.271 ± 0.000 |
| `closest_team` (majority 0.51) | **0.981 ± 0.038** | **0.995 ± 0.002** | 0.977 ± 0.004 | **0.898 ± 0.309** | **0.985 ± 0.024** | **0.993 ± 0.004** | **0.993 ± 0.031** | 0.594 ± 0.003 | 0.594 ± 0.014 |
| `exist_pair` (majority 0.54) | 0.972 ± 0.012 | 0.989 ± 0.004 | 0.945 ± 0.014 | 0.712 ± 0.019 | **0.992 ± 0.009** | **0.993 ± 0.006** | 0.975 ± 0.015 | 0.624 ± 0.007 | 0.627 ± 0.005 |
| `local_vel_avg` | 0.885 ± 0.007 | 0.913 ± 0.006 | 0.885 ± 0.019 | 0.878 ± 0.008 | 0.985 ± 0.002 | **0.988 ± 0.001** | 0.970 ± 0.004 | 0.659 ± 0.008 | 0.157 ± 0.001 |
| `nbody_acc` | 0.99976 ± 0.00002 | 0.99598 ± 0.00004 | 0.99931 ± 0.00004 | 0.99926 ± 0.00014 | 0.99950 ± 0.00010 | 0.99521 ± 0.00049 | **0.99984 ± 0.00003** | 0.98214 ± 0.00076 | 0.68967 ± 0.00251 |
<!-- /table -->

<!-- table:main_cost -->
| model | params | width | step cost relative to LESNet (indicative) |
|---|---|---|---|
| LESNet | 149,160 | 156, 188 | 1.0x |
| LESNet, 8 nearest | 149,160 | 156, 188 | 1.2x |
| Transformer + relation | 151,672 | 72, 80 | 1.5x |
| Transformer, no relation | 151,576 | 72, 80 | 1.3x |
| graph network | 152,176 | 92, 100 | 3.3x |
| graph network, 8 nearest | 152,176 | 92, 100 | 3.4x |
| GENConv | 152,618 | 116, 128 | 1.2x |
| DeepSets | 148,545 | 136, 156 | 0.8x |
| MLP-flat | 146,686 | 72 | 0.6x |
<!-- /table -->

<!-- table:ood -->
| task | model | in-distribution (n 8-24) | n = 48 | n = 96 |
|---|---|---|---|---|
| `nn_type` | LESNet | 0.979 ± 0.003 | 0.969 ± 0.003 | 0.944 ± 0.035 |
| `nn_type` | LESNet, 8 nearest | 0.982 ± 0.003 | 0.980 ± 0.005 | 0.980 ± 0.004 |
| `nn_type` | Transformer + relation | 0.963 ± 0.015 | 0.834 ± 0.181 | 0.634 ± 0.220 |
| `nn_type` | Transformer, no relation | 0.957 ± 0.003 | 0.764 ± 0.038 | 0.546 ± 0.021 |
| `nn_type` | graph network | 0.995 ± 0.001 | 0.977 ± 0.009 | 0.905 ± 0.050 |
| `nn_type` | graph network, 8 nearest | 0.996 ± 0.001 | 0.995 ± 0.001 | 0.988 ± 0.004 |
| `nn_type` | GENConv | 0.989 ± 0.004 | 0.985 ± 0.004 | 0.978 ± 0.007 |
| `nn_type` | DeepSets | 0.761 ± 0.014 | 0.606 ± 0.021 | 0.494 ± 0.022 |
| `nn_type` | MLP-flat | 0.563 ± 0.002 | n/a | n/a |
| `count_r` | LESNet | 0.916 ± 0.010 | 0.599 ± 0.319 | −0.444 ± 1.550 |
| `count_r` | LESNet, 8 nearest | 0.951 ± 0.002 | 0.954 ± 0.003 | 0.954 ± 0.003 |
| `count_r` | Transformer + relation | 0.877 ± 0.022 | 0.306 ± 0.090 | −1.000 ± 0.000 |
| `count_r` | Transformer, no relation | 0.895 ± 0.015 | 0.254 ± 0.176 | −1.000 ± 0.000 |
| `count_r` | graph network | 0.989 ± 0.000 | 0.744 ± 0.053 | −0.231 ± 0.526 |
| `count_r` | graph network, 8 nearest | 0.991 ± 0.001 | 0.987 ± 0.002 | 0.961 ± 0.027 |
| `count_r` | GENConv | 0.969 ± 0.009 | 0.377 ± 0.905 | −0.698 ± 1.300 |
| `count_r` | DeepSets | 0.816 ± 0.007 | 0.385 ± 0.113 | −0.806 ± 0.133 |
| `count_r` | MLP-flat | 0.009 ± 0.000 | n/a | n/a |
| `local_vel_avg` | LESNet | 0.885 ± 0.007 | 0.767 ± 0.036 | 0.431 ± 0.335 |
| `local_vel_avg` | LESNet, 8 nearest | 0.913 ± 0.006 | 0.904 ± 0.006 | 0.902 ± 0.005 |
| `local_vel_avg` | Transformer + relation | 0.885 ± 0.019 | 0.640 ± 0.053 | 0.017 ± 0.198 |
| `local_vel_avg` | Transformer, no relation | 0.878 ± 0.008 | 0.489 ± 0.028 | −0.569 ± 0.142 |
| `local_vel_avg` | graph network | 0.985 ± 0.002 | 0.921 ± 0.013 | 0.636 ± 0.277 |
| `local_vel_avg` | graph network, 8 nearest | 0.988 ± 0.001 | 0.980 ± 0.002 | 0.953 ± 0.013 |
| `local_vel_avg` | GENConv | 0.970 ± 0.004 | 0.791 ± 0.384 | −0.575 ± 1.827 |
| `local_vel_avg` | DeepSets | 0.659 ± 0.008 | 0.264 ± 0.031 | −0.260 ± 0.288 |
| `local_vel_avg` | MLP-flat | 0.157 ± 0.001 | n/a | n/a |
| `closest_team` | LESNet | 0.981 ± 0.038 | 0.964 ± 0.060 | 0.943 ± 0.035 |
| `closest_team` | LESNet, 8 nearest | 0.995 ± 0.002 | 0.992 ± 0.006 | 0.992 ± 0.010 |
| `closest_team` | Transformer + relation | 0.977 ± 0.004 | 0.971 ± 0.007 | 0.957 ± 0.018 |
| `closest_team` | Transformer, no relation | 0.898 ± 0.309 | 0.786 ± 0.408 | 0.688 ± 0.341 |
| `closest_team` | graph network | 0.985 ± 0.024 | 0.955 ± 0.048 | 0.854 ± 0.014 |
| `closest_team` | graph network, 8 nearest | 0.993 ± 0.004 | 0.991 ± 0.002 | 0.982 ± 0.010 |
| `closest_team` | GENConv | 0.993 ± 0.031 | 0.989 ± 0.039 | 0.985 ± 0.034 |
| `closest_team` | DeepSets | 0.594 ± 0.003 | 0.558 ± 0.022 | 0.528 ± 0.037 |
| `closest_team` | MLP-flat | 0.594 ± 0.014 | n/a | n/a |
<!-- /table -->

<!-- table:ablate -->
| variant | `nn_type` | `count_r` | `closest_team` | params |
|---|---|---|---|---|
| relation: position, axis | 0.926 ± 0.014 | 0.800 ± 0.101 | 0.957 ± 0.040 | 49,359 |
| no relation | 0.807 ± 0.122 | 0.783 ± 0.090 | 0.616 ± 0.009 | 48,548 |
| relation: position, axis, distance, direction | 0.958 ± 0.015 | 0.865 ± 0.192 | 0.983 ± 0.040 | 50,271 |
<!-- /table -->

<!-- table:capacity_closest_team -->
|  | depth 1 | depth 2 | depth 4 |
|---|---|---|---|
| width 32 | 0.970 | 0.994 | 0.994 |
| width 64 | 0.988 | 0.996 | 0.994 |
| width 128 | 0.990 | 0.978 | 0.959 |
| width 256 | 0.989 | 0.992 | 0.962 |
<!-- /table -->

<!-- table:capacity_local_vel_avg -->
|  | depth 1 | depth 2 | depth 4 |
|---|---|---|---|
| width 32 | 0.821 | 0.850 | 0.861 |
| width 64 | 0.829 | 0.853 | 0.877 |
| width 128 | 0.831 | 0.870 | 0.889 |
| width 256 | 0.834 | 0.877 | 0.884 |
<!-- /table -->

<!-- table:capacity_nn_type -->
|  | depth 1 | depth 2 | depth 4 |
|---|---|---|---|
| width 32 | 0.978 | 0.973 | 0.980 |
| width 64 | 0.978 | 0.975 | 0.979 |
| width 128 | 0.978 | 0.978 | 0.979 |
| width 256 | 0.978 | 0.982 | 0.979 |
<!-- /table -->

<!-- table:ksweep -->
| task | K=1 | K=2 | K=4 | K=8 | K=16 | all-to-all |
|---|---|---|---|---|---|---|
| `nn_type` | 1.000 ± 0.000 | 0.995 ± 0.001 | 0.983 ± 0.000 | 0.977 ± 0.013 | 0.973 ± 0.008 | 0.973 ± 0.001 |
| `count_r` | 0.446 ± 0.016 | 0.771 ± 0.023 | 0.918 ± 0.020 | 0.923 ± 0.006 | 0.892 ± 0.062 | 0.891 ± 0.055 |
| `local_vel_avg` | 0.507 ± 0.003 | 0.815 ± 0.005 | 0.901 ± 0.005 | 0.886 ± 0.055 | 0.868 ± 0.029 | 0.866 ± 0.009 |
<!-- /table -->

<!-- table:speed_ms -->
| model | N=16 | N=32 | N=64 | N=128 | N=256 | N=512 | N=1024 | N=2048 | N=4096 |
|---|---|---|---|---|---|---|---|---|---|
| LES all-to-all (CUDA) | 1.0x | 1.0x | 1.5x | 3.2x | 7.8x | 12.3x | 24.6x | 47.3x | 91.5x |
| LES list K=16 (CUDA) | 1.5x | 1.1x | 1.0x | 1.0x | 1.0x | 1.0x | 1.0x | 1.0x | 1.0x |
| LES all-to-all (PyTorch) | 9.0x | 12.0x | 23.0x | OOM | OOM | OOM | n/a | n/a | n/a |
| Transformer, no relation | 3.4x | 2.2x | 2.0x | 2.2x | 3.1x | 5.1x | 8.9x | 15.4x | 27.8x |
| Transformer + relation | 4.0x | 2.5x | 2.6x | 3.3x | 5.5x | 10.3x | 19.6x | n/a | n/a |
| GNN all-to-all (PyTorch) | 7.4x | 9.5x | 16.8x | OOM | OOM | n/a | n/a | n/a | n/a |
| GNN gather K=16 | 7.8x | 5.9x | 5.7x | 5.1x | 5.3x | 5.6x | 5.5x | 5.3x | 5.0x |
| LES list K=16 (CUDA), relative to its own N=16 | 1.00x | 0.99x | 0.97x | 1.07x | 1.03x | 0.98x | 0.98x | 1.04x | 1.12x |
<!-- /table -->

<!-- table:speed_mem -->
| model | N=16 | N=32 | N=64 | N=128 | N=256 | N=512 | N=1024 | N=2048 | N=4096 |
|---|---|---|---|---|---|---|---|---|---|
| LES all-to-all (CUDA) | 1.0x | 1.0x | 1.1x | 1.2x | 1.5x | 2.2x | 4.3x | 8.4x | 16.6x |
| LES list K=16 (CUDA) | 1.0x | 1.0x | 1.0x | 1.0x | 1.0x | 1.0x | 1.0x | 1.0x | 1.0x |
| LES all-to-all (PyTorch) | 8.6x | 16.8x | 33.3x | OOM | OOM | OOM | n/a | n/a | n/a |
| Transformer, no relation | 1.8x | 1.7x | 1.7x | 1.8x | 1.9x | 2.2x | 2.8x | 4.1x | 6.6x |
| Transformer + relation | 1.9x | 1.9x | 2.0x | 2.5x | 3.3x | 5.1x | 8.6x | n/a | n/a |
| GNN all-to-all (PyTorch) | 7.1x | 13.6x | 26.6x | OOM | OOM | n/a | n/a | n/a | n/a |
| GNN gather K=16 | 7.3x | 7.7x | 7.7x | 7.7x | 7.7x | 7.7x | 7.7x | 7.7x | 7.7x |
<!-- /table -->

<!-- table:fluid -->
| run | model | params | step (ms of sim) | Chamfer @0.5 s (spacings) | position err (spacings) | density err | KE ratio | ms / learned step | speed-up vs native |
|---|---|---|---|---|---|---|---|---|---|
| les_M2 | LESNet | 105,218 | 4.2 | 0.81 (sd 0.28) | 5.9 | 8.8% | 1.10 | 4.19 | 1.0x |
| gnn_M2 | graph network | 334,466 | 4.2 | 0.66 (sd 0.23) | 4.4 | 6.6% | 0.91 | 6.61 | 0.6x |
| gnn72_M2 | graph network | 107,498 | 4.2 | 0.66 (sd 0.23) | 4.6 | 6.7% | 0.94 | 6.39 | 0.6x |
| les_M4 | LESNet | 105,218 | 8.3 | 0.84 (sd 0.49) | 6.0 | 9.0% | 0.95 | 4.28 | 1.9x |
| gnn72_M4 | graph network | 107,498 | 8.3 | 0.64 (sd 0.25) | 4.8 | 7.4% | 0.92 | 6.72 | 1.2x |
| les256_M4 | LESNet | 407,042 | 8.3 | 0.75 (sd 0.23) | 5.7 | 9.2% | 0.99 | 4.25 | 1.9x |
| les512_M4 | LESNet | 1,600,514 | 8.3 | 0.68 (sd 0.14) | 5.6 | 8.6% | 0.98 | 4.37 | 1.8x |
| les_M8 | LESNet | 105,218 | 16.6 | 1.11 (sd 0.53) | 8.9 | 16.6% | 1.31 | 4.28 | 3.7x |
| gnn72_M8 | graph network | 107,498 | 16.6 | 0.91 (sd 0.48) | 6.6 | 11.3% | 0.82 | 6.77 | 2.4x |

native SPH: 0.96 wall-seconds per simulated second at 800 particles (timings on an idle GPU). Held-out scenes have 513-849 particles.
<!-- /table -->

<!-- table:fluid_big -->
| run | model | params | step (ms of sim) | Chamfer @0.5 s (spacings) | position err (spacings) | density err | KE ratio | ms / learned step | speed-up vs native |
|---|---|---|---|---|---|---|---|---|---|
| les_M2 | LESNet | 105,218 | 4.2 | 0.59 (sd 0.12) | 5.5 | 6.4% | 3.00 | 4.19 | 1.0x |
| gnn72_M2 | graph network | 107,498 | 4.2 | 0.56 (sd 0.08) | 4.9 | 7.6% | 4.24 | 6.39 | 0.6x |
| les_M4 | LESNet | 105,218 | 8.3 | 0.51 (sd 0.05) | 4.5 | 7.4% | 2.66 | 4.28 | 1.9x |
| gnn72_M4 | graph network | 107,498 | 8.3 | 0.56 (sd 0.09) | 5.5 | 4.9% | 1.65 | 6.72 | 1.2x |
| les_M8 | LESNet | 105,218 | 16.6 | 0.65 (sd 0.12) | 5.4 | 19.8% | 4.88 | 4.28 | 3.7x |
| gnn72_M8 | graph network | 107,498 | 16.6 | 0.47 (sd 0.04) | 4.6 | 14.7% | 2.59 | 6.77 | 2.4x |

native SPH: 0.96 wall-seconds per simulated second at 800 particles (timings on an idle GPU). Held-out scenes have 1620-2162 particles.
<!-- /table -->

<!-- table:fluid_noise -->
| run | model | params | step (ms of sim) | Chamfer @0.5 s (spacings) | position err (spacings) | density err | KE ratio | ms / learned step | speed-up vs native |
|---|---|---|---|---|---|---|---|---|---|
| les_M2 | LESNet | 105,218 | 4.2 | 0.81 (sd 0.28) | 5.9 | 8.8% | 1.10 | 4.19 | 1.0x |
| les_M2_n01 | LESNet | 105,218 | 4.2 | 0.81 (sd 0.39) | 6.0 | 9.3% | 1.18 | 4.18 | 1.0x |
| gnn72_M2 | graph network | 107,498 | 4.2 | 0.66 (sd 0.23) | 4.6 | 6.7% | 0.94 | 6.39 | 0.6x |
| gnn72_M2_n01 | graph network | 107,498 | 4.2 | 0.64 (sd 0.19) | 5.4 | 7.6% | 1.05 | 6.70 | 0.6x |

native SPH: 0.96 wall-seconds per simulated second at 800 particles (timings on an idle GPU). Held-out scenes have 513-849 particles.
<!-- /table -->

<!-- table:fluid_scaling -->
| particles | native | les_M2 | gnn72_M2 | les_M4 | gnn72_M4 | les_M8 | gnn72_M8 | les512_M4 |
|---|---|---|---|---|---|---|---|---|
| 500 | 1.45 | 1.03 (1.4x) | 1.58 (0.9x) | 0.50 (2.9x) | 0.78 (1.9x) | 0.27 (5.4x) | 0.40 (3.6x) | 0.51 (2.8x) |
| 1000 | 2.08 | 1.01 (2.1x) | 1.64 (1.3x) | 0.50 (4.1x) | 0.81 (2.5x) | 0.26 (8.1x) | 0.39 (5.4x) | 0.54 (3.8x) |
| 2000 | 3.32 | 1.07 (3.1x) | 1.89 (1.8x) | 0.51 (6.5x) | 0.92 (3.6x) | 0.26 (12.9x) | 0.48 (6.9x) | 0.83 (4.0x) |
| 3000 | 4.58 | 1.03 (4.5x) | 2.46 (1.9x) | 0.52 (8.8x) | 1.25 (3.7x) | 0.26 (17.9x) | 0.64 (7.2x) | 1.21 (3.8x) |
| 4000 | 5.98 | 1.20 (5.0x) | 3.37 (1.8x) | 0.63 (9.4x) | 1.63 (3.7x) | 0.27 (22.4x) | 0.85 (7.0x) | 1.36 (4.4x) |
<!-- /table -->

<!-- table:switch -->
| combine | switch | score on A before | score on B after switch | score on B from scratch |
|---|---|---|---|---|
| softmax | local_vel_avg -> nbody_acc | 0.653 ± 0.030 | 0.939 ± 0.048 | 0.917 ± 0.025 |
| softmax | nbody_acc -> local_vel_avg | 0.937 ± 0.012 | 0.554 ± 0.170 | 0.628 ± 0.014 |
| max | local_vel_avg -> nbody_acc | 0.676 ± 0.009 | 0.937 ± 0.026 | 0.908 ± 0.056 |
| max | nbody_acc -> local_vel_avg | 0.935 ± 0.028 | 0.651 ± 0.051 | 0.657 ± 0.045 |
| sum | local_vel_avg -> nbody_acc | 0.249 ± 0.302 | 0.848 ± 0.116 | 0.820 ± 0.025 |
| sum | nbody_acc -> local_vel_avg | 0.849 ± 0.191 | 0.320 ± 0.042 | 0.184 ± 0.086 |
<!-- /table -->

<!-- table:initsens -->
| task | combine | init x0.1 | init x1 | init x10 |
|---|---|---|---|---|
| `nn_type` | softmax | 0.795 ± 0.114 | 0.812 ± 0.035 | 0.495 ± 0.059 |
| `nn_type` | sum | 0.884 ± 0.012 | 0.573 ± 0.058 | 0.484 ± 0.138 |
| `nn_type` | max | 0.942 ± 0.004 | 0.925 ± 0.009 | 0.496 ± 0.001 |
<!-- /table -->

<!-- table:scale -->
| task | model | ~30k params | ~150k params | ~600k params |
|---|---|---|---|---|
| `nn_type` | LESNet | 0.978 ± 0.009 | 0.979 ± 0.000 | 0.980 ± 0.019 |
| `nn_type` | Transformer + relation | 0.941 ± 0.034 | 0.960 ± 0.007 | 0.961 ± 0.048 |
| `nn_type` | graph network | 0.991 ± 0.010 | 0.995 ± 0.002 | 0.995 ± 0.001 |
<!-- /table -->

<!-- table:isotime -->
| task | LESNet | LESNet, 8 nearest | Transformer + relation | graph network | graph network, 8 nearest |
|---|---|---|---|---|---|
| `nn_type` | 0.987 | 0.987 | 0.969 | 0.993 | 0.995 |
| `closest_team` | 0.977 | 0.997 | 0.989 | 0.972 | 0.993 |
| `local_vel_avg` | 0.916 | 0.931 | 0.910 | 0.983 | 0.987 |
<!-- /table -->

<!-- table:pairs -->
| task | comparison | mean paired difference | paired t-test p (n=3) |
|---|---|---|---|
| `nn_type` | LES - Transformer+pf | +0.016 | 0.032 |
| `nn_type` | LES - GNN | −0.016 | 0.002 |
| `nn_type` | LES - DeepSets | +0.218 | 0.000 |
| `count_r` | LES - Transformer+pf | +0.040 | 0.024 |
| `count_r` | LES - GNN | −0.073 | 0.001 |
| `count_r` | LES - DeepSets | +0.101 | 0.001 |
| `rank_ball` | LES - Transformer+pf | −0.005 | 0.011 |
| `rank_ball` | LES - GNN | −0.007 | 0.002 |
| `rank_ball` | LES - DeepSets | +0.011 | 0.018 |
| `closest_team` | LES - Transformer+pf | +0.004 | 0.749 |
| `closest_team` | LES - GNN | −0.004 | 0.690 |
| `closest_team` | LES - DeepSets | +0.387 | 0.001 |
| `exist_pair` | LES - Transformer+pf | +0.027 | 0.043 |
| `exist_pair` | LES - GNN | −0.020 | 0.053 |
| `exist_pair` | LES - DeepSets | +0.348 | 0.000 |
| `local_vel_avg` | LES - Transformer+pf | −0.000 | 0.937 |
| `local_vel_avg` | LES - GNN | −0.100 | 0.000 |
| `local_vel_avg` | LES - DeepSets | +0.226 | 0.000 |
| `nbody_acc` | LES - Transformer+pf | +0.000 | 0.000 |
| `nbody_acc` | LES - GNN | +0.000 | 0.010 |
| `nbody_acc` | LES - DeepSets | +0.018 | 0.000 |
<!-- /table -->

<!-- table:correctness -->
| suite | cases passed | max forward error | max gradient error |
|---|---|---|---|
| network-level checks (`tests.py`) | 25 | 7.3e-07 | 1.9e-06 |
| kernel size suite, all-to-all (`tests_sizes.py`) | 18 | 1.8e-06 | 1.3e-06 |
| kernel size suite, list layout | 14 | 1.7e-06 | 1.7e-06 |
<!-- /table -->

<!-- table:lr_tune -->
| model | lr 0.0003 | lr 0.001 | lr 0.003 | lr 0.01 | lr 0.03 | lr 0.1 | chosen |
|---|---|---|---|---|---|---|---|
| LESNet | 0.807 | 0.870 | 0.918 | 0.942 | 0.947 | 0.777 | 0.03 |
| Transformer, no relation | 0.882 | 0.906 | 0.936 | 0.817 | n/a | n/a | 0.003 |
| graph network | 0.901 | 0.972 | 0.992 | 0.989 | n/a | n/a | 0.003 |
| GENConv | n/a | 0.927 | 0.965 | 0.985 | 0.961 | n/a | 0.01 |
| DeepSets | 0.457 | 0.534 | 0.603 | 0.671 | 0.623 | 0.440 | 0.01 |
| MLP-flat | 0.435 | 0.436 | 0.437 | 0.436 | n/a | n/a | 0.003 |
<!-- /table -->

<!-- table:toptag -->
| model | params | steps | accuracy | AUC | 1/ε_B at ε_S = 50% | 1/ε_B at ε_S = 30% | seeds |
|---|---|---|---|---|---|---|---|
| DeepSets | 149,737 | 1,000 | 0.9213 ± 0.0015 | 0.9756 ± 0.0010 | 129 ± 8 | 425 ± 60 | 3 |
| DeepSets | 149,737 | 2,000 | 0.9262 ± 0.0013 | 0.9783 ± 0.0007 | 159 ± 12 | 525 ± 21 | 3 |
| DeepSets | 149,737 | 11,800 | 0.9322 ± 0.0007 | 0.9815 ± 0.0001 | 224 ± 1 | 841 ± 38 | 3 |
| Transformer + relation | 137,977 | 1,000 | 0.9296 ± 0.0012 | 0.9802 ± 0.0005 | 198 ± 7 | 645 ± 23 | 3 |
| Transformer + relation | 137,977 | 2,000 | 0.9323 ± 0.0007 | 0.9816 ± 0.0003 | 227 ± 21 | 778 ± 23 | 3 |
| Transformer + relation | 137,977 | 11,800 | 0.9364 ± 0.0002 | 0.9839 ± 0.0002 | 304 ± 11 | 1114 ± 84 | 3 |
| Transformer + relation | 137,977 | 23,600 | 0.9369 | 0.9842 | 318 | 1167 | 1 |
| graph network | 147,385 | 1,000 | 0.9321 ± 0.0006 | 0.9816 ± 0.0002 | 216 ± 5 | 701 ± 52 | 3 |
| graph network | 147,385 | 2,000 | 0.9350 ± 0.0006 | 0.9830 ± 0.0001 | 255 ± 6 | 885 ± 55 | 3 |
| graph network | 147,385 | 11,800 | 0.9386 ± 0.0002 | 0.9850 ± 0.0003 | 337 ± 24 | 1217 ± 234 | 3 |
| GENConv | 152,428 | 1,000 | 0.9310 ± 0.0008 | 0.9810 ± 0.0003 | 205 ± 12 | 644 ± 25 | 3 |
| GENConv | 152,428 | 2,000 | 0.9342 ± 0.0005 | 0.9827 ± 0.0002 | 252 ± 15 | 804 ± 21 | 3 |
| GENConv | 152,428 | 11,800 | 0.9379 | 0.9846 | 316 | 1109 | 1 |
| LESNet | 151,633 | 1,000 | 0.9279 ± 0.0016 | 0.9790 ± 0.0007 | 161 ± 12 | 491 ± 17 | 3 |
| LESNet | 151,633 | 2,000 | 0.9317 ± 0.0008 | 0.9811 ± 0.0002 | 200 ± 9 | 659 ± 54 | 3 |
| LESNet | 151,633 | 6,400 | 0.9360 | 0.9834 | 271 | 918 | 1 |
| LESNet | 151,633 | 11,800 | 0.9371 ± 0.0003 | 0.9840 ± 0.0001 | 286 ± 18 | 981 ± 105 | 3 |
| LESNet | 151,633 | 23,600 | 0.9376 | 0.9841 | 289 | 1091 | 1 |
| ParticleNet (published)† | 366,000 | n/a | 0.940 | 0.9858 | 397 | 1615 | n/a |
| Particle Flow Network (published)† | n/a | n/a | 0.932 | 0.9819 | 247 | 888 | n/a |
<!-- /table -->

<!-- table:washout -->
| layers | LESNet | LESNet + re-concatenated input | graph net (residual) | graph net (no residual) |
|---|---|---|---|---|
| 3 | 0.887 ± 0.004 | 0.888 ± 0.004 | 0.895 ± 0.002 | 0.894 ± 0.003 |
| 8 | 0.917 ± 0.003 | 0.920 ± 0.004 | 0.933 ± 0.022 | 0.920 ± 0.005 |
| 12 | 0.900 ± 0.030 | 0.923 ± 0.014 | 0.941 ± 0.061 | 0.252 ± 0.031 |
<!-- /table -->

<!-- table:multiswitch -->
| width | variant | seeds | target 1: switched / from scratch | target 2: switched / from scratch | target 4: switched / from scratch | target 6: switched / from scratch |
|---|---|---|---|---|---|---|
| 24 | softmax+leaky | 3 | 0.2007 / 0.1816 | 0.2987 / 0.2163 | 0.4523 / 0.2039 | 0.5707 / 0.1773 |
| 24 | max+leaky | 3 | 0.3046 / 0.2816 | 0.3261 / 0.3425 | 0.4509 / 0.3430 | 0.5455 / 0.3857 |
| 32 | softmax+leaky | 3 | 0.0704 / 0.0595 | 0.0820 / 0.0715 | 0.1471 / 0.0548 | 0.1827 / 0.0810 |
| 32 | max+leaky | 3 | 0.1748 / 0.1920 | 0.1537 / 0.2456 | 0.1794 / 0.1807 | 0.2427 / 0.1396 |
<!-- /table -->

<!-- table:hiddeneq -->
| variant | parameters | width | layers | `nn_type` | `count_r` | `closest_team` |
|---|---|---|---|---|---|---|
| plain LES (50k params) | 49,359 | 108 | 3 | 0.925 ± 0.003 | 0.800 ± 0.101 | 0.957 ± 0.040 |
| hidden pair layer (50k params) | 50,906 | 84 | 3 | 0.992 ± 0.011 | 0.968 ± 0.023 | 0.979 ± 0.062 |
| plain LES, same FLOPs as hidden (width 188) | 170,079 | 188 | 3 | 0.930 ± 0.001 | 0.803 ± 0.107 | 0.957 ± 0.245 |
| plain LES, 6 layers, same FLOPs as hidden (width 124) | 169,262 | 124 | 6 | 0.928 ± 0.001 | 0.891 ± 0.072 | 0.925 ± 0.031 |
| plain LES, 9 layers, same FLOPs as hidden (width 96) | 159,106 | 96 | 9 | 0.927 ± 0.017 | 0.891 ± 0.032 | 0.887 ± 0.679 |
<!-- /table -->

<!-- table:fluid_v2 -->
| model | params | Chamfer @0.5 s (spacings) | position err | density err | KE ratio |
|---|---|---|---|---|---|
| LESNet (105k) | 105,218 | 0.55 (sd 0.17) | 3.8 | 6.3% | 1.04 |
| graph network (107k) | 107,498 | 0.52 (sd 0.14) | 3.4 | 11.0% | 0.84 |
| LESNet (407k) | 407,042 | 0.51 (sd 0.12) | 3.4 | 5.1% | 0.92 |
<!-- /table -->

<!-- table:fluid_rest -->
| model | mean speed, last 0.75 s (m/s) | change of fluid height (mm) | Chamfer distance at 1 s (spacings) |
|---|---|---|---|
| LESNet (105k) | 0.111 | −4.3 | 0.46 |
| graph network (107k) | 0.074 | −9.7 | 0.48 |
| LESNet (407k) | 0.067 | −3.0 | 0.45 |
<!-- /table -->

<!-- table:fluid_big_v2 -->
| model | Chamfer @0.5 s | density err | KE ratio |
|---|---|---|---|
| LESNet (105k) | 0.51 | 5.6% | 2.90 |
| graph network (107k) | 0.51 | 26.6% | 2.09 |
| LESNet (407k) | 0.50 | 5.4% | 2.28 |
<!-- /table -->

<!-- table:fluid_scaling_v2 -->
| particles | SPH, cell list | SPH, all pairs | LESNet 105k, 8.3 ms | graph network, 8.3 ms | LESNet 407k, 8.3 ms |
|---|---|---|---|---|---|
| 500 | 1.93 | 2.86 | 1.00 | 1.47 | 0.95 |
| 1000 | 2.01 | 4.00 | 0.99 | 1.43 | 0.96 |
| 2000 | 1.75 | 6.26 | 0.94 | 1.76 | 0.99 |
| 4000 | n/a | n/a | 1.03 | 2.98 | 1.44 |
| 8000 | n/a | n/a | 2.17 | 5.89 | 3.02 |
| 16000 | n/a | n/a | 6.79 | 13.51 | 9.16 |
<!-- /table -->

<!-- table:fluid_cluster -->
| model | Chamfer @0.5 s (spacings) | position error (spacings) | density error |
|---|---|---|---|
| LESNet (105k), seed 0 | 0.55 ± 0.05 | 3.8 ± 0.8 | 6.3 ± 1.0% |
| graph network (107k), seed 0 | 0.52 ± 0.04 | 3.4 ± 0.7 | 11.0 ± 1.6% |
| LESNet (105k), seed 1 | 0.52 ± 0.05 | 3.6 ± 0.9 | 6.3 ± 1.5% |
| graph network (107k), seed 1 | 0.53 ± 0.06 | 3.5 ± 0.9 | 7.4 ± 0.6% |
| LESNet (105k), seed 2 | 0.54 ± 0.04 | 3.8 ± 0.7 | 5.9 ± 0.8% |
| graph network (107k), seed 2 | 0.53 ± 0.05 | 3.5 ± 0.8 | 9.3 ± 1.4% |
| LESNet (105k), seed 3 | 0.53 ± 0.05 | 3.4 ± 0.9 | 5.9 ± 1.2% |
| graph network (107k), seed 3 | 0.53 ± 0.05 | 3.4 ± 1.0 | 5.5 ± 1.2% |
| LESNet (407k), seed 0 | 0.51 ± 0.03 | 3.4 ± 0.6 | 5.1 ± 0.8% |
<!-- /table -->

<!-- table:fluid_cluster_diff -->
| LESNet minus graph network | Chamfer | position error (spacings) | density error |
|---|---|---|---|
| seed 0 | 0.028 ± 0.033 | 0.46 ± 0.35 | −4.73 ± 1.43% |
| seed 1 | −0.005 ± 0.027 | 0.06 ± 0.19 | −1.13 ± 1.07% |
| seed 2 | 0.011 ± 0.025 | 0.28 ± 0.48 | −3.35 ± 1.41% |
| seed 3 | −0.002 ± 0.021 | 0.00 ± 0.24 | 0.41 ± 0.30% |
| mean of 4 seeds | 0.008 ± 0.014 | 0.20 ± 0.16 | −2.20 ± 0.89% |
<!-- /table -->

<!-- table:softmax_probe -->
| task | learning rate | score | normalised entropy of the weights | mean largest weight | share of the gradient on the largest-gradient partner |
|---|---|---|---|---|---|
| `local_vel_avg` | 0.001 | 0.665 | 0.941 | 0.159 | 0.262 |
| `local_vel_avg` | 0.003 | 0.751 | 0.968 | 0.129 | 0.204 |
| `local_vel_avg` | 0.01 | 0.814 | 0.984 | 0.104 | 0.150 |
| `local_vel_avg` | 0.03 | 0.841 | 0.991 | 0.090 | 0.118 |
| `nn_type` | 0.001 | 0.861 | 0.889 | 0.207 | 0.320 |
| `nn_type` | 0.003 | 0.930 | 0.926 | 0.173 | 0.270 |
| `nn_type` | 0.01 | 0.956 | 0.962 | 0.129 | 0.187 |
| `nn_type` | 0.03 | 0.965 | 0.971 | 0.114 | 0.153 |
<!-- /table -->

<!-- table:nbody_check -->
| predictor of the `nbody_acc` target | R² |
|---|---|
| force of the 1 nearest partner only | 0.16 |
| force of the 2 nearest partners only | 0.33 |
| force of the 3 nearest partners only | 0.47 |
| force of the 5 nearest partners only | 0.68 |
| force of the 8 nearest partners only | 0.85 |
| linear function of the entity's position and the centroid of the set | 0.68 |
<!-- /table -->

<!-- table:multiswitch_remedy -->
| optimiser | combine | seeds | target 1: switched / scratch | target 3 | target 6 |
|---|---|---|---|---|---|
| Adam, lr 0.03 (the main run) | softmax+leaky | 3 | 0.201 / 0.182 | 0.401 / 0.188 | 0.571 / 0.177 |
| Adam, lr 0.03 (the main run) | max+leaky | 3 | 0.305 / 0.282 | 0.396 / 0.318 | 0.546 / 0.386 |
| Adam, lr 0.01 | softmax+leaky | 3 | 0.305 / 0.305 | 0.343 / 0.290 | 0.538 / 0.306 |
| Adam, lr 0.01 | max+leaky | 3 | 0.387 / 0.390 | 0.448 / 0.420 | 0.566 / 0.398 |
| AdamW, lr 0.03, weight decay 0.1 | softmax+leaky | 3 | 0.252 / 0.250 | 0.211 / 0.232 | 0.372 / 0.207 |
| AdamW, lr 0.03, weight decay 0.1 | max+leaky | 3 | 0.284 / 0.300 | 0.418 / 0.298 | 0.525 / 0.282 |
<!-- /table -->

<!-- table:toptag_bench -->
| model | width | params | step cost relative to LESNet | peak memory relative to LESNet |
|---|---|---|---|---|
| DeepSets | 136 | 149,737 | 0.3x | 0.5x |
| LESNet | 156 | 151,633 | 1.0x | 1.0x |
| Transformer + relation | 72 | 137,977 | 1.7x | 1.6x |
| graph network | 92 | 147,385 | 3.1x | 5.5x |
| GENConv | 116 | 152,428 | 5.7x | 4.1x |
<!-- /table -->

<!-- table:fluid_seeds -->
| model | training seed | kinetic-energy ratio at 0.5 s (all test trajectories) | one-step val loss | calm-down speed, quiet scenes (m/s) | calm-down kinetic-energy ratio at 1 s |
|---|---|---|---|---|---|
| LESNet (105k) | 0 | 1.04 | 0.0376 | 0.088 | 2.29 |
| LESNet (105k) | 1 | 1.00 | 0.0394 | 0.084 | 0.89 |
| LESNet (105k) | 2 | 0.92 | 0.0398 | 0.104 | 1.38 |
| LESNet (105k) | 3 | 0.91 | 0.0401 | 0.050 | 1.18 |
| graph network (107k) | 0 | 0.84 | 0.0323 | 0.069 | 0.86 |
| graph network (107k) | 1 | 0.81 | 0.0333 | 0.150 | 3.88 |
| graph network (107k) | 2 | 0.90 | 0.0307 | 0.125 | 3.16 |
| graph network (107k) | 3 | 0.79 | 0.0337 | 0.048 | 0.99 |
<!-- /table -->

<!-- table:noact -->
| variant | `nn_type` | `count_r` | `closest_team` | `local_vel_avg` |
|---|---|---|---|---|
| leaky ReLU (LESNet) | 0.926 ± 0.003 | 0.797 ± 0.016 | 0.948 ± 0.014 | 0.792 ± 0.003 |
| no activation | 0.926 ± 0.004 | 0.713 ± 0.002 | 0.945 ± 0.004 | 0.733 ± 0.005 |
<!-- /table -->

<!-- table:bench_cifar -->
| model | parameters | accuracy (%) | seeds |
|---|---|---|---|
| LESNet | 150,442 | 63.02 ± 0.08 | 3 |
| CNN | 162,994 | 88.51 ± 0.06 | 3 |
<!-- /table -->

<!-- table:bench_cifark25 -->
| model | parameters | accuracy (%) | seeds |
|---|---|---|---|
| LESNet | 150,442 | 60.41 ± 0.69 | 3 |
<!-- /table -->

<!-- table:bench_cifarlong -->
| model | parameters | accuracy (%) | seeds |
|---|---|---|---|
| LESNet | 150,442 | 68.50 | 1 |
| CNN | 162,994 | 89.01 | 1 |
<!-- /table -->

<!-- table:bench_pointset -->
| model | parameters | accuracy (%) | seeds |
|---|---|---|---|
| LESNet | 151,798 | 99.02 ± 0.09 | 3 |
| graph network | 147,854 | 99.15 ± 0.03 | 3 |
| Transformer + relation bias | 138,346 | 98.79 ± 0.04 | 3 |
| DeepSets | 150,426 | 98.52 ± 0.10 | 3 |
<!-- /table -->

<!-- table:bench_charlm -->
| model | parameters | bits per character | seeds |
|---|---|---|---|
| LESNet | 806,969 | 2.313 ± 0.009 | 3 |
| GPT | 827,281 | 2.237 ± 0.019 | 3 |
<!-- /table -->

<!-- table:bench_cora -->
| model | parameters | accuracy (%) | seeds |
|---|---|---|---|
| LESNet | 192,455 | 77.29 ± 1.28 | 10 |
| GCN | 92,231 | 80.87 ± 0.40 | 10 |
| GAT | 92,302 | 79.72 ± 0.89 | 10 |
<!-- /table -->

<!-- table:bench_citeseer -->
| model | parameters | accuracy (%) | seeds |
|---|---|---|---|
| LESNet | 482,950 | 64.55 ± 0.89 | 10 |
| GCN | 237,446 | 69.44 ± 1.11 | 10 |
| GAT | 237,516 | 69.99 ± 0.83 | 10 |
<!-- /table -->

<!-- table:realdata -->
| benchmark | model | parameters | score | seeds |
|---|---|---|---|---|
| MNIST point set, accuracy (%) | LESNet | 151,798 | 99.02 ± 0.22 | 3 |
| MNIST point set, accuracy (%) | graph network | 147,854 | 99.15 ± 0.07 | 3 |
| MNIST point set, accuracy (%) | Transformer + relation | 138,346 | 98.79 ± 0.10 | 3 |
| MNIST point set, accuracy (%) | DeepSets | 150,426 | 98.52 ± 0.25 | 3 |
| Cora, accuracy (%) | LESNet | 192,455 | 77.29 ± 0.92 | 10 |
| Cora, accuracy (%) | GCN | 92,231 | 80.87 ± 0.29 | 10 |
| Cora, accuracy (%) | GAT | 92,302 | 79.72 ± 0.63 | 10 |
| CiteSeer, accuracy (%) | LESNet | 482,950 | 64.55 ± 0.63 | 10 |
| CiteSeer, accuracy (%) | GCN | 237,446 | 69.44 ± 0.79 | 10 |
| CiteSeer, accuracy (%) | GAT | 237,516 | 69.99 ± 0.59 | 10 |
| CIFAR-10, accuracy after 40 epochs (%) | LESNet | 150,442 | 63.02 ± 0.19 | 3 |
| CIFAR-10, accuracy after 40 epochs (%) | CNN | 162,994 | 88.51 ± 0.14 | 3 |
| CIFAR-10, 5×5 window, accuracy after 40 epochs (%) | LESNet | 150,442 | 60.41 ± 1.72 | 3 |
| Tiny Shakespeare, validation bits per character | LESNet | 806,969 | 2.313 ± 0.023 | 3 |
| Tiny Shakespeare, validation bits per character | GPT | 827,281 | 2.237 ± 0.047 | 3 |
<!-- /table -->

<!-- table:bench_timing -->
| task | model | parameters | step cost relative to LESNet | peak memory relative to LESNet |
|---|---|---|---|---|
| CIFAR-10 | LESNet | 150,442 | 1.0x | 1.0x |
| CIFAR-10 | CNN | 162,994 | 0.5x | 0.5x |
| MNIST point set | LESNet | 151,798 | 1.0x | 1.0x |
| MNIST point set | graph network | 147,854 | 2.6x | 3.2x |
| MNIST point set | Transformer + relation bias | 138,346 | 1.3x | 1.3x |
| MNIST point set | DeepSets | 150,426 | 0.2x | 0.8x |
| character language model | LESNet | 806,969 | 1.0x | 1.0x |
| character language model | GPT | 827,281 | 0.8x | 1.0x |
<!-- /table -->

<!-- table:cost -->
| benchmark | model | parameters | step cost | peak memory |
|---|---|---|---|---|
| jet tagging | DeepSets | 149,737 | 0.3× | 0.5× |
| jet tagging | LESNet | 151,633 | 1.0× | 1.0× |
| jet tagging | Transformer + relation | 137,977 | 1.7× | 1.6× |
| jet tagging | graph network | 147,385 | 3.1× | 5.5× |
| jet tagging | GENConv | 152,428 | 5.7× | 4.1× |
| CIFAR-10 | LESNet | 150,442 | 1.0× | 1.0× |
| CIFAR-10 | CNN | 162,994 | 0.5× | 0.5× |
| MNIST point set | LESNet | 151,798 | 1.0× | 1.0× |
| MNIST point set | graph network | 147,854 | 2.6× | 3.2× |
| MNIST point set | Transformer + relation | 138,346 | 1.3× | 1.3× |
| MNIST point set | DeepSets | 150,426 | 0.2× | 0.8× |
| character language model | LESNet | 806,969 | 1.0× | 1.0× |
| character language model | GPT | 827,281 | 0.8× | 1.0× |
<!-- /table -->