"""Analytic cost of one layer (multiply-adds counted as 2 FLOPs), so the measured speed can be read
against what each layer fundamentally has to do.

    LES (dense)   projections 4*N*C*W   +   pairs N*(N-1)*W*(2P + c_pair)
    LES (list)    projections 4*N*C*W   +   pairs N*K*W*(2P + c_pair)
    Transformer   16*N*W^2  (qkv, out, FFN 2W)   +   4*N^2*W  (scores and weighted sum)
    GNN (dense)   4*N*W^2 (A,B) + 4*N*W^2 (update)   +   N*(N-1)*(2*W^2 + 2*P*W + 3*W)  (message MLP)

`c_pair` is the per-(pair, slice) cost of the pair filter's adds, leaky ReLU and the online soft-max
(one exp, a few multiply-adds): taken as 20.
"""
C_PAIR = 20


def les(N, W, C, P, K=None):
    pairs = N * ((N - 1) if K is None else K)
    return 4 * N * C * W + pairs * W * (2 * P + C_PAIR)


def transformer(N, W):
    return 16 * N * W * W + 4 * N * N * W


def gnn(N, W, P, K=None):
    pairs = N * ((N - 1) if K is None else K)
    return 8 * N * W * W + pairs * (2 * W * W + 2 * P * W + 3 * W)


def table(N_list=(32, 256, 1024, 4096), W=128, C=128, P=7, K=16):
    rows = []
    for N in N_list:
        rows.append((N, les(N, W, C, P) / 1e9, les(N, W, C, P, K) / 1e9, transformer(N, W) / 1e9,
                     gnn(N, W, P) / 1e9, gnn(N, W, P, K) / 1e9))
    return rows


if __name__ == "__main__":
    print("GFLOP per layer per set, W=128, P=7, K=16")
    print(f"{'N':>6} {'LES dense':>10} {'LES K=16':>10} {'Transf':>10} {'GNN dense':>10} {'GNN K=16':>10}")
    for r in table():
        print(f"{r[0]:6d} " + " ".join(f"{v:10.3f}" for v in r[1:]))
