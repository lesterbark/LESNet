"""Convert the Top Quark Tagging Reference Dataset (Zenodo 2603256: train.h5 / val.h5 / test.h5, pandas HDF5) into
memory-mappable arrays:  data_toptag/{split}_X.npy  (n, 100, 4) float32  = (E, px, py, pz) of the 100 leading constituents
                         data_toptag/{split}_y.npy  (n,) int8             = 1 for a top-quark jet, 0 for a QCD jet

    run.bat toptag_prep.py            # all splits that were downloaded

h5py is installed in paper/deps (pip --target, so the global site-packages are untouched) and appended to sys.path.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "deps"))
import hdf5plugin  # noqa: E402,F401  (registers the blosc filter the pandas files use)
import h5py  # noqa: E402
import numpy as np  # noqa: E402

D = os.path.join(HERE, "data_toptag")
NC = 100
CHUNK = 50_000


def convert(split):
    src = os.path.join(D, f"{split}.h5")
    if not os.path.exists(src):
        return
    with h5py.File(src, "r") as f:
        g = f["table"]["table"] if "table" in f["table"] else None
        names = g.dtype.names
        print(split, "fields:", names, "rows:", g.shape, flush=True)
        n = g.shape[0]
        X = np.lib.format.open_memmap(os.path.join(D, f"{split}_X.npy"), mode="w+", dtype=np.float32, shape=(n, NC, 4))
        y = np.zeros(n, np.int8)
        # column order inside values_block_0 is INTERLEAVED per constituent: E_0, PX_0, PY_0, PZ_0, E_1, PX_1, ... (800
        # values), then the truth four-vector (4 values); the ints in values_block_1 are (ttv, is_signal_new).
        for s in range(0, n, CHUNK):
            blk = g[s:s + CHUNK]
            v0 = blk["values_block_0"]
            v1 = blk["values_block_1"] if "values_block_1" in names else None
            m = len(blk)
            for c in range(4):
                X[s:s + m, :, c] = v0[:, c:4 * NC:4]
            y[s:s + m] = v1[:, -1] if v1 is not None else v0[:, -1]
            print(f"{split} {s + m}/{n}", flush=True)
        X.flush()
        chk = np.asarray(X[:2000])
        E, p2 = chk[..., 0], (chk[..., 1:] ** 2).sum(-1)
        phys = float(((E ** 2 - p2) >= -1e-2 * E ** 2 - 1e-3)[E > 0].mean())
        print(f"{split}: fraction of constituents with E >= |p| (sanity check of the column layout): {phys:.4f}; "
              f"mean constituents per jet {(E > 0).sum(1).mean():.1f}", flush=True)
        assert phys > 0.99, "column layout is wrong"
        np.save(os.path.join(D, f"{split}_y.npy"), y)
        print(split, "signal fraction", y.mean(), flush=True)


if __name__ == "__main__":
    for sp in ("val", "test", "train"):
        convert(sp)
