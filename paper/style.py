"""Shared figure style: print-size fonts and one colour and marker per model family, used by every figure of the paper."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "lines.linewidth": 1.3, "lines.markersize": 3.5, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "figure.dpi": 150,
})

INK = "#1d2330"
# colour-blind-safe palette (Okabe and Ito), one entry per model family
STYLE = {
    "LESNet": ("#c0392b", "o"),
    "graph network": ("#0072b2", "s"),
    "Transformer": ("#009e73", "^"),
    "GENConv": ("#cc79a7", "D"),
    "DeepSets": ("#e69f00", "v"),
    "MLP-flat": ("#4d4d4d", "x"),
    "CNN": ("#56b4e9", "P"),
    "GCN": ("#56b4e9", "P"),
    "GAT": ("#999999", "X"),
}
# labels used in the result files -> (family, display name, dashed variant, hollow variant)
LABELS = {
    "LES": ("LESNet", "LESNet", False, False),
    "LES-kNN8": ("LESNet", "LESNet, 8 nearest", True, False),
    "Transformer+pf": ("Transformer", "Transformer + relation", False, False),
    "Transformer": ("Transformer", "Transformer, no relation", False, True),
    "GNN": ("graph network", "graph network", False, False),
    "GNN-kNN8": ("graph network", "graph network, 8 nearest", True, False),
    "GENConv": ("GENConv", "GENConv", False, False),
    "DeepSets": ("DeepSets", "DeepSets", False, False),
    "MLP-flat": ("MLP-flat", "MLP-flat", False, False),
}


def style_of(label):
    fam, name, dashed, hollow = LABELS[label]
    c, mk = STYLE[fam]
    return {"color": c, "marker": mk, "ls": "--" if dashed else ("-" if not hollow else ":"), "mfc": "white" if hollow else c, "name": name}
