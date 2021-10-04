import argparse
import glob

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sp
from spektral.utils import degree_power, laplacian

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="results/Ring/")
parser.add_argument("--scale", action="store_true")
parser.add_argument("--show", action="store_true")
parser.add_argument("--fmt", default="png")
args = parser.parse_args()

names = ["DiffPool", "MinCut", "NMF", "LaPool", "TopK", "SAGPool", "NDP", "Graclus"]
npzs = [glob.glob(args.path + "{}*.npz".format(n))[0] for n in names]

X = [np.load(f)["X"] for f in npzs]
A = [np.load(f, allow_pickle=True)["A"] for f in npzs]
X_pool = [np.load(f)["X_pool"] for f in npzs]
A_pool = [np.load(f, allow_pickle=True)["A_pool"] for f in npzs]
S = [np.load(f, allow_pickle=True)["S"] for f in npzs]

# Config
n_rows = 3
n_cols = len(names) + 1
threshold = 1e-9  # Sparsification threshold
scale = 1.1  # Scale of the whole figure

# Sparsify pooled adjacency matrices
for i in range(len(A_pool)):
    A_pool[i][A_pool[i] < threshold] = 0

plt.figure(figsize=(n_cols * 1 * scale, n_rows * 1.2 * scale))

################################################################################
# Adjacency matrices row
################################################################################
row = 1
plt.subplot(n_rows, n_cols, (row - 1) * n_cols + 1)
plt.imshow(A[0])
plt.title("Original")
plt.xticks([])
plt.yticks([])
for col in range(len(names)):
    plt.subplot(n_rows, n_cols, (row - 1) * n_cols + col + 2)
    plt.imshow(A_pool[col])
    plt.title(names[col])
    plt.xticks([])
    plt.yticks([])

################################################################################
# NX graphs
################################################################################
row += 1
plt.subplot(n_rows, n_cols, (row - 1) * n_cols + 1)
nx.draw(
    nx.Graph(A[0]), pos=X[0][:, :2], node_size=1, node_color="k", edge_color="#00000022"
)
for col in range(len(names)):
    plt.subplot(n_rows, n_cols, (row - 1) * n_cols + col + 2)
    if names[col] in ["DiffPool", "MinCut", "TopK", "SAGPool"]:
        s = S[col]
        pos = s.T.dot(X[col])
    else:
        pos = X_pool[col]
    nx.draw(nx.Graph(A_pool[col]), pos=pos[:, :2], node_size=1, edge_color="#00000022")

################################################################################
# Plot spectrum
################################################################################
row += 1
for col in range(len(names)):
    # Compute Laplacians and eigenvalues
    D = degree_power(A[col], -0.5)
    L = D @ laplacian(A[col]) @ D
    D_pool = degree_power(A_pool[col], -0.5)
    L_pool = D_pool @ laplacian(A_pool[col]) @ D_pool

    eigvals = sp.linalg.eigsh(L, return_eigenvectors=False, k=A[col].shape[0])
    eigvals_pool = sp.linalg.eigsh(
        L_pool, return_eigenvectors=False, k=A_pool[col].shape[0]
    )

    # Plot all eigenvalues with normalized ranges
    plt.subplot(n_rows, n_cols, (row - 1) * n_cols + col + 2)
    plt.plot(np.linspace(0, 1, len(eigvals)), eigvals, c="k", label="Orig,")
    plt.plot(
        np.linspace(0, 1, len(eigvals_pool)), eigvals_pool, c="tab:blue", label="Pool."
    )

    plt.xlabel("$i/n$")
    plt.xticks([0, 1])
    if col == 0:
        plt.ylabel(r"$\lambda_i$")
        plt.legend(bbox_to_anchor=(-1.2, -0.3), loc="lower left")
    else:
        plt.yticks([])
    plt.gca().xaxis.set_label_coords(0.5, -0.2)
    plt.gca().yaxis.set_label_coords(-0.2, 0.5)

plt.tight_layout()
plt.subplots_adjust(wspace=0.1, hspace=0.2)
name = args.path.split("/")[-2]
plt.savefig(args.path + f"spectral_similarity_{name}.{args.fmt}", bbox_inches="tight")
if args.show:
    plt.show()
