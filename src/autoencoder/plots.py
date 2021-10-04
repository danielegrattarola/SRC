import argparse
import glob

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sp
from spektral.utils import laplacian

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="results/Ring/")
parser.add_argument("--scale", action="store_true")
parser.add_argument("--fmt", default="pdf")
parser.add_argument("--show", action="store_true")
args = parser.parse_args()

names = ["MinCut", "DiffPool", "NMF", "LaPool", "TopK", "SAGPool", "NDP", "Graclus"]
npzs = [glob.glob(args.path + "{}*.npz".format(n))[0] for n in names]

# Original graph
A_orig = np.load(npzs[0])["A"]
X_orig = np.load(npzs[0])["X"]

# Reconstructed graphs
A_list = [np.load(f)["A_pred"] for f in npzs]
A_list = [sp.csr_matrix(a) for a in A_list]
X_list = [np.load(f)["X_pred"] for f in npzs]
S_list = [np.load(f)["S"] for f in npzs]
A_pool = [np.load(f)["A_pool"] for f in npzs]
X_pool = [np.load(f)["X_pool"] for f in npzs]
times = [np.load(f)["training_times"] for f in npzs]
losses = [np.load(f)["loss"] for f in npzs]

rescale = lambda x_: x_ / x_.max()

# Plot parameters
mat_cmap = "Blues_r"  # Colormap used to plot matrices with imshow
n_cols = len(names)  # Number of columns in plot
n_rows = 4  # Number of rows in plot
row = -1  # Current row
threshold = 0.0  # Sparsification threshold
alpha = 0.5  # Alpha value for plotting nodes and edges
fig = plt.figure(figsize=(14, 2 * n_rows))  # Init figure
fig.set_tight_layout(True)

################################################################################
# General plots
################################################################################
a = A_orig
a = np.where(a > threshold, 1.0, 0)
a = sp.csr_matrix(a)
G = nx.Graph(a)
W = sp.triu(a)
node_colors = X_orig[:, 0] + X_orig[:, 1]  # Colors of nodes
edge_colors = W.tocoo().data
pos = X_orig[:, :2]
pos_orig_rs = rescale(pos)

################################################################################
# Plot the pooled adjacency matrix A_red
################################################################################
print("Plot the pooled adjacency matrix A_red")
row += 1
for col in range(n_cols):
    plt.subplot(n_rows, n_cols, row * n_cols + (col + 1))
    if col == 0:
        plt.ylabel(r"Pooled ($\mathbf{A}^\prime$)")
    plt.title(names[col])
    plt.imshow(A_pool[col], aspect="auto", cmap=mat_cmap)
    plt.xticks([])
    plt.yticks([])

################################################################################
# Plot pooled graphs as actual graphs
################################################################################
print("Plot pooled graphs as actual graphs")
row += 1
for col in range(n_cols):
    plt.subplot(n_rows, n_cols, row * n_cols + (col + 1))
    a = A_pool[col]
    a = np.where(a > threshold, a, 0)
    np.fill_diagonal(a, 0)
    a = sp.csr_matrix(a)
    G = nx.Graph(a)
    W = sp.triu(a)
    node_colors = X_pool[col][:, 0] + X_pool[col][:, 1]  # Colors of nodes
    edge_colors = W.tocoo().data
    pos = rescale(X_pool[col][:, :2])
    nx.draw(
        G,
        pos,
        node_size=5,
        node_color=node_colors,
        width=1,
        edge_color=edge_colors,
        edge_cmap=plt.cm.Greys,
        alpha=alpha,
        edge_vmin=0,
        edge_vmax=edge_colors.max(),
    )
    if args.scale:
        plt.xlim(
            min(pos[:, 0].min(), pos_orig_rs[:, 0].min()) - 0.1,
            max(pos[:, 0].max(), pos_orig_rs[:, 0].max()) + 0.1,
        )
        plt.ylim(
            min(pos[:, 1].min(), pos_orig_rs[:, 1].min()) - 0.1,
            max(pos[:, 1].max(), pos_orig_rs[:, 1].max()) + 0.1,
        )

################################################################################
# Plot the selection matrix S
################################################################################
print("Plot the selection matrix S")
row += 1
for col in range(n_cols):
    plt.subplot(n_rows, n_cols, row * n_cols + (col + 1))
    if col == 0:
        plt.ylabel(r"Select ($\mathbf{S}$)")
    s = S_list[col]
    plt.imshow(s, cmap=mat_cmap)
    plt.xticks([])
    plt.yticks([])

################################################################################
# Plot spectrum
################################################################################
row += 1
for col in range(n_cols):
    plt.subplot(n_rows, n_cols, row * n_cols + (col + 1))
    if col == 0:
        plt.ylabel("$\lambda_n$")
    else:
        plt.yticks([])
    plt.xlabel("$n / N$")
    a_pred, a_pool = A_list[col], A_pool[col]
    l_orig = laplacian(A_orig)
    l_pred = laplacian(a_pred)
    l_pool = laplacian(a_pool)
    eigvals_orig = sp.linalg.eigsh(l_orig, return_eigenvectors=False, k=A_orig.shape[0])
    eigvals_pred = sp.linalg.eigsh(
        l_pred.toarray(), return_eigenvectors=False, k=a_pred.shape[0]
    )
    eigvals_pool = sp.linalg.eigsh(l_pool, return_eigenvectors=False, k=a_pool.shape[0])
    eigvals_orig = eigvals_orig / eigvals_orig.max()
    eigvals_pred = eigvals_pred / eigvals_pred.max()
    eigvals_pool = eigvals_pool / eigvals_pool.max()
    plt.plot(np.linspace(0, 1, len(eigvals_orig)), eigvals_orig, label="Original")
    plt.plot(np.linspace(0, 1, len(eigvals_pred)), eigvals_pred, label="Predicted")
    plt.plot(np.linspace(0, 1, len(eigvals_pool)), eigvals_pool, label="Pooled")

plt.tight_layout()
plt.subplots_adjust(wspace=0.6, hspace=0.6)
plt.savefig(
    args.path + "plot_graphs_and_matrices.{}".format(args.fmt), bbox_inches="tight"
)

if args.show:
    plt.show()
