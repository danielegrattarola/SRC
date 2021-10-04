import argparse
import glob

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="results/")
parser.add_argument(
    "--datasets",
    nargs="*",
    default=[
        "Grid2d",
        "Ring",
        "Bunny",
        "Airplane",
        "Car",
        "Guitar",
        "Person",
    ],
)
parser.add_argument("--scale", action="store_true")
parser.add_argument("--fmt", default="png")
parser.add_argument("--show", action="store_true")  # , default="True")
parser.add_argument(
    "--graph-type", choices=["pool", "rec", "mod_red", "sel_mat"], default="rec"
)  # "rec", "pool" or "sel"

args = parser.parse_args()

names = ["DiffPool", "MinCut", "NMF", "LaPool", "TopK", "SAGPool", "NDP", "Graclus"]
rescale = lambda x_: x_ / x_.max()

# Config
mat_cmap = "Blues_r"  # Colormap used to plot matrices with imshow
n_cols = len(names) + 1  # Number of columns in plot
n_rows = len(args.datasets)  # Number of rows in plot
row = -1  # Current row
threshold = 1e-9  # Sparsification threshold
alpha = 0.5  # Alpha value for plotting nodes and edges
fig_scale = 0.5
figsize = (fig_scale * n_cols * 2, fig_scale * n_rows * 2)
fig = plt.figure(figsize=figsize)  # Init figure
fig.set_tight_layout(True)
density_csvs = []
dist_mean_ref = {}

for dataset_idx, dataset in enumerate(args.datasets):
    path = f"{args.path}{dataset}/"
    npzs = []
    avail = []
    for n in names:
        f = glob.glob(path + "{}*.npz".format(n))
        if len(f) > 0:
            npzs.append(f[0])
            avail.append(True)
        else:
            avail.append(False)

    # Original graph
    A_orig = [np.load(f)["A"] for f in npzs]
    X_orig = [np.load(f)["X"] for f in npzs]

    # Reconstructed graphs
    A_list = [np.load(f)["A_pred"] for f in npzs]
    A_list = [sp.csr_matrix(a) for a in A_list]
    X_list = [np.load(f)["X_pred"] for f in npzs]
    S_list = [np.load(f, allow_pickle=True)["S"] for f in npzs]
    A_pool = [np.load(f)["A_pool"] for f in npzs]
    X_pool = [np.load(f)["X_pool"] for f in npzs]
    A_pred = [np.load(f)["A_pred"] for f in npzs]
    X_pred = [np.load(f)["X_pred"] for f in npzs]
    times = [np.load(f)["training_times"] for f in npzs]
    losses = [np.load(f)["loss"] for f in npzs]

    ################################################################################
    # General plots
    ################################################################################
    col = 0
    row += 1

    # Original graph
    plt.figure(1)
    plt.subplot(n_rows, n_cols, row * n_cols + (col + 1))
    if dataset_idx == 0:
        plt.title("Original")
    a = A_orig[0]
    a = np.where(a > threshold, a, 0)
    a = sp.csr_matrix(a)
    G = nx.Graph(a)
    W = sp.triu(a)
    node_colors = X_orig[0][:, 0] + X_orig[0][:, 1]  # Colors of nodes
    edge_colors = W.tocoo().data
    pos = X_orig[0][:, :2]
    pos_orig_rs = rescale(pos)
    nx.draw(
        G,
        (pos[:, ::-1] * np.array([[1, -1]])) if dataset not in ["Bunny"] else pos,
        node_color="black",
        node_size=1,
        edge_color="#00000022",
    )
    plt.axis("on")
    plt.gca().tick_params(left=False, bottom=False, labelleft=True, labelbottom=True)
    plt.xticks([])
    plt.yticks([])
    plt.box(on=None)
    plt.ylabel(dataset)
    # if dataset_idx == 0:
    #     # plt.gca().set_facecolor((.5, 0.5, 0.5))
    #     plt.gca().set_facecolor("red")

    def dist_neigh_mean(X, A, A_th=1e-6):
        N, F = X.shape

        G = X.dot(X.T)
        dist_mean = np.diag(G).reshape(N, 1) + np.diag(G).reshape(1, N) - 2 * G
        dist_mean = dist_mean.sum() / N / (N - 1) / F

        np.fill_diagonal(A, 0)
        r, c = np.where(A > A_th)
        dist_adj_mean = ((X[r] - X[c]) ** 2).mean()

        return float(dist_mean), float(dist_adj_mean)

    dist_mean, dist_adj_mean = dist_neigh_mean(X_orig[0], A_orig[0])
    print(dataset, 1e3 * dist_mean, 1e3 * dist_adj_mean)
    dist_mean_ref[dataset] = (dist_mean, dist_adj_mean)

    ################################################################################
    # Plot pooled graphs as actual graphs
    ################################################################################
    density_csv = {}
    print("Plot pooled graphs as actual graphs")
    min_a, max_a, max_annz = 100, -1, -1
    shift = 0
    for col in range(n_cols - 1):
        if not avail[col]:
            shift += 1
            continue
        col -= shift
        plt.subplot(n_rows, n_cols, row * n_cols + (col + 2 + shift))
        if dataset_idx == 0:
            plt.title(names[col])

        if args.graph_type == "pool":
            a = A_pool[col]
            node_col = "tab:green"
            node_colors = X_pool[col][:, 0] + X_pool[col][:, 1]  # Colors of nodes
            pos = PCA(n_components=2).fit_transform(X_pool[col])
            pos = rescale(pos[:, :2])
        elif args.graph_type == "rec":
            a = A_orig[col]
            node_col = "orange"  # "tab:red"
            node_colors = X_pred[col][:, 0] + X_pred[col][:, 1]  # Colors of nodes
            pos = X_pred[col]
            pos = rescale(pos[:, :2])
        elif args.graph_type == "mod_red":
            a = A_pool[col]
            node_col = "tab:blue"
            s = S_list[col]
            pos = s.T.dot(X_orig[col])
            pos = rescale(pos[:, :2])

        if args.graph_type == "sel_mat":
            s = S_list[col]
            plt.imshow(s, cmap=plt.get_cmap("cividis"))

            plt.axis("off")
        else:
            a = np.where(a > threshold, a, 0)
            np.fill_diagonal(a, 0)
            a = sp.csr_matrix(a)
            G = nx.Graph(a)
            W = sp.triu(a)
            edge_colors = W.tocoo().data
            nx.draw(
                G,
                (pos[:, ::-1] * np.array([[1, -1]]))
                if dataset not in ["Bunny"]
                else pos,
                node_color=node_col,
                node_size=1,
                edge_color="#00000022",
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

        density_csv[names[col]] = (a.nnz / (a.shape[0] * a.shape[1]), np.median(a.data))
        if a.data.max() > max_a:
            max_a = a.data.max()
        if a.data.min() < min_a:
            min_a = a.data.min()
        if a.nnz > max_annz:
            max_annz = a.nnz

    density_csv = pd.DataFrame(density_csv, index=["Density", "Median"])
    density_csv["Dataset"] = dataset
    density_csvs.append(density_csv)

    ####################################################
    # Boxplots of the edge weights
    ####################################################
    plt.figure(2)
    bplot = []
    dens = []
    plt.subplot(n_rows, 1, row + 1)
    shift = 0
    for col in range(n_cols - 1):
        if not avail[col]:
            shift += 1
            continue
        col -= shift
        a = A_pool[col]
        a = np.where(a > threshold, a, 0)
        np.fill_diagonal(a, 0)
        a = sp.csr_matrix(a)

        bplot.append(a.data.ravel())
        dens.append(f"{names[col]}\n{a.nnz / (a.shape[0] * a.shape[1]):.2f}")

    plt.ylabel(dataset)
    plt.gca().yaxis.tick_left()

    plt.boxplot(bplot, showfliers=False)


# Save reference values
import json

json.dump(dist_mean_ref, open("results/dist_mean_ref.json", "w", encoding="utf-8"))

if args.show:
    plt.show()

plt.figure(1)
plt.tight_layout()
plt.subplots_adjust(wspace=4.6, hspace=0.6)
plt.savefig(
    args.path + "plot_datasets_{}.{}".format(args.graph_type, args.fmt),
    bbox_inches="tight",
    dpi=200,
    transparent=args.graph_type == "sel_mat",
)

plt.figure(2)
plt.tight_layout()
plt.subplots_adjust(wspace=0.2, hspace=0.2)
plt.savefig(
    args.path + "boxplot_datasets.{}".format(args.fmt), bbox_inches="tight", dpi=200
)


def formatter(s):
    if s > 1e-3:
        return f"{s:.3f}"
    else:
        out = f"{s:.2e}"
        exp = int(out[-3:])
        out = out[:-4] + fr"$\cdot 10^{{{exp}}}$"

        return out


df = pd.concat(density_csvs).reset_index().set_index(["Dataset", "index"])
df = df.applymap(formatter)
df.columns = list(map(lambda s: rf"\textbf{{{s}}}", df.columns))
with pd.option_context("max_colwidth", 1000):
    print(df.to_latex(escape=False, bold_rows=True, multirow=True))
