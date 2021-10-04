import argparse
import glob

import numpy as np
import pandas as pd
import scipy.sparse as sp

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="results/")
parser.add_argument(
    "--datasets",
    nargs="*",
    default=["Grid2d", "Ring", "Bunny", "Minnesota", "Sensor", "Airfoil"],
)
args = parser.parse_args()

names = ["DiffPool", "MinCut", "NMF", "LaPool", "TopK", "SAGPool", "NDP", "Graclus"]
n_cols = len(names) + 1  # Number of columns in plot
threshold = 1e-9  # Sparsification threshold
density_csvs = []

for dataset_idx, dataset in enumerate(args.datasets):
    path = f"{args.path}{dataset}/"
    npzs = []
    for n in names:
        f = glob.glob(path + "{}*.npz".format(n))
        npzs.append(f[0])

    # Original graph
    A_orig = [np.load(f)["A"] for f in npzs]
    S_list = [np.load(f, allow_pickle=True)["S"] for f in npzs]
    A_pool = [np.load(f)["A_pool"] for f in npzs]

    density_csv = {}
    a = A_orig[0]
    a = np.where(a > threshold, a, 0)
    np.fill_diagonal(a, 0)
    a = sp.csr_matrix(a)
    density_csv["Original"] = (a.nnz / (a.shape[0] * a.shape[1]), np.median(a.data))
    for col in range(n_cols - 1):
        a = A_pool[col]
        a = np.where(a > threshold, a, 0)
        np.fill_diagonal(a, 0)
        a = sp.csr_matrix(a)
        density_csv[names[col]] = (a.nnz / (a.shape[0] * a.shape[1]), np.median(a.data))

    density_csv = pd.DataFrame(density_csv, index=["Density", "Median"])
    density_csv["Dataset"] = dataset
    density_csvs.append(density_csv)


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
