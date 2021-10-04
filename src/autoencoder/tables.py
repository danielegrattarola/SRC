import argparse
import glob

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="./results/")
args = parser.parse_args()

csvs = glob.glob(args.path + "*.csv")
csvs_df = [
    pd.read_csv(
        csv, header=None, index_col=0, names=[csv.split("_")[-2].split("/")[-1]]
    )
    for csv in csvs
]
df_out = pd.concat(csvs_df, axis=1, join="outer")


def formatter(s):
    if isinstance(s, float):
        return "OOR"
    avg, std = s.split("+-")
    avg = float(avg.strip())
    std = float(std.strip())
    out_avg = r"{:.5s}".format(f"{1e3 * avg:.4f}")
    out_std = r"{:.5s}".format(f"{1e3 * std:.4f}")
    out = fr"{out_avg} \tiny{{$\pm${out_std}}}"

    return out


def to_numerical(s):
    if s == "OOR":
        return 9999999999999999
    else:
        return float(s.split(" ")[0])


def crop(v):
    return f"{v:.2f}"


import json

dd = json.load(open("results/dist_mean_ref.json", "r"))

df_out = df_out.applymap(formatter)
df_out = df_out[["Grid2d", "Ring", "Bunny", "Airplane", "Car", "Guitar", "Person"]]
df_out = df_out.transpose()
df_out = df_out[
    ["DiffPool", "MinCut", "NMF", "LaPool", "TopK", "SAGPool", "NDP", "Graclus"]
]
df_out.columns = list(map(lambda s: rf"\textbf{{{s}}}", df_out.columns))
df_out.loc["Rank"] = (
    df_out.applymap(to_numerical)
    .rank(axis=1, ascending=True, method="min")
    .mean(axis=0)
    .map(crop)
)

with pd.option_context("max_colwidth", 1000):
    print(df_out.to_latex(escape=False, bold_rows=True))
