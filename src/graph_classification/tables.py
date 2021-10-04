import argparse
import glob

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="results/")
args = parser.parse_args()

csvs = glob.glob(args.path + "*.csv")
names = [csv.split("_")[-2].split("/")[-1] for csv in csvs]
csvs_df = [
    pd.read_csv(
        csv, header=None, index_col=0, names=[names[i] + " Loss", names[i] + " Acc"]
    )
    for i, csv in enumerate(csvs)
]

df_out = pd.concat(csvs_df, axis=1, join="outer")


def formatter(s):
    avg, std = s.split("+-")
    avg = float(avg.strip())
    std = float(std.strip())
    out = rf"{100*avg:.1f} \tiny{{$\pm${100*std:.1f}}}"

    return out


def to_numerical(s):
    if s == "OOR" or s == "$>10^3$" or isinstance(s, float):
        return np.nan
    else:
        return float(s.split(" ")[0])


def crop(v):
    return f"{v:.2f}"


df_out = df_out.applymap(formatter, na_action="ignore")

df_acc = df_out[df_out.columns[df_out.columns.str.contains(pat="Acc")]]
df_acc.columns = df_acc.columns.str.replace(" Acc", "")
df_acc = df_acc[
    ["COLORS-3", "TRIANGLES", "PROTEINS", "ENZYMES", "DD", "Mutagenicity", "10"]
]
df_acc.columns = [
    "Colors-3",
    "Triangles",
    "Proteins",
    "Enzymes",
    "DD",
    "Mutagen.",
    "ModelNet",
]
df_acc = df_acc.transpose()
df_acc = df_acc[
    ["Flat", "DiffPool", "MinCut", "NMF", "LaPool", "TopK", "SAGPool", "NDP", "Graclus"]
]
# df_acc = df_acc.transpose()
df_acc.columns = list(map(lambda s: rf"\textbf{{{s}}}", df_acc.columns))
df_acc.loc["Rank"] = (
    df_acc.applymap(to_numerical)
    .rank(axis=1, ascending=False, method="min")
    .mean(axis=0)
    .map(crop)
)

print(df_acc.to_latex(escape=False, bold_rows=True))
