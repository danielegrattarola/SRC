import argparse

from src.modules.graclus import GRACLUS, preprocess
from src.spectral_similarity.training_nt import (results_to_file,
                                                 run_experiment)

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="Grid2d")
parser.add_argument("--runs", type=int, default=3)
args = parser.parse_args()


def pooling(X, A):
    X, A = preprocess(X, A)
    X_out, A_out, S_out = GRACLUS([X], [A], [0, 1])
    return A_out[0][0], X_out[0], A_out[0][1], S_out[0][0]


results = run_experiment(
    name=args.name,
    method="Graclus",
    pooling=pooling,
    runs=args.runs,
)
results_to_file(args.name, "Graclus", *results)
