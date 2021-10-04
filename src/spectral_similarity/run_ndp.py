import argparse

from src.modules.ndp import NDP, preprocess
from src.spectral_similarity.training_nt import (results_to_file,
                                                 run_experiment)

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="Grid2d")
parser.add_argument("--runs", type=int, default=3)
args = parser.parse_args()


def pooling(X, A):
    _, L = preprocess(X, A)
    A_out, S_out = NDP([L], 1)
    return A, X, A_out[0], S_out[0]


results = run_experiment(name=args.name, method="NDP", pooling=pooling, runs=args.runs)
results_to_file(args.name, "NDP", *results)
