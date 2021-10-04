import argparse

from spektral.utils import convolution

from src.modules.nmf import NMF, preprocess
from src.spectral_similarity.training_nt import (results_to_file,
                                                 run_experiment)

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="Grid2d")
parser.add_argument("--runs", type=int, default=3)
args = parser.parse_args()


def pooling(X, A):
    _, A_in = preprocess(X, A)
    A_in = convolution.gcn_filter(A)
    A_out, S_out = NMF([A_in], 0.5)
    return A, X, A_out[0], S_out[0]


results = run_experiment(name=args.name, method="NMF", pooling=pooling, runs=args.runs)
results_to_file(args.name, "NMF", *results)
