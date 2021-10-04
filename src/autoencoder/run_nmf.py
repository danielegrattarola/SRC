import argparse

from spektral.utils import convolution

from src.autoencoder.training import results_to_file
from src.autoencoder.training_nt import run_experiment
from src.modules.nmf import NMF, preprocess

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="Grid2d")
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--patience", type=int, default=1000)
parser.add_argument("--tol", type=float, default=1e-6)
parser.add_argument("--runs", type=int, default=3)
args = parser.parse_args()


def pooling(X, A):
    _, A_in = preprocess(X, A)
    A_in = convolution.gcn_filter(A)
    A_out, S_out = NMF([A_in], 0.5)
    return A, X, A_out[0], S_out[0]


results = run_experiment(
    name=args.name,
    method="NMF",
    pooling=pooling,
    learning_rate=args.lr,
    es_patience=args.patience,
    es_tol=args.tol,
    runs=args.runs,
)
results_to_file(args.name, "NMF", *results)
