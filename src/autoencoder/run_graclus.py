import argparse

from src.autoencoder.training import results_to_file
from src.autoencoder.training_nt import run_experiment
from src.modules.graclus import GRACLUS, preprocess

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="Grid2d")
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--patience", type=int, default=1000)
parser.add_argument("--tol", type=float, default=1e-6)
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
    learning_rate=args.lr,
    es_patience=args.patience,
    es_tol=args.tol,
    runs=args.runs,
)
results_to_file(args.name, "Graclus", *results)
