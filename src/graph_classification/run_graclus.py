import argparse

import numpy as np

from src.graph_classification.training_nt import (results_to_file,
                                                  run_experiments)
from src.modules.graclus import GRACLUS

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="PROTEINS")
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--patience", type=int, default=50)
parser.add_argument("--runs", type=int, default=3)
args = parser.parse_args()


def pooling(X, A):
    for i in range(len(A)):
        mask = np.array(A[i].sum(-1))[:, 0] != 0
        A[i] = A[i].tocsr()[mask, :][:, mask].tocoo()
        X[i] = X[i][mask]
    X, A, S = GRACLUS(X, A, [0, 1])
    A, A_pool = list(zip(*A))
    S = [s[0] for s in S]
    return X, A, A_pool, S


results = run_experiments(
    runs=args.runs,
    pooling=pooling,
    dataset_name=args.dataset,
    learning_rate=args.lr,
    batch_size=args.batch_size,
    patience=args.patience,
)
results_to_file(args.dataset, "Graclus", *results)
