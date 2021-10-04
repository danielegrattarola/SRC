import argparse

from src.graph_classification.training_nt import (results_to_file,
                                                  run_experiments)
from src.modules.ndp import NDP, preprocess

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="PROTEINS")
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--patience", type=int, default=50)
parser.add_argument("--runs", type=int, default=3)
args = parser.parse_args()


def pooling(X, A):
    _, L = zip(*[preprocess(x, a) for x, a in zip(X, A)])
    A_pool, S = NDP(L, 1)
    return X, A, A_pool, S


results = run_experiments(
    runs=args.runs,
    pooling=pooling,
    dataset_name=args.dataset,
    learning_rate=args.lr,
    batch_size=args.batch_size,
    patience=args.patience,
)
results_to_file(args.dataset, "NDP", *results)
