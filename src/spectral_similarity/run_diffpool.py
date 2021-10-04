import argparse

from src.layers import DiffPool
from src.models.poolers import SimplePooler
from src.spectral_similarity.training import (results_to_file,
                                              run_experiment)

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="cora")
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--patience", type=int, default=50)
parser.add_argument("--tol", type=float, default=1e-6)
parser.add_argument("--runs", type=int, default=3)
args = parser.parse_args()


def create_model(**kwargs):
    pool = DiffPool(kwargs.get("k"), return_sel=True)
    model = SimplePooler(pool)

    return model


results = run_experiment(
    name=args.name,
    method="DiffPool",
    create_model=create_model,
    learning_rate=args.lr,
    es_patience=args.patience,
    es_tol=args.tol,
    runs=args.runs,
)
results_to_file(args.name, "DiffPool", *results)
