import argparse

from spektral.data import BatchLoader

from src.graph_classification.training import results_to_file, run_experiments
from src.layers import DiffPool
from src.models.classifiers import MainModel

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="PROTEINS")
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--patience", type=int, default=50)
parser.add_argument("--runs", type=int, default=3)
args = parser.parse_args()


def create_model(n_out, **kwargs):
    pool = DiffPool(kwargs.get("k"))
    model = MainModel(n_out, pool, mask=True)

    return model


results = run_experiments(
    runs=args.runs,
    create_model=create_model,
    loader_class=BatchLoader,
    dataset_name=args.dataset,
    learning_rate=args.lr,
    batch_size=args.batch_size,
    patience=args.patience,
)
results_to_file(args.dataset, "DiffPool", *results)
