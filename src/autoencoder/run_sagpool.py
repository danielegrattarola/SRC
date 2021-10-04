import argparse

from tensorflow.keras.layers import Lambda

from src.autoencoder.run_topk import upsampling_top_k
from src.autoencoder.training import results_to_file, run_experiment
from src.layers import SAGPool
from src.models.autoencoders import Autoencoder

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="Grid2d")
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--patience", type=int, default=1000)
parser.add_argument("--tol", type=float, default=1e-6)
parser.add_argument("--runs", type=int, default=3)
args = parser.parse_args()


def make_model(F, **kwargs):
    pool = SAGPool(kwargs.get("ratio"), return_sel=True, return_score=True)
    lift = Lambda(upsampling_top_k)
    model = Autoencoder(F, pool, lift)
    return model


results = run_experiment(
    name=args.name,
    method="SAGPool",
    create_model=make_model,
    learning_rate=args.lr,
    es_patience=args.patience,
    es_tol=args.tol,
    runs=args.runs,
)
results_to_file(args.name, "SAGPool", *results)
