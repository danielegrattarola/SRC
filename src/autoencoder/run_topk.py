import argparse

import tensorflow as tf
from tensorflow.keras.layers import Lambda

from src.autoencoder.training import results_to_file, run_experiment
from src.layers.topk import TopKPool
from src.models.autoencoders import Autoencoder
from src.modules.upsampling import upsampling_from_matrix

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="Grid2d")
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--patience", type=int, default=1000)
parser.add_argument("--tol", type=float, default=1e-6)
parser.add_argument("--runs", type=int, default=3)
args = parser.parse_args()


def upsampling_top_k(inputs):
    if len(inputs) == 5:
        X, A, I, M, y = inputs
    else:
        X, A, M, y = inputs
        I = None

    S = tf.eye(tf.shape(M)[0])  # N x N
    S = tf.boolean_mask(S, M)  # K x N
    S = tf.transpose(S)  # N x K

    # Ensure that we are computing the left pseudo-inverse of S with shape N x K
    S = tf.transpose(tf.linalg.pinv(y * S))  # N x K  # K x N  # N x K

    return upsampling_from_matrix([X, A, I, S])


def make_model(F, **kwargs):
    pool = TopKPool(kwargs.get("ratio"), return_sel=True, return_score=True)
    lift = Lambda(upsampling_top_k)
    model = Autoencoder(F, pool, lift)
    return model


if __name__ == "__main__":
    results = run_experiment(
        name=args.name,
        method="TopK",
        create_model=make_model,
        learning_rate=args.lr,
        es_patience=args.patience,
        es_tol=args.tol,
        runs=args.runs,
    )
    results_to_file(args.name, "TopK", *results)
