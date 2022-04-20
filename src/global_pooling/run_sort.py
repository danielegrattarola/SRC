import argparse

from spektral.data import DisjointLoader
from spektral.layers import SortPool

from src.graph_classification.training import results_to_file, run_experiments
from src.models.classifiers import MainModel

import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="PROTEINS")
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--patience", type=int, default=50)
parser.add_argument("--runs", type=int, default=3)
args = parser.parse_args()


class SortPoolFlat(SortPool):
    def call(self, inputs):
        out = super().call(inputs)
        out_shape = tf.shape(out)

        # We have to hardcode the shape because Tensorflow crashes otherwise.
        # The layer is applied just after the skip connection of the second
        # message-passing layer in MainModel:
        #     - the input in disjoint mode has shape (batch*n_nodes, 3*256)
        #     - the output has shape (batch, k, 3*256)
        #     - we reshape to (batch, k*3*256)
        out = tf.reshape(out, (out_shape[0], self.k * 3 * 256))

        return out


def create_model(n_out, **kwargs):
    pool = lambda x: x
    k = int(kwargs.get("k") / kwargs.get("ratio"))
    global_pool = SortPoolFlat(k=k)
    model = MainModel(n_out, pool, global_pool=global_pool)

    return model


results = run_experiments(
    runs=args.runs,
    create_model=create_model,
    loader_class=DisjointLoader,
    dataset_name=args.dataset,
    learning_rate=args.lr,
    batch_size=args.batch_size,
    patience=args.patience,
)
results_to_file(args.dataset, "Sort", *results)
