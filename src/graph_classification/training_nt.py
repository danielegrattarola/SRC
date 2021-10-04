import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from spektral.data import Loader
from spektral.data.utils import to_disjoint
from spektral.layers import ops
from spektral.layers.ops import sp_matrix_to_sp_tensor
from tensorflow.keras.layers import Lambda

from src.graph_classification.training import load_dataset, main
from src.models.classifiers import MfreeModel

physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


class MFreeLoader(Loader):
    """
    Takes as input a MFreeDataset
    """

    def __init__(self, dataset, batch_size=1, epochs=None, shuffle=True):
        self.F = dataset.n_node_features
        self.n_out = dataset.n_labels
        self.a_dtype = tf.as_dtype(dataset[0].a.dtype)
        self.a_1_dtype = tf.as_dtype(dataset[0].a_1.dtype)
        self.s_dtype = tf.as_dtype(dataset[0].s.dtype)
        self.y_dtype = tf.as_dtype(dataset[0].y.dtype)
        super().__init__(dataset, batch_size=batch_size, epochs=epochs, shuffle=shuffle)

    def _pack(self, batch):
        return [
            list(elem) for elem in zip(*[[g.x, g.a, g.a_1, g.y, g.s] for g in batch])
        ]

    def collate(self, batch):
        packed = self._pack(batch)
        x, a, _ = to_disjoint(*packed[:2])
        a_1 = sp.block_diag(packed[2])
        y = np.array(packed[3]).astype("f4")
        s = sp.block_diag(packed[4]).astype("f4")
        n_nodes = np.array([a.shape[0] for a in packed[2]])
        batch_idx = np.repeat(np.arange(len(n_nodes)), n_nodes)
        a = sp_matrix_to_sp_tensor(a.astype("f4"))
        a_1 = sp_matrix_to_sp_tensor(a_1.astype("f4"))
        s = sp_matrix_to_sp_tensor(s.astype("f4"))

        return (x, a, a_1, batch_idx, s), y

    def tf(self):
        pass

    def tf_signature(self):
        return (
            (
                tf.TensorSpec((None, self.F)),
                tf.SparseTensorSpec((None, None), dtype=tf.float32),
                tf.SparseTensorSpec((None, None), dtype=tf.float32),
                tf.TensorSpec((None,), dtype=tf.int64),
                tf.SparseTensorSpec((None, None), dtype=tf.float32),
            ),
            tf.TensorSpec((None, self.n_out), dtype=tf.float32),
        )


def downsampling(inputs):
    X, S = inputs
    return ops.modal_dot(S, X, transpose_a=True)


def run_experiments(runs, pooling, dataset_name, learning_rate, batch_size, patience):
    # Data
    dataset, dataset_te = load_dataset(dataset_name)

    def pooling_transform(graph):
        graph.x, graph.a, graph.a_1, graph.s = pooling([graph.x], [graph.a])
        graph.x, graph.a, graph.a_1, graph.s = (
            graph.x[0],
            graph.a[0],
            graph.a_1[0],
            graph.s[0],
        )
        return graph

    dataset.apply(pooling_transform)
    if dataset_te is not None:
        dataset_te.apply(pooling_transform)

    # Model
    def create_model(n_out, **kwargs):
        pool = Lambda(downsampling)
        model = MfreeModel(n_out, pool)

        return model

    results = []
    for r in range(runs):
        print("{} of {}".format(r + 1, runs))
        results.append(
            main(
                dataset,
                create_model,
                MFreeLoader,
                learning_rate,
                batch_size,
                patience,
                dataset_te=dataset_te,
            )
        )
    avg_results = np.mean(results, axis=0)
    std_results = np.std(results, axis=0)

    print(
        "{} - Test loss: {:.4f} +- {:.4f} - Test accuracy: {:.4f} +- {:.4f}".format(
            dataset_name, avg_results[0], std_results[0], avg_results[1], std_results[1]
        )
    )

    return avg_results, std_results


def results_to_file(dataset, method, avg_results, std_results):
    filename = "{}_result.csv".format(dataset)
    with open(filename, "a") as f:
        line = "{}, {:.4f} +- {:.4f}, {:.4f} +- {:.4f}\n".format(
            method, avg_results[0], std_results[0], avg_results[1], std_results[1]
        )
        f.write(line)
