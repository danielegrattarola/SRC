import numpy as np
import scipy.sparse as sp
import tensorflow as tf


def to_numpy(x):
    if sp.issparse(x):
        return x.A
    elif hasattr(x, "numpy"):
        return x.numpy()
    elif tf.keras.backend.is_sparse(x):
        return tf.sparse.to_dense(x).numpy()
    else:
        return np.array(x)
