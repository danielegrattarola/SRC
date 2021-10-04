import tensorflow as tf
from scipy import sparse
from spektral.layers import ops
from tensorflow.keras import backend as K

from src.layers.src import SRCPool


class LaPool(SRCPool):
    def __init__(self, shortest_path_reg=True, return_sel=False, **kwargs):
        super().__init__(return_sel=return_sel, **kwargs)

        self.shortest_path_reg = shortest_path_reg

    def call(self, inputs, **kwargs):
        X, A, I = self.get_inputs(inputs)

        # Select leaders
        L = laplacian(A)
        V = ops.modal_dot(L, X)
        V = tf.norm(V, axis=-1, keepdims=1)

        row = A.indices[:, 0]
        col = A.indices[:, 1]
        leader_check = tf.cast(tf.gather(V, row) >= tf.gather(V, col), tf.int32)
        leader_mask = ops.scatter_prod(leader_check[:, 0], row, self.N)
        leader_mask = tf.cast(leader_mask, tf.bool)

        return self.pool(X, A, I, leader_mask=leader_mask)

    def select(self, X, A, I, leader_mask=None):
        # Cosine similarity
        if I is None:
            I = tf.zeros(self.N, dtype=tf.int32)
        cosine_similarity = sparse_cosine_similarity(X, self.N, leader_mask, I)

        # Shortest path regularization
        if self.shortest_path_reg:

            def shortest_path(a_):
                return sparse.csgraph.shortest_path(a_, directed=False)

            np_fn_input = tf.sparse.to_dense(A) if K.is_sparse(A) else A
            beta = 1 / tf.numpy_function(shortest_path, [np_fn_input], tf.float64)
            beta = tf.where(tf.math.is_inf(beta), tf.zeros_like(beta), beta)
            beta = tf.boolean_mask(beta, leader_mask, axis=1)
            beta = tf.cast(
                tf.ensure_shape(beta, cosine_similarity.shape), cosine_similarity.dtype
            )
        else:
            beta = 1.0

        S = tf.sparse.softmax(cosine_similarity)
        S = beta * tf.sparse.to_dense(S)

        # Leaders end up entirely in their own cluster
        kronecker_delta = tf.boolean_mask(tf.eye(self.N), leader_mask, axis=1)

        # Create clustering
        S = tf.where(leader_mask[:, None], kronecker_delta, S)

        return S

    def reduce(self, X, S, **kwargs):
        return ops.modal_dot(S, X, transpose_a=True)

    def connect(self, A, S, **kwargs):
        return ops.matmul_at_b_a(S, A)

    def reduce_index(self, I, S, leader_mask=None):
        I_pool = tf.boolean_mask(I, leader_mask)

        return I_pool

    def get_config(self):
        config = {"shortest_path_reg": self.shortest_path_reg}
        base_config = super().get_config()
        return {**base_config, **config}


def laplacian(A):
    D = ops.degree_matrix(A, return_sparse_batch=True)
    if K.is_sparse(A):
        A = A.__mul__(-1)
    else:
        A = -A

    return tf.sparse.add(D, A)


def reduce_sum(x, **kwargs):
    if K.is_sparse(x):
        return tf.sparse.reduce_sum(x, **kwargs)
    else:
        return tf.reduce_sum(x, **kwargs)


def sparse_cosine_similarity(X, N, mask, I):
    mask = tf.cast(mask, tf.int32)
    leader_idx = tf.where(mask)

    # Number of nodes in each graph
    Ns = tf.math.segment_sum(tf.ones_like(I), I)
    Ks = tf.math.segment_sum(mask, I)

    # S will be block-diagonal matrix where entry i,j is the cosine
    # similarity between node i and leader j.
    # The code below creates the indices of the sparse block-diagonal matrix
    # Row indices of the block-diagonal S
    starts = tf.cumsum(Ns) - Ns
    starts = tf.repeat(starts, Ks)
    stops = tf.cumsum(Ns)
    stops = tf.repeat(stops, Ks)
    index_n = tf.ragged.range(starts, stops).flat_values

    # Column indices of the block-diagonal S
    index_k = tf.repeat(leader_idx, tf.repeat(Ns, Ks))
    index_k_for_S = tf.repeat(tf.range(tf.reduce_sum(Ks)), tf.repeat(Ns, Ks))

    # Make index int64
    index_n = tf.cast(index_n, tf.int64)
    index_k = tf.cast(index_k, tf.int64)
    index_k_for_S = tf.cast(index_k_for_S, tf.int64)

    # Compute similarity between nodes and leaders
    X_n = tf.gather(X, index_n)
    X_n_norm = tf.norm(X_n, axis=-1)
    X_k = tf.gather(X, index_k)
    X_k_norm = tf.norm(X_k, axis=-1)
    values = tf.reduce_sum(X_n * X_k, -1) / (X_n_norm * X_k_norm)

    # Create a sparse tensor for S
    indices = tf.stack((index_n, index_k_for_S), 1)
    S = tf.SparseTensor(
        values=values, indices=indices, dense_shape=(N, tf.reduce_sum(Ks))
    )
    S = tf.sparse.reorder(S)

    return S
