import tensorflow as tf
from spektral.layers import ops
from tensorflow.keras import backend as K

from src.layers.src import SRCPool


class TopKPool(SRCPool):
    def __init__(
        self,
        ratio,
        return_sel=False,
        return_score=False,
        sigmoid_gating=False,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        kernel_constraint=None,
        **kwargs
    ):
        super().__init__(
            return_sel=return_sel,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            kernel_constraint=kernel_constraint,
            **kwargs
        )
        self.ratio = ratio
        self.return_score = return_score
        self.sigmoid_gating = sigmoid_gating
        self.gating_op = K.sigmoid if self.sigmoid_gating else K.tanh

    def build(self, input_shape):
        self.F = input_shape[0][-1]
        self.N = input_shape[0][0]
        self.kernel = self.add_weight(
            shape=(self.F, 1),
            name="kernel",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        super().build(input_shape)

    def call(self, inputs):
        X, A, I = self.get_inputs(inputs)
        y = K.dot(X, K.l2_normalize(self.kernel))
        output = self.pool(X, A, I, y=y)
        if self.return_score:
            output.append(y)

        return output

    def select(self, X, A, I, y=None):
        if I is None:
            I = tf.zeros(self.N)
        S = segment_top_k(y[:, 0], I, self.ratio)

        return tf.sort(S)

    def reduce(self, X, S, y=None):
        X_pool = tf.gather(X * self.gating_op(y), S)

        return X_pool

    def get_outputs(self, X_pool, A_pool, I_pool, S):
        output = [X_pool, A_pool]
        if I_pool is not None:
            output.append(I_pool)
        if self.return_sel:
            # Convert sparse indices to boolean mask
            S = tf.scatter_nd(S[:, None], tf.ones_like(S), (self.N,))
            output.append(S)

        return output

    def get_config(self):
        config = {
            "ratio": self.ratio,
        }
        base_config = super().get_config()
        return {**base_config, **config}


def segment_top_k(x, I, ratio):
    """
    Returns indices to get the top K values in x segment-wise, according to
    the segments defined in I. K is not fixed, but it is defined as a ratio of
    the number of elements in each segment.
    :param x: a rank 1 Tensor;
    :param I: a rank 1 Tensor with segment IDs for x;
    :param ratio: float, ratio of elements to keep for each segment;
    :return: a rank 1 Tensor containing the indices to get the top K values of
    each segment in x.
    """
    I = tf.cast(I, tf.int32)
    N = tf.shape(I)[0]
    n_nodes = tf.math.segment_sum(tf.ones_like(I), I)
    batch_size = tf.shape(n_nodes)[0]
    n_nodes_max = tf.reduce_max(n_nodes)
    cumulative_n_nodes = tf.concat(
        (tf.zeros(1, dtype=n_nodes.dtype), tf.cumsum(n_nodes)[:-1]), 0
    )
    index = tf.range(N)
    index = (index - tf.gather(cumulative_n_nodes, I)) + (I * n_nodes_max)

    dense_x = tf.zeros(batch_size * n_nodes_max, dtype=x.dtype) - 1e20
    dense_x = tf.tensor_scatter_nd_update(dense_x, index[:, None], x)
    dense_x = tf.reshape(dense_x, (batch_size, n_nodes_max))

    perm = tf.argsort(dense_x, direction="DESCENDING")
    perm = perm + cumulative_n_nodes[:, None]
    perm = tf.reshape(perm, (-1,))

    k = tf.cast(tf.math.ceil(ratio * tf.cast(n_nodes, tf.float32)), I.dtype)

    # This costs more memory
    # to_rep = tf.tile(tf.constant([1., 0.]), (batch_size,))
    # rep_times = tf.reshape(tf.concat((k[:, None], (n_nodes_max - k)[:, None]), -1), (-1,))
    # mask = ops.repeat(to_rep, rep_times)
    # perm = tf.boolean_mask(perm, mask)

    # This is slower
    r_range = tf.ragged.range(k).flat_values
    r_delta = ops.repeat(tf.range(batch_size) * n_nodes_max, k)
    mask = r_range + r_delta
    perm = tf.gather(perm, mask)

    # This crashes if not in eager mode (broadcasting problem between ragged and normal tensor??)
    # r_range = tf.ragged.range(k)
    # r_delta = (n_nodes_max * tf.range(batch_size))[:, None]
    # mask = r_range + r_delta
    # mask = mask.flat_values
    # perm = tf.gather(perm, mask)

    return perm
