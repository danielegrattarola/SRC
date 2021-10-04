import tensorflow as tf
from spektral.layers import ops
from tensorflow.keras import activations
from tensorflow.keras import backend as K

from src.layers.src import SRCPool


class DiffPool(SRCPool):
    def __init__(
        self,
        k,
        channels=None,
        return_sel=False,
        activation=None,
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
        self.k = k
        self.channels = channels
        self.activation = activations.get(activation)

    def build(self, input_shape):
        F = input_shape[0][-1]
        if self.channels is None:
            self.channels = F
        self.kernel_emb = self.add_weight(
            shape=(F, self.channels),
            name="kernel_emb",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )

        self.kernel_pool = self.add_weight(
            shape=(F, self.k),
            name="kernel_pool",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        super().build(input_shape)

    def call(self, inputs, mask=None):
        X, A, I = self.get_inputs(inputs)

        # Graph filter for GNNs
        if K.is_sparse(A):
            I_N = tf.sparse.eye(self.N, dtype=A.dtype)
            A_ = tf.sparse.add(A, I_N)
        else:
            I_N = tf.eye(self.N, dtype=A.dtype)
            A_ = A + I_N
        fltr = ops.normalize_A(A_)

        output = self.pool(X, A, I, fltr=fltr, mask=mask)
        return output

    def select(self, X, A, I, fltr=None, mask=None):
        S = ops.modal_dot(fltr, K.dot(X, self.kernel_pool))
        S = activations.softmax(S, axis=-1)
        if mask is not None:
            S *= mask[0]

        # Auxiliary losses
        LP_loss = self.link_prediction_loss(A, S)
        entr_loss = self.entropy_loss(S)
        if K.ndim(X) == 3:
            LP_loss = K.mean(LP_loss)
            entr_loss = K.mean(entr_loss)
        self.add_loss(LP_loss)
        self.add_loss(entr_loss)

        return S

    def reduce(self, X, S, fltr=None):
        Z = ops.modal_dot(fltr, K.dot(X, self.kernel_emb))
        Z = self.activation(Z)

        return ops.modal_dot(S, Z, transpose_a=True)

    def connect(self, A, S, **kwargs):
        return ops.matmul_at_b_a(S, A)

    def reduce_index(self, I, S):
        I_mean = tf.math.segment_mean(I, I)
        I_pool = ops.repeat(I_mean, tf.ones_like(I_mean) * self.k)

        return I_pool

    @staticmethod
    def link_prediction_loss(A, S):
        S_gram = ops.modal_dot(S, S, transpose_b=True)
        if K.is_sparse(A):
            LP_loss = tf.sparse.add(A, -S_gram)
        else:
            LP_loss = A - S_gram
        LP_loss = tf.norm(LP_loss, axis=(-1, -2))
        return LP_loss

    @staticmethod
    def entropy_loss(S):
        entr = tf.negative(
            tf.reduce_sum(tf.multiply(S, K.log(S + K.epsilon())), axis=-1)
        )
        entr_loss = K.mean(entr, axis=-1)
        return entr_loss

    def get_config(self):
        config = {"k": self.k, "channels": self.channels}
        base_config = super().get_config()
        return {**base_config, **config}
