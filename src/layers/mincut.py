import tensorflow as tf
from spektral.layers import ops
from tensorflow.keras import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense

from src.layers.src import SRCPool


class MinCutPool(SRCPool):
    def __init__(
        self,
        k,
        mlp_hidden=None,
        mlp_activation="relu",
        return_sel=False,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super().__init__(
            return_sel=return_sel,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )

        self.k = k
        self.mlp_hidden = mlp_hidden if mlp_hidden is not None else []
        self.mlp_activation = mlp_activation

    def build(self, input_shape):
        layer_kwargs = dict(
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
        )
        self.mlp = Sequential(
            [
                Dense(channels, self.mlp_activation, **layer_kwargs)
                for channels in self.mlp_hidden
            ]
            + [Dense(self.k, "softmax", **layer_kwargs)]
        )

        super().build(input_shape)

    def call(self, inputs, mask=None):
        X, A, I = self.get_inputs(inputs)
        return self.pool(X, A, I, mask=mask)

    def select(self, X, A, I, mask=None):
        S = self.mlp(X)
        if mask is not None:
            S *= mask[0]

        # Orthogonality loss
        ortho_loss = self.orthogonality_loss(S)
        if K.ndim(A) == 3:
            ortho_loss = K.mean(ortho_loss)
        self.add_loss(ortho_loss)

        return S

    def reduce(self, X, S):
        return ops.modal_dot(S, X, transpose_a=True)

    def connect(self, A, S):
        A_pool = ops.matmul_at_b_a(S, A)

        # MinCut loss
        cut_loss = self.mincut_loss(A, S, A_pool)
        if K.ndim(A) == 3:
            cut_loss = K.mean(cut_loss)
        self.add_loss(cut_loss)

        # Post-processing of A
        A_pool = tf.linalg.set_diag(
            A_pool, tf.zeros(K.shape(A_pool)[:-1], dtype=A_pool.dtype)
        )
        A_pool = ops.normalize_A(A_pool)

        return A_pool

    def reduce_index(self, I, S):
        I_mean = tf.math.segment_mean(I, I)
        I_pool = ops.repeat(I_mean, tf.ones_like(I_mean) * self.k)

        return I_pool

    def orthogonality_loss(self, S):
        SS = ops.modal_dot(S, S, transpose_a=True)
        I_S = tf.eye(self.k, dtype=SS.dtype)
        ortho_loss = tf.norm(
            SS / tf.norm(SS, axis=(-1, -2), keepdims=True) - I_S / tf.norm(I_S),
            axis=(-1, -2),
        )
        return ortho_loss

    @staticmethod
    def mincut_loss(A, S, A_pool):
        num = tf.linalg.trace(A_pool)
        D = ops.degree_matrix(A)
        den = tf.linalg.trace(ops.matmul_at_b_a(S, D))
        cut_loss = -(num / den)
        return cut_loss

    def get_config(self):
        config = {
            "k": self.k,
            "mlp_hidden": self.mlp_hidden,
            "mlp_activation": self.mlp_activation,
        }
        base_config = super().get_config()
        return {**base_config, **config}
