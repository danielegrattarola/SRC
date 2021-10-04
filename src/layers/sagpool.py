import tensorflow as tf
from spektral.layers import ops
from tensorflow.keras import backend as K

from src.layers.topk import TopKPool


class SAGPool(TopKPool):
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
            ratio,
            return_sel=return_sel,
            return_score=return_score,
            sigmoid_gating=sigmoid_gating,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            kernel_constraint=kernel_constraint,
            **kwargs
        )

    def call(self, inputs):
        X, A, I = self.get_inputs(inputs)

        # Graph filter for GNN
        if K.is_sparse(A):
            I_N = tf.sparse.eye(self.N, dtype=A.dtype)
            A_ = tf.sparse.add(A, I_N)
        else:
            I_N = tf.eye(self.N, dtype=A.dtype)
            A_ = A + I_N
        fltr = ops.normalize_A(A_)

        y = ops.modal_dot(fltr, K.dot(X, self.kernel))
        output = self.pool(X, A, I, y=y)
        if self.return_score:
            output.append(y)

        return output
