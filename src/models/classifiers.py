import tensorflow as tf
from spektral.layers import GlobalSumPool, GraphMasking
from spektral.models.general_gnn import MLP
from tensorflow.keras import Model
from tensorflow.keras.layers import Concatenate

from src.layers import GeneralConv


class MainModel(Model):
    def __init__(self, n_out, pool, mask=False, global_pool=None):
        super().__init__()
        # PReLU cannot be used in batch node if the number of nodes changes between
        # batches
        self.mask = mask
        if mask:
            self.masking_layer = GraphMasking()
        self.pre = MLP(256, activation="relu")
        self.gnn1 = GeneralConv(activation="relu")
        self.skip = Concatenate()
        self.pool = pool
        self.gnn2 = GeneralConv(activation="relu")
        if global_pool is None:
            self.global_pool = GlobalSumPool()
        else:
            self.global_pool = global_pool
        self.post = MLP(n_out, activation="relu", final_activation="softmax")

    def call(self, inputs, training=None, mask=None):
        if len(inputs) == 2:
            x, a = inputs
            i = None
        elif len(inputs) == 3:
            x, a, i = inputs
        else:
            raise ValueError("Input must be [x, a] or [x, a, i].")

        if self.mask:
            x = self.masking_layer(x)
        x = self.pre(x)
        x = self.skip([self.gnn1([x, a]), x])
        if self.mask:
            # Concatenation messes up the mask for some reason
            x._keras_mask = tf.cast(x._keras_mask[..., None], x.dtype)

        pool_input = [x, a]
        if i is not None:
            pool_input.append(i)
        pool_output = self.pool(pool_input)

        if len(pool_output) == 2:
            x, a = pool_output
        elif len(pool_output) == 3:
            x, a, i = pool_output

        x = self.skip([self.gnn2([x, a]), x])
        x = self.global_pool(x if i is None else [x, i])
        output = self.post(x)

        return output


class MfreeModel(MainModel):
    def call(self, inputs, training=None, mask=None):
        x, a, a_1, i_1, s = inputs

        x = self.pre(x)
        x = self.skip([self.gnn1([x, a]), x])

        x_1 = self.pool([x, s])

        x_1 = self.skip([self.gnn2([x_1, a_1]), x_1])
        x_1 = self.global_pool([x_1, i_1])
        output = self.post(x_1)

        return output
