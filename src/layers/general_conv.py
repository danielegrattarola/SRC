import tensorflow as tf
from spektral.layers import ops
from spektral.layers.convolutional.conv import Conv
from tensorflow.keras import activations
from tensorflow.keras.layers import BatchNormalization, Dropout, PReLU


class GeneralConv(Conv):
    def __init__(
        self,
        channels=256,
        batch_norm=True,
        dropout=0.0,
        activation="prelu",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super().__init__(
            activation=None,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )
        self.channels = channels
        self.dropout_rate = dropout
        self.use_batch_norm = batch_norm
        if activation == "prelu" or "prelu" in kwargs:
            self.activation = PReLU()
        else:
            self.activation = activations.get(activation)

    def build(self, input_shape):
        input_dim = input_shape[0][-1]
        self.dropout = Dropout(self.dropout_rate)
        if self.use_batch_norm:
            self.batch_norm = BatchNormalization()
        self.kernel = self.add_weight(
            shape=(input_dim, self.channels),
            initializer=self.kernel_initializer,
            name="kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.channels,),
                initializer=self.bias_initializer,
                name="bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        self.built = True

    def call(self, inputs, **kwargs):
        x, a = inputs

        x = tf.matmul(x, self.kernel)
        if self.use_bias:
            x = tf.nn.bias_add(x, self.bias)
        if self.use_batch_norm:
            x = self.batch_norm(x)
        x = self.dropout(x)
        x = self.activation(x)

        return ops.modal_dot(a, x)

    @property
    def config(self):
        config = {
            "channels": self.channels,
        }
        if self.activation.__class__.__name__ == "PReLU":
            config["prelu"] = True

        return config
