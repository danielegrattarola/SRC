from tensorflow.keras import Model


class SimplePooler(Model):
    """
    A Model that applies a pooling layer to input graphs.
    """

    def __init__(self, pool):
        """
        :param pool: callable, a pooling operator that takes as input a list [x, a].
        """
        super().__init__()
        self.pool = pool

    def call(self, inputs, training=None, mask=None):
        x, a = inputs

        x_pool, a_pool, s = self.pool([x, a])

        return x_pool, a_pool, s
