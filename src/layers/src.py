import inspect

import tensorflow as tf
from spektral.utils.keras import (deserialize_kwarg, is_keras_kwarg,
                                  is_layer_kwarg, serialize_kwarg)
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

from src.modules.ops import sparse_connect


class SRCPool(Layer):
    r"""
    A general class for Select, Reduce, Connect graph pooling.

    This layer computes:
    $$
        \begin{align}
            & \mathcal{S} = \left\{\mathcal{S}_k\right\}_{k=1:K} = \textsc{Sel}(\mathcal{G}) \\
            & \mathcal{X}'=\left\{\textsc{Red}( \mathcal{G}, \mathcal{S}_k )\right\}_{k=1:K} \\
            & \mathcal{E}'=\left\{\textsc{Con}( \mathcal{G}, \mathcal{S}_k, \mathcal{S}_l )\right\}_{k,L=1:K} \\
        \end{align}
    $$
    Where \(\textsc{Sel}\) is a node equivariant selection function that computes
    the supernode assignments \(\mathcal{S}_k\), \(\textsc{Red}\) is a
    permutation-invariant function to reduce the supernodes into the new node
    attributes, and \(\textsc{Con}\) is a permutation-invariant connection
    function that computes the link between the pooled nodes.

    By extending this class, it is possible to create any pooling layer in the
    SRC formalism.

    **Input**

    - `X`: Tensor of shape `([batch], N, F)` representing node features;
    - `A`: Tensor or SparseTensor of shape `([batch], N, N)` representing the
    adjacency matrix;
    - `I`: (optional) Tensor of integers with shape `(N, )` representing the
    batch index;

    **Output**

    - `X_pool`: Tensor of shape `([batch], K, F)`, representing the node
    features of the output. `K` is the number of output nodes and depends on the
    specific pooling strategy;
    - `A_pool`: Tensor or SparseTensor of shape `([batch], K, K)` representing
    the adjacency matrix of the output;
    - `I_pool`: (only if I was given as input) Tensor of integers with shape
    `(K, )` representing the batch index of the output;
    - `S_pool`: (if `return_sel=True`) Tensor or SparseTensor representing the
    supernode assignments;

    **API**

    - `pool(X, A, I, **kwargs)`: pools the graph and returns the reduced node
    features and adjacency matrix. If the batch index `I` is not `None`, a
    reduced version of `I` will be returned as well.
    Any given `kwargs` will be passed as keyword arguments to `select()`,
    `reduce()` and `connect()` if any matching key is found.
    The mandatory arguments of `pool()` **must** be computed in `call()` by
    calling `self.get_inputs(inputs)`.
    - `select(X, A, I, **kwargs)`: computes supernode assignments mapping the
    nodes of the input graph to the nodes of the output.
    - `reduce(X, S, **kwargs)`: reduces the supernodes to form the nodes of the
    pooled graph.
    - `connect(A, S, **kwargs)`: connects the reduced supernodes.
    - `reduce_index(I, S, **kwargs)`: helper function to reduce the batch index
    (only called if `I` is given as input).

    When overriding any function of the API, it is possible to access the
    true number of nodes of the input (`N`) as a Tensor in the instance variable
    `self.N` (this is populated by `self.get_inputs()` at the beginning of
    `call()`).

    **Arguments**:

    - `return_sel`: if `True`, the Tensor used to represent supernode assignments
    will be returned with `X_pool`, `A_pool`, and `I_pool`;
    """

    def __init__(self, return_sel=False, **kwargs):
        # kwargs for the Layer class are handled automatically
        super().__init__(**{k: v for k, v in kwargs.items() if is_keras_kwarg(k)})
        self.supports_masking = True
        self.return_sel = return_sel

        # *_regularizer, *_constraint, *_initializer, activation, and use_bias
        # are dealt with automatically if passed to the constructor
        self.kwargs_keys = []
        for key in kwargs:
            if is_layer_kwarg(key):
                attr = kwargs[key]
                attr = deserialize_kwarg(key, attr)
                self.kwargs_keys.append(key)
                setattr(self, key, attr)

        # Signature of the SRC functions
        self.sel_signature = inspect.signature(self.select).parameters
        self.red_signature = inspect.signature(self.reduce).parameters
        self.con_signature = inspect.signature(self.connect).parameters
        self.i_red_signature = inspect.signature(self.reduce_index).parameters

        self._n_nodes = None

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        # Always start the call() method with get_inputs(inputs) to set self.N
        X, A, I = self.get_inputs(inputs)

        return self.pool(X, A, I)

    def pool(self, X, A, I, **kwargs):
        """
        This is the core method of the SRC class, which runs a full pass of
        selection, reduction and connection.
        It is usually not necessary to modify this function. Any previous/shared
        operations should be done in `call()` and their results can be passed to
        the three SRC functions via keyword arguments (any kwargs given to this
        function will be matched to the signature of `select()`, `reduce()` and
        `connect()` and propagated as input to the three functions).
        Any pooling logic should go in the SRC functions themselves.
        :param X: Tensor of shape `([batch], N, F)`;
        :param A: Tensor or SparseTensor of shape `([batch], N, N)`;
        :param I: only in single/disjoint mode, Tensor of integers with shape
        `(N, )`; otherwise, `None`;
        :param kwargs: additional keyword arguments for `select()`, `reduce()`
        and `connect()`. Any matching kwargs will be passed to each of the three
        functions.
        :return:
            - `X_pool`: Tensor of shape `([batch], K, F)`, where `K` is the
            number of output nodes and depends on the pooling strategy;
            - `A_pool`: Tensor or SparseTensor of shape `([batch], K, K)`;
            - `I_pool`: (only if I is not `None`) Tensor of integers with shape
            `(K, )`;
        """
        # Select
        sel_kwargs = self._get_kwargs(X, A, I, self.sel_signature, kwargs)
        S = self.select(X, A, I, **sel_kwargs)

        # Reduce
        red_kwargs = self._get_kwargs(X, A, I, self.red_signature, kwargs)
        X_pool = self.reduce(X, S, **red_kwargs)

        # Index reduce
        i_red_kwargs = self._get_kwargs(X, A, I, self.i_red_signature, kwargs)
        I_pool = self.reduce_index(I, S, **i_red_kwargs) if I is not None else None

        # Connect
        con_kwargs = self._get_kwargs(X, A, I, self.con_signature, kwargs)
        A_pool = self.connect(A, S, **con_kwargs)

        return self.get_outputs(X_pool, A_pool, I_pool, S)

    def select(self, X, A, I, **kwargs):
        """
        Selection function. Given the graph, computes the supernode assignments
        that will eventually be mapped to the `K` nodes of the pooled graph.
        Supernode assignments are usually represented as a dense matrix of shape
        `(N, K)` or sparse indices of shape `(K, )`.
        :param X: Tensor of shape `([batch], N, F)`;
        :param A: Tensor or SparseTensor (depending on the implementation of the
        SRC functions) of shape `([batch], N, N)`;
        :param I: Tensor of integers with shape `(N, )` or `None`;
        :param kwargs: additional keyword arguments.
        :return: Tensor representing supernode assignments.
        """
        return tf.range(tf.shape(I))

    def reduce(self, X, S, **kwargs):
        """
        Reduction function. Given a selection, reduces the supernodes to form
        the nodes of the new graph.
        :param X: Tensor of shape `([batch], N, F)`;
        :param S: Tensor representing supernode assignments, as computed by
        `select()`;
        :param kwargs: additional keyword arguments; when overriding this
        function, any keyword argument defined explicitly as `key=None` will be
        automatically filled in when calling `pool(key=value)`.
        :return: Tensor of shape `([batch], K, F)` representing the node attributes of
        the pooled graph.
        """
        return tf.gather(X, S)

    def connect(self, A, S, **kwargs):
        """
        Connection function. Given a selection, connects the nodes of the pooled
        graphs.
        :param A: Tensor or SparseTensor of shape `([batch], N, N)`;
        :param S: Tensor representing supernode assignments, as computed by
        `select()`;
        :param kwargs: additional keyword arguments; when overriding this
        function, any keyword argument defined explicitly as `key=None` will be
        automatically filled in when calling `pool(key=value)`.
        :return: Tensor or SparseTensor of shape `([batch], K, K)` representing
        the adjacency matrix of the pooled graph.
        """
        return sparse_connect(A, S, self.n_nodes)

    def reduce_index(self, I, S, **kwargs):
        """
        Helper function to reduce the batch index `I`. Given a selection,
        returns a new batch index for the pooled graph. This is only called by
        `pool()` when `I` is given as input to the layer.
        :param I: Tensor of integers with shape `(N, )`;
        :param S: Tensor representing supernode assignments, as computed by
        `select()`.
        :param kwargs: additional keyword arguments; when overriding this
        function, any keyword argument defined explicitly as `key=None` will be
        automatically filled in when calling `pool(key=value)`.
        :return: Tensor of integers of shape `(K, )`.
        """
        return tf.gather(I, S)

    @staticmethod
    def _get_kwargs(X, A, I, signature, kwargs):
        output = {}
        for k in signature.keys():
            if signature[k].default is inspect.Parameter.empty or k == "kwargs":
                pass
            elif k == "X":
                output[k] = X
            elif k == "A":
                output[k] = A
            elif k == "I":
                output[k] = I
            elif k in kwargs:
                output[k] = kwargs[k]
            else:
                raise ValueError("Missing key {} for signature {}".format(k, signature))

        return output

    def get_inputs(self, inputs):
        if len(inputs) == 3:
            X, A, I = inputs
            if K.ndim(I) == 2:
                I = I[:, 0]
            assert K.ndim(I) == 1, "I must have rank 1"
        elif len(inputs) == 2:
            X, A = inputs
            I = None
        else:
            raise ValueError(
                "Expected 2 or 3 inputs tensors (X, A, I), got {}.".format(len(inputs))
            )

        self.n_nodes = tf.shape(X)[-2]

        return X, A, I

    def get_outputs(self, X_pool, A_pool, I_pool, S):
        output = [X_pool, A_pool]
        if I_pool is not None:
            output.append(I_pool)
        if self.return_sel:
            output.append(S)

        return output

    def get_config(self):
        config = {
            "return_sel": self.return_sel,
        }
        for key in self.kwargs_keys:
            config[key] = serialize_kwarg(key, getattr(self, key))
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_mask(self, inputs, mask=None):
        # After pooling all nodes are always valid
        return None

    @property
    def n_nodes(self):
        if self._n_nodes is None:
            raise ValueError(
                "self.N has not been defined. Have you called "
                "self.get_inputs(inputs) at the beginning of "
                "call()?"
            )
        return self._n_nodes

    @n_nodes.setter
    def n_nodes(self, value):
        self._n_nodes = value

    @n_nodes.deleter
    def n_nodes(self):
        self._n_nodes = None
