# Select, Reduce, Connect

![](images/src.png)

This repository contains the code used for the experiments of:

**"Understanding Pooling in Graph Neural Networks"**  

# Setup

Install TensorFlow and other dependencies: 

```bash
pip install -r requirements.txt
```

# Running experiments

Experiments are found in the following folders: 

- `autoencoder/`
- `spectral_similarity/`
- `graph_classification/`

Each folder has a bash script called `run_all.sh` that will reproduce the results reported in the paper. 

To generate the plots and tables that we included in the paper, you can use the `plots.py`, `plots_datasets.py`, or `tables.py` found in the folders.

To run experiments for an individual pooling operator, you can use the `run_[OPERATOR NAME].py` scripts in each folder. 

The pooling operators that we used for the experiments are found in `layers/` (trainable) and `modules/` (non-trainable).
The GNN architectures used in the experiments are found in `models/`. 

# The SRCPool class

The core of this repository is the `SRCPool` class that implements a general 
interface to create SRC pooling layers with the Keras API.

Our implementation of MinCutPool, DiffPool, LaPool, Top-K, and SAGPool using the
`SRCPool` class can be found in `src/layers`.

In general, SRC layers compute:

![](images/src.svg)

Where ![](https://latex.codecogs.com/svg.latex?\textsc{Sel}) is a node 
equivariant selection function that computes the supernode assignments 
![](https://latex.codecogs.com/svg.latex?\mathcal{S}_k), 
![](https://latex.codecogs.com/svg.latex?\textsc{Red}) is a
permutation-invariant function to reduce the supernodes into the new node
attributes, and ![](https://latex.codecogs.com/svg.latex?\textsc{Con})
is a permutation-invariant connection function that computes the links between 
the pooled nodes.

By extending this class, it is possible to create any pooling layer in the
SRC framework.

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
- `I_pool`: (only if `I` was given as input) Tensor of integers with shape
`(K, )` representing the batch index of the output;
- `S_pool`: (if `return_sel=True`) Tensor or SparseTensor representing the
supernode assignments;

**API**

- `pool(X, A, I, **kwargs)`: pools the graph and returns the reduced node
features and adjacency matrix. If the batch index `I` is not `None`, a
reduced version of `I` will be returned as well.
Any given `kwargs` will be passed as keyword arguments to `select()`,
`reduce()` and `connect()` if any matching key is found.
The mandatory arguments of `pool()` (`X`, `A`, and `I`) **must** be computed in 
`call()` by calling `self.get_inputs(inputs)`.
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
