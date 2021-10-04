import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import non_negative_factorization
from spektral.utils import convolution


def NMF(A_list, p):
    r"""
    Non-negative Matrix Factorization pooling by [Bacciu and Di Sotto
    (2019)](https://arxiv.org/abs/1909.03287).
    Pools a graph by computing the non-negative matrix factorization
    $$
        \mathbf{A} = \mathbf{W}\mathbf{H}
    $$
    and using \(\mathbf{H}^\top\) as a dense clustering matrix.
    Pooled graphs are computed as:
    $$
        \mathbf{A}^{pool} = \mathbf{S}^\top \mathbf{A} \mathbf{S}; \;\;
        \mathbf{X}^{pool} = \mathbf{S}^\top \mathbf{X}
    $$

    :param A_list: list of adjacency matrices.
    :param p: float, ratio of nodes to keep.
    :return:
        - A_out: list of pooled adjacency matrices;
        - S_out: list of selection matrices
    """
    A_out = []
    S_out = []
    for A in A_list:
        A_pool, S = _pool(A, p)
        A_out.append(A_pool)
        S_out.append(S)

    return A_out, S_out


def preprocess(X, A):
    return X, convolution.gcn_filter(A)


def _pool(A, p):
    """
    Pool one graph.
    :param A: adjacency matrix of shape (N, N);
    :param p: float (0 < p < 1), ratio of nodes to keep;
    :return:
        - A_out: pooled adjacency matrix of shape (K, K), where K = ceil(p * N);
        - S: selection matrix of shape (N, K).
    """
    S = _select(A, p)
    A_out = _connect(A, S)

    return A_out, S


def _select(A, p):
    """
    Selection function.
    :param A: adjacency matrix of shape (N, N);
    :param p: float (0 < p < 1), ratio of nodes to keep;
    :return: selection matrix of shape (N, K).
    """
    N = A.shape[-1]
    K = int(np.ceil(p * N))
    A = sp.csr_matrix(A, dtype=np.float32)
    _, H, _ = non_negative_factorization(A, n_components=K, init="random", max_iter=400)
    S = sp.csr_matrix(H.T)

    return S


def _connect(A, S):
    """
    Connection function.
    :param A: adjacency matrix of shape (N, N);
    :param S: selection matrix of shape (N, K).
    :return: pooled adjacency matrix of shape (K, K).
    """
    return S.T.dot(A).dot(S)
