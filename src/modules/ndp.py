import warnings

import numpy as np
import scipy.sparse as sp
from spektral.utils import convolution


def NDP(A_list, level, sparsify=1e-2):
    """
    Node Decimation Pooling as presented by
    [Bianchi et al.](https://arxiv.org/abs/1910.11436).

    :param A_list: list of adjacency matrices.
    :param level: int, level of pooling to apply to each graph (each level ~ halves nodes)
    :param sparsify: float, sparsification threshold for the output adjacency
    matrices (all edges < sparsify will be set to 0).
    :return:
        - A_out: list of pooled adjacency matrices;
        - S_out: list of selection matrices;
    """
    A_out = []
    S_out = []
    for a in A_list:
        A_pool, S = _pool(a, level, sparsify=sparsify)
        A_out.append(A_pool)
        S_out.append(S)

    return A_out, S_out


def preprocess(X, A):
    return X, A


def _pool(A, level, sparsify=1e-2):
    """
    Pool one graph.
    :param A: np.array or scipy.sparse matrix, adjacency matrix of shape (N, N);
    :param level: int, level of pooling to apply to the graph (each level almost
    halves nodes)
    :param sparsify: float, sparsification threshold (all edges < sparsify will
    be set to 0).
    :return:
        - A_out: reduced adjacency matrix of shape (K, K), where K ~= N / (2 * level)
        - S: selection matrix of shape (N, K)
    """
    masks = []
    A_out = A
    for i in range(level):
        A_out, mask = _select_and_connect(A_out)
        masks.append(mask)
    S = _masks_to_matrix(masks)

    # Sparsification
    A_out = A_out.tocsr()
    A_out = A_out.multiply(np.abs(A_out) > sparsify)

    return A_out, S


def _select_and_connect(A):
    """
    Compute selection and connection in a single step.
    :param A: adjacency matrix of shape (N, N);
    :return:
        - A_pool: pooled adjacency matrix of shape (K, K);
        - mask: boolean selection mask of shape (N, ).
    """
    A = sp.csc_matrix(A)
    L = convolution.laplacian(A)
    Ls = convolution.normalized_laplacian(A)

    # Compute spectral cut
    if L.shape == (1, 1):
        # No need for pooling
        idx_pos = np.zeros(1, dtype=int)
        V = np.ones(1)
    else:
        try:
            V = sp.linalg.eigsh(Ls, k=1, which="LM", v0=np.ones(A.shape[0]))[1][:, 0]
        except Exception:
            # Random split if eigen-decomposition is not possible
            print("Eigen-decomposition failed. Splitting nodes randomly instead.")
            V = np.random.choice([-1, 1], size=(A.shape[0],))

        idx_pos = np.nonzero(V >= 0)[0]
        idx_neg = np.nonzero(V < 0)[0]

        # Fallback to random cut if spectral cut is smaller than 0.5
        # Evaluate the size of the cut
        z = np.ones((A.shape[0], 1))  # partition vector
        z[idx_neg] = -1
        cut_size = eval_cut(A, L, z)
        if cut_size < 0.5:
            print(
                "Spectral cut lower than 0.5 {}: returning random cut".format(cut_size)
            )
            V = np.random.choice([-1, 1], size=(Ls.shape[0],))
            idx_pos = np.nonzero(V >= 0)[0]
            idx_neg = np.nonzero(V < 0)[0]

    if len(idx_pos) <= 1:
        # This happens if the graph cannot be split in half enough times
        # In this case, we skip pooling and return the identity
        L_pool = sp.csc_matrix(np.zeros((1, 1)))
        idx_pos = np.zeros(1, dtype=int)
    else:
        # Kron reduction
        L_pool = _kron_reduction(L, idx_pos, idx_neg)

    # Make the Laplacian symmetric if it is almost symmetric
    if np.abs(L_pool - L_pool.T).sum() < np.spacing(1) * np.abs(L_pool).sum():
        L_pool = (L_pool + L_pool.T) / 2.0

    A_pool = -L_pool
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        A_pool.setdiag(0)
        A_pool.eliminate_zeros()

    mask = np.zeros_like(V, dtype=bool)
    mask[idx_pos] = True
    return A_pool, mask


def _kron_reduction(L, idx_pos, idx_neg):
    """
    Compute the Kron reduction of the given Laplacian.
    :param L: Laplacian matrix of shape (N, N);
    :param idx_pos: array of indices for the "positive" nodes;
    :param idx_neg: array of indices for the "negative" nodes;
    :return:
        - Reduced Laplacian of shape (K, K)
    """
    L_red = L[np.ix_(idx_pos, idx_pos)]
    L_in_out = L[np.ix_(idx_pos, idx_neg)]
    L_out_in = L[np.ix_(idx_neg, idx_pos)].tocsc()
    L_comp = L[np.ix_(idx_neg, idx_neg)].tocsc()
    try:
        L_pool = L_red - L_in_out.dot(sp.linalg.spsolve(L_comp, L_out_in))
    except RuntimeError:
        # If L_comp is exactly singular, damp the inversion with
        # Marquardt-Levenberg coefficient ml_c.
        ml_c = sp.csc_matrix(sp.eye(L_comp.shape[0]) * 1e-6)
        L_pool = L_red - L_in_out.dot(sp.linalg.spsolve(ml_c + L_comp, L_out_in))

    return L_pool


def eval_cut(A, L, z):
    """
    Computes the normalized size of a cut in [0,1]
    """
    cut = z.T.dot(L.dot(z))
    cut /= 2 * np.sum(A)
    return cut


def _masks_to_matrix(masks):
    """
    Converts a list of boolean selection masks (one for each level) to a sparse
    selection matrix.
    :param masks: list of boolean selection masks, one for each coarsening level.
    :return: selection matrix of shape (N, K)
    """
    S_ = sp.eye(masks[0].shape[0], dtype=np.float32).tocsr()
    S_ = S_[:, masks[0]]
    for i in range(1, len(masks)):
        S_next = sp.eye(masks[i].shape[0], dtype=np.float32).tocsr()
        S_next = S_next[:, masks[i]]
        S_ = S_.dot(S_next)

    return S_
