import numpy as np
import scipy.sparse as sp
from spektral.utils import convolution


def GRACLUS(X_list, A_list, levels):
    """
    Graclus pooling as presented by Dhillon (2007).
    The implementation is largely taken from [this repository](https://github.com/mdeff/cnn_graph).
    :param X_list: list of node features.
    :param A_list: list of adjacency matrices.
    :param levels: levels of coarsening to compute.
    :return:
        - X_out: list of node features with the nodes re-arranged.
        - A_out: list of lists of reduced adjacency matrices. Each sublist
        contains the pooled adjacency matrices for each level (including original
        graph).
        - S_out: list of lists of selection matrices mapping from one level to
        the next.
    """
    X_out = []
    A_out = []
    S_out = []
    for i in range(len(A_list)):
        A = sp.csr_matrix(A_list[i])
        X = X_list[i]
        X_pool, A_pool, S_pool = _pool(X, A, levels)
        X_out.append(X_pool)
        A_out.append(A_pool)
        S_out.append(S_pool)

    return X_out, A_out, S_out


def preprocess(X, A):
    degrees = A.sum(0)
    if isinstance(degrees, np.matrix):
        degrees = degrees.A[0, :]
    idx = np.argwhere(degrees == 0)[:, 0]
    A[idx, idx] = 1
    return X, A


def _pool(X, A, levels):
    """
    Pool one graph and re-arrange its features to perform online reduction with
    1D max-pooling.
    :param X: node features of shape (N, F);
    :param A: adjacency matrix of shape (N, N);
    :param levels: coarsening levels to return (each level almost halves the
    nodes).
    :return:
        - X_out: re-arranged node features of shape (N, F);
        - A_out: list of pooled adjacency matrices for each coarsening level;
        - S: list of selection matrix for each coarsening level.
    """
    A_out, permutations = _select_and_connect(A, levels)

    # Get selection matrices explicitly in case they are needed
    S_out = _selection_matrices(A_out)

    # Re-arrange nodes to perform reduction with regular 1D max-pooling
    X_out = _permute_data(X, permutations)

    return X_out, A_out, S_out


def _select_and_connect(A, levels):
    """
    Compute selection and connection in a single step.
    :param A: adjacency matrix of shape (N, N);
    :param levels: coarsening levels to return (each level almost halves the
    nodes).
    :return:
        - A_pool: list of pooled adjacency matrices for each coarsening level;
        - permutations: permutation of the indices to re-arrange the nodes.
    """
    A_pool, permutations = _coarsen(A, levels=max(levels) + 1)
    A_pool = [A_pool[i] for i in levels]  # Keep only the right graphs
    A_pool = [convolution.normalized_adjacency(a_c) for a_c in A_pool]

    return A_pool, permutations


def _coarsen(A, levels, self_loops=True):
    """
    Compute pooled adjacency matrices.
    :param A: adjacency matrix of shape (N, N);
    :param levels: int, max of the coarsening levels to compute.
    :param self_loops: add self-loops to A.
    :return:
        - graphs: list of pooled adjacency matrices;
        - permutations: permutation of the indices to re-arrange the nodes.
    """
    graphs, parents = _METIS(A, levels)
    permutations = _compute_permutation(parents)

    for i, A in enumerate(graphs):
        if not self_loops:
            A = A.tocoo()
            A.setdiag(0)
        if i < levels:
            A = _permute_adjacency(A, permutations[i])
        A = A.tocsr()
        A.eliminate_zeros()
        graphs[i] = A

    return graphs, permutations[0] if levels > 0 else None


def _selection_matrices(graphs):
    """
    Compute selection matrices from the pooled adjacency matrices.
    :param graphs: list of pooled adjacency matrices at different levels.
    :return: list of selection matrices.
    """
    if len(graphs) < 2:
        raise ValueError("Need at least two levels.")

    S_list = []
    for i in range(len(graphs) - 1):
        N = graphs[i].shape[0]
        N_ = graphs[i + 1].shape[0]
        offset = N // N_
        I = np.eye(N, dtype=np.float32)
        S_list.append(I[1::offset, :])
    S_list.append(np.eye(N_, dtype=np.float32))

    # S must have shape (N, K)
    S_list = [s.T for s in S_list]

    return S_list


def _METIS(A, levels):
    """
    Coarsens a graph multiple times using the METIS algorithm.

    :param A: adjacency matrix of shape (N, N);
    :param levels: int, max of the coarsening levels to compute.
    :return:
        - graphs: list of pooled adjacency matrices at different levels.
        - parents: list of indices indicating the parents of each node in the
        previous graph.
    """
    N, _ = A.shape
    rid = np.random.permutation(range(N))
    parents = []
    degree = A.sum(axis=0) - A.diagonal()
    graphs = [A]

    for _ in range(levels):
        weights = degree  # Graclus weights
        weights = np.array(weights).squeeze()

        # Pair vertices and construct root vector
        idx_row, idx_col, val = sp.find(A)
        perm = np.argsort(idx_row)
        rr = idx_row[perm]
        cc = idx_col[perm]
        vv = val[perm]
        cluster_id = _METIS_one_level(rr, cc, vv, rid, weights)
        parents.append(cluster_id)

        # Compute new graph
        n_rr = cluster_id[rr]
        n_cc = cluster_id[cc]
        n_vv = vv
        N_new = cluster_id.max() + 1
        A = sp.csr_matrix((n_vv, (n_rr, n_cc)), shape=(N_new, N_new))
        A.eliminate_zeros()
        graphs.append(A)

        degree = A.sum(axis=0)

        # Choose the order in which vertices will be visited at the next iteration
        ss = np.array(A.sum(axis=0)).squeeze()
        rid = np.argsort(ss)

    return graphs, parents


def _METIS_one_level(rr, cc, vv, rid, weights):
    # Coarsen a graph given by rr,cc,vv
    # rr is assumed to be ordered
    nnz = rr.shape[0]
    N = rr[nnz - 1] + 1

    marked = np.zeros(N, np.bool)
    row_start = np.zeros(N, np.int32)
    row_length = np.zeros(N, np.int32)
    cluster_id = np.zeros(N, np.int32)

    old_val = rr[0]
    count = 0
    cluster_count = 0

    for ii in range(nnz):
        row_length[count] = row_length[count] + 1
        if rr[ii] > old_val:
            old_val = rr[ii]
            row_start[count + 1] = ii
            count = count + 1

    for ii in range(N):
        tid = rid[ii]
        if not marked[tid]:
            wmax = 0.0
            rs = row_start[tid]
            marked[tid] = True
            best_neighbor = -1
            for jj in range(row_length[tid]):
                nid = cc[rs + jj]
                if marked[nid]:
                    tval = 0.0
                else:
                    tval = vv[rs + jj] * (1.0 / weights[tid] + 1.0 / weights[nid])
                if tval > wmax:
                    wmax = tval
                    best_neighbor = nid
            cluster_id[tid] = cluster_count
            if best_neighbor > -1:
                cluster_id[best_neighbor] = cluster_count
                marked[best_neighbor] = True
            cluster_count += 1

    return cluster_id


def _compute_permutation(parents):
    # Order of last layer is random (chosen by the clustering algorithm).
    indices = []
    if len(parents) > 0:
        N_last = max(parents[-1]) + 1
        indices.append(list(range(N_last)))

    for parent in parents[::-1]:
        # Fake nodes go after real ones.
        pool_singeltons = len(parent)

        indices_layer = []
        for i in indices[-1]:
            indices_node = list(np.where(parent == i)[0])
            assert 0 <= len(indices_node) <= 2

            # Add a node to go with a singelton
            if len(indices_node) is 1:
                indices_node.append(pool_singeltons)
                pool_singeltons += 1
            elif len(indices_node) is 0:
                # Add two nodes as children of a singelton in the parent.
                indices_node.append(pool_singeltons + 0)
                indices_node.append(pool_singeltons + 1)
                pool_singeltons += 2
            indices_layer.extend(indices_node)
        indices.append(indices_layer)

    # Sanity checks
    for i, indices_layer in enumerate(indices):
        N = N_last * 2 ** i
        # Reduction by 2 at each layer (binary tree).
        assert len(indices[0] == N)
        # The new ordering does not omit an index.
        assert sorted(indices_layer) == list(range(N))

    return indices[::-1]


def _permute_data(X, indices):
    X = X.T
    if indices is None:
        return X.T

    N, F = X.shape
    F_new = len(indices)
    assert F_new >= F
    X_new = np.empty((N, F_new))
    for i, j in enumerate(indices):
        if j < F:
            X_new[:, i] = X[:, j]
        else:
            # Fake vertex because of singeltons.
            X_new[:, i] = np.zeros(N)
    return X_new.T


def _permute_adjacency(A, indices):
    if indices is None:
        return A

    N = A.shape[0]
    N_new = len(indices)
    A = A.tocoo()

    if N_new > N:
        rows = sp.coo_matrix((N_new - N, N), dtype=np.float32)
        cols = sp.coo_matrix((N_new, N_new - N), dtype=np.float32)
        A = sp.vstack([A, rows])
        A = sp.hstack([A, cols])

    # Permute the rows and the columns.
    perm = np.argsort(indices)
    A.row = np.array(perm)[A.row]
    A.col = np.array(perm)[A.col]

    return A
