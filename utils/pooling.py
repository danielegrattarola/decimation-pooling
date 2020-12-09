import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import non_negative_factorization
from spektral.utils import convolution


def pooling_matrices(A_list, coarsening_levels, pool, sparsify=True, sparse_thresh=1e-2):
    """
    Creates all the necessary matrices for decimation pooling and average
    pooling.
    :param A_list: a list of adjacency matrices
    :param coarsening_levels: a list of coarsening levels for pooling
    :param sparsify: if True, applies sparsification
    :return:
        D_out: list of lists with decimation matrices for each graph
        A_out: list of Laplacians pyramids for each graph
        num_nodes_list: list with the number of nodes for each graph
        partition_list: list of lists, each list has N repeating integers where
        N is the size of each graph. Used as second argument for tf.segment_sum.
    """
    A_out = []
    D_out = []

    if pool == 'decim':
        for a_ in A_list:
            # Compute pyramid of coarsened graphs
            adjacencies, indices = reduction(a_,
                                             levels=max(coarsening_levels) + 1,
                                             sparsify=sparsify,
                                             sparse_thresh=sparse_thresh)

            # Get pooling matrices
            dm, adj = decimation_matrices(levels=coarsening_levels,
                                          A_list=adjacencies,
                                          indices=indices)
            D_out.append(dm)
            A_out.append(adj)

    elif pool == 'graclus':
        perm_list = []
        for a in A_list:
            a = sp.csr_matrix(a)  # pooling.coarsen wants sparse matrices
            graphs, perm = coarsen(a, levels=max(coarsening_levels) + 1)
            graphs = [graphs[i] for i in coarsening_levels]  # Keep only the right graphs
            dm = graclus_pooling_matrices(graphs)
            D_out.append(dm)
            A_out.append(graphs)
            perm_list.append(perm)

    elif pool == 'nmf':
        for a in A_list:
            dm, adj = nmf_pooling(A=a, levels=coarsening_levels)
            D_out.append(dm)
            A_out.append(adj)
    else:
        raise ValueError("pool must be 'decim', 'graclus' or 'nmf'")

    if pool == 'graclus':
        return A_out, D_out, perm_list
    else:
        return A_out, D_out


################################################################################
# METHODS FOR GRAPH COARSENING WITH DECIMATION MATRICES (Kron reduction)
################################################################################
def reduction(A, levels, sparsify=True, sparse_thresh=1e-3):
    """
    Decimates nodes and performs graph reduction.
    INPUTS
        A: adjacency matrix
        levels: number of decimations/ graph reductions to do
        sparsify: if True, applies sparsification
        sparse_thresh: if sparsify is True, drop edges lower than sparse_thresh
    OUTPUT
        adjacencies: list of reduced graph
        indexes: indexes of the nodes to keep after each decimation
    """
    adjacencies = []
    indexes = []

    if isinstance(A, np.matrix):
        A = sp.csc_matrix(A)
    elif hasattr(A, 'tocsc'):
        A = A.tocsc()
    else:
        # print('WARNING: got dense adjacency, attempting conversion')
        A = sp.csc_matrix(A)

    A_new = A

    for i in range(levels):

        # compute Laplacian and symmetric Laplacian
        L = convolution.laplacian(A_new)
        Ls = convolution.normalized_laplacian(A_new)

        # Compute spectral cut
        if L.shape == (1, 1):
            # No need for pooling
            idx_pos = np.array([0])
        else:
            try:
                V = sp.linalg.eigsh(Ls, k=1, which='LM', v0=np.ones(A_new.shape[0]))[1][:, 0]
            except Exception:  # ArpackNoConvergence:
                # Random split if eigendecomposition is not possible
                print('Level %d --Eigendecomposition is not possible, splitting randomly instead' % i)
                V = np.random.choice([-1, 1], size=(A_new.shape[0],))

            idx_pos = np.nonzero(V >= 0)[0]
            idx_neg = np.nonzero(V < 0)[0]

            # Evaluate the size of the cut 
            z = np.ones((A_new.shape[0], 1))  # partition vector
            z[idx_neg] = -1
            cut_size = eval_cut(A_new, L, z)

            # If the cut is smaller than 0.5, return a random cut
            if cut_size < 0.5:
                print("Level %d -- Spectral cut lower than 0.5 (%.2f): returning random cut" % (i, cut_size))
                V = np.random.choice([-1, 1], size=(Ls.shape[0],))
                idx_pos = np.nonzero(V >= 0)[0]
                idx_neg = np.nonzero(V < 0)[0]

        # Link reconstruction with Kron reduction
        if len(idx_pos) <= 1:
            # No need to compute Kron reduction with 0 or 1 node
            Lnew = sp.csc_matrix(-np.ones((1, 1)))  # L = -1
            idx_pos = np.array([0])  # make sure there is at least 1 index
        else:
            # Kron reduction
            L_red = L[np.ix_(idx_pos, idx_pos)]
            L_in_out = L[np.ix_(idx_pos, idx_neg)]
            L_out_in = L[np.ix_(idx_neg, idx_pos)].tocsc()
            L_comp = L[np.ix_(idx_neg, idx_neg)].tocsc()
            try:
                Lnew = L_red - L_in_out.dot(sp.linalg.spsolve(L_comp, L_out_in))
            except RuntimeError:
                # If L_comp is exactly singular, damp the inversion with
                # Marquardt-Levenberg coefficient ml_c
                ml_c = sp.csc_matrix(sp.eye(L_comp.shape[0]) * 1e-6)
                Lnew = L_red - L_in_out.dot(sp.linalg.spsolve(ml_c + L_comp, L_out_in))

            # Make the laplacian symmetric if it is almost symmetric
            if np.abs(Lnew - Lnew.T).sum() < np.spacing(1) * np.abs(Lnew).sum():
                Lnew = (Lnew + Lnew.T) / 2.

        # Update output lists
        adjacencies.append(A_new)
        indexes.append(idx_pos)

        A_new = -Lnew
        A_new.setdiag(0)
        A_new.eliminate_zeros()

    # Apply sparsification to the list of coarsened adjacencies
    if sparsify:
        sparse_adjacencies = [adjacencies[0]]
        for a_ in adjacencies[1:]:
            a_ = a_.multiply(np.abs(a_) > sparse_thresh)
            sparse_adjacencies.append(a_)
        adjacencies = sparse_adjacencies

    return adjacencies, indexes


def eval_cut(A, L, z):
    """
    Computes the normalized size of a cut in [0,1]
    """
    cut = z.T.dot(L.dot(z))
    cut /= 2 * np.sum(A)
    return cut


def sparsify(A_list, sparse_thresh=1e-3):
    """
    Sparsify the adjacency matrices.
    Useful for debugging and for checking different sparsification thresholds
    """
    sparse_adjacencies = []
    for a_ in A_list:
        a_ = a_.multiply(np.abs(a_) > sparse_thresh)
        sparse_adjacencies.append(a_)
    return sparse_adjacencies


def decimation_matrices(levels, A_list, indices):
    """
    INPUT
        levels: list of integes indicating the graphs of the pyramid to keep
        A_list: pyramid of adjacency matrices
        indices: list of indices of the nodes to keep in each level
    OUTPUT
        - S_list: list of decimation matrices
        - A_keep_list: list of adjacencies to be kept (the ones in levels) 
    """
    S_list = []
    A_keep_list = []
    S_prev = sp.eye(A_list[0].shape[0], dtype=np.float32)
    for i in range(max(levels) + 1):
        nodes_i = A_list[i].shape[0]
        I = sp.eye(nodes_i, dtype=np.float32)
        S_i = I.tocsr()[indices[i], :]
        S_prev = S_i.dot(S_prev)
        if i in levels:
            A_keep_list.append(A_list[i])
        if i + 1 == max(levels) + 1:
            S_list.append(S_prev)
        elif i + 1 in levels:
            S_list.append(S_prev)
            S_prev = sp.eye(len(indices[i]), dtype=np.float32)

    return S_list, A_keep_list


################################################################################
# METHODS FOR GRAPH COARSENING WITH NON-NEGATIVE MATRIX FACTORIZATION
################################################################################
def nmf_pooling(A, levels, binarize=False):
    S_list = []
    A_list = []
    S_prev = sp.eye(A.shape[0], dtype=np.float32)
    for i in range(max(levels) + 1):
        A = sp.csr_matrix(A, dtype=np.float32)
        if i in levels:
            A_list.append(A)
        n_nodes = A.shape[0]
        n_comp = np.maximum(n_nodes // 2, 2)
        _, H, _ = non_negative_factorization(A, n_components=n_comp, init='random', random_state=0, max_iter=10)
        H = sp.csr_matrix(H, dtype=np.float32)
        A = (H.dot(A)).dot(H.T)

        # binarize H (hard cluster assignment)
        if binarize:
            H = H.toarray()
            S_i = np.zeros_like(H)
            S_i[np.arange(len(H)), H.argmax(1)] = 1
            S_i = sp.csr_matrix(S_i, dtype=np.float32)
        else:
            S_i = H

        # save the right pooling matrices
        S_prev = S_i.dot(S_prev)
        if i + 1 == max(levels) + 1:
            S_list.append(S_prev)
        elif i + 1 in levels:
            S_list.append(S_prev)
            S_prev = sp.eye(A.shape[0], dtype=np.float32)

    return S_list, A_list


################################################################################
# METHODS FOR GRAPH COARSENING WITH GRACLUS (M. Defferrard's code)
################################################################################
def graclus_pooling_matrices(graphs):
    S = []
    for i in range(len(graphs) - 1):
        M_i = graphs[i].shape[0]
        M_ii = graphs[i + 1].shape[0]
        offset = M_i // M_ii
        I = np.eye(M_i, dtype=np.float32)
        S.append(I[1::offset, :])

    # This last matrix will not be used
    # is here just for having the same number of pooling mats and coarsened adjs
    S.append(np.eye(M_ii, dtype=np.float32))

    return S


def coarsen(A, levels, self_connections=True):
    """
    Coarsen a graph, represented by its adjacency matrix A, at multiple levels.
    """
    graphs, parents = metis(A, levels)
    perms = compute_perm(parents)

    for i, A in enumerate(graphs):
        if not self_connections:
            A = A.tocoo()
            A.setdiag(0)
        if i < levels:
            A = perm_adjacency(A, perms[i])
        A = A.tocsr()
        A.eliminate_zeros()
        graphs[i] = A

    return graphs, perms[0] if levels > 0 else None


def metis(W, levels, rid=None):
    """
    Coarsen a graph multiple times using the METIS algorithm.

    INPUT
    W: symmetric sparse weight (adjacency) matrix
    levels: the number of coarsened graphs

    OUTPUT
    graph[0]: original graph of size N_1
    graph[2]: coarser graph of size N_2 < N_1
    graph[levels]: coarsest graph of Size N_levels < ... < N_2 < N_1
    parents[i] is a vector of size N_i with entries ranging from 1 to N_{i+1}
        which indicate the parents in the coarser graph[i+1]
    nd_sz{i} is a vector of size N_i that contains the size of the supernode in the graph{i}

    NOTE
    if "graph" is a list of length k, then "parents" will be a list of length k-1
    """
    N, _ = W.shape
    if rid is None:
        rid = np.random.permutation(range(N))
    parents = []
    degree = W.sum(axis=0) - W.diagonal()
    graphs = [W]

    for _ in range(levels):
        # CHOOSE THE WEIGHTS FOR THE PAIRING
        weights = degree  # Graclus weights
        weights = np.array(weights).squeeze()

        # PAIR THE VERTICES AND CONSTRUCT THE ROOT VECTOR
        idx_row, idx_col, val = sp.find(W)
        perm = np.argsort(idx_row)
        rr = idx_row[perm]
        cc = idx_col[perm]
        vv = val[perm]
        cluster_id = metis_one_level(rr, cc, vv, rid, weights)  # rr is ordered
        parents.append(cluster_id)

        # COMPUTE THE EDGES WEIGHTS FOR THE NEW GRAPH
        nrr = cluster_id[rr]
        ncc = cluster_id[cc]
        nvv = vv
        Nnew = cluster_id.max() + 1
        # CSR is more appropriate: row,val pairs appear multiple times
        W = sp.csr_matrix((nvv, (nrr, ncc)), shape=(Nnew, Nnew))
        W.eliminate_zeros()
        # Add new graph to the list of all coarsened graphs
        graphs.append(W)

        # COMPUTE THE DEGREE (OMIT OR NOT SELF LOOPS)
        degree = W.sum(axis=0)

        # CHOOSE THE ORDER IN WHICH VERTICES WILL BE VISTED AT THE NEXT PASS
        ss = np.array(W.sum(axis=0)).squeeze()
        rid = np.argsort(ss)

    return graphs, parents


def metis_one_level(rr, cc, vv, rid, weights):
    # Coarsen a graph given by rr,cc,vv.  rr is assumed to be ordered
    nnz = rr.shape[0]
    N = rr[nnz - 1] + 1

    marked = np.zeros(N, np.bool)
    rowstart = np.zeros(N, np.int32)
    rowlength = np.zeros(N, np.int32)
    cluster_id = np.zeros(N, np.int32)

    oldval = rr[0]
    count = 0
    clustercount = 0

    for ii in range(nnz):
        rowlength[count] = rowlength[count] + 1
        if rr[ii] > oldval:
            oldval = rr[ii]
            rowstart[count + 1] = ii
            count = count + 1

    for ii in range(N):
        tid = rid[ii]
        if not marked[tid]:
            wmax = 0.0
            rs = rowstart[tid]
            marked[tid] = True
            bestneighbor = -1
            for jj in range(rowlength[tid]):
                nid = cc[rs + jj]
                if marked[nid]:
                    tval = 0.0
                else:
                    tval = vv[rs + jj] * (1.0 / weights[tid] + 1.0 / weights[nid])
                if tval > wmax:
                    wmax = tval
                    bestneighbor = nid
            cluster_id[tid] = clustercount
            if bestneighbor > -1:
                cluster_id[bestneighbor] = clustercount
                marked[bestneighbor] = True
            clustercount += 1

    return cluster_id


def compute_perm(parents):
    """
    Return a list of indices to reorder the adjacency and data matrices so
    that the union of two neighbors from layer to layer forms a binary tree.
    """
    # Order of last layer is random (chosen by the clustering algorithm).
    indices = []
    if len(parents) > 0:
        M_last = max(parents[-1]) + 1
        indices.append(list(range(M_last)))

    for parent in parents[::-1]:
        # Fake nodes go after real ones.
        pool_singeltons = len(parent)

        indices_layer = []
        for i in indices[-1]:
            indices_node = list(np.where(parent == i)[0])
            assert 0 <= len(indices_node) <= 2

            # Add a node to go with a singelton.
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

    # Sanity checks.
    for i, indices_layer in enumerate(indices):
        M = M_last * 2 ** i
        # Reduction by 2 at each layer (binary tree).
        assert len(indices[0] == M)
        # The new ordering does not omit an index.
        assert sorted(indices_layer) == list(range(M))

    return indices[::-1]


def perm_data(x, indices):
    """
    Permute data matrix, i.e. exchange node ids,
    so that binary unions form the clustering tree.
    """
    if indices is None:
        return x

    N, M = x.shape
    Mnew = len(indices)
    assert Mnew >= M
    xnew = np.empty((N, Mnew))
    for i, j in enumerate(indices):
        # Existing vertex, i.e. real data.
        if j < M:
            xnew[:, i] = x[:, j]
        else:
            # Fake vertex because of singeltons.
            # They will stay 0 so that max pooling chooses the singelton. Or -infty ?
            xnew[:, i] = np.zeros(N)
    return xnew


def perm_adjacency(A, indices):
    """
    Permute adjacency matrix, i.e. exchange node ids,
    so that binary unions form the clustering tree.
    """
    if indices is None:
        return A

    M, M = A.shape
    Mnew = len(indices)
    assert Mnew >= M
    A = A.tocoo()

    if Mnew > M:
        # Add Mnew - M isolated vertices.
        rows = sp.coo_matrix((Mnew - M, M), dtype=np.float32)
        cols = sp.coo_matrix((Mnew, Mnew - M), dtype=np.float32)
        A = sp.vstack([A, rows])
        A = sp.hstack([A, cols])

    # Permute the rows and the columns.
    perm = np.argsort(indices)
    A.row = np.array(perm)[A.row]
    A.col = np.array(perm)[A.col]

    assert type(A) is sp.coo.coo_matrix
    return A
