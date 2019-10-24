import numpy as np
from scipy import sparse as sp
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import pooling


def preprocess(A_list, X_list, coarsening_levels, pool):
    mats = pooling.pooling_matrices(A_list, coarsening_levels, pool)
    if pool == 'graclus':
        A_list, D_list, perm_list = mats
        # Permute the nodes to be consistent with clustering indices
        for i, perm_i in enumerate(perm_list):
            X_list[i] = pooling.perm_data(X_list[i].T, perm_i).T
    elif pool == 'decim' or pool == 'nmf':
        A_list, D_list = mats
    else:
        raise ValueError("pool must be 'decimation', 'graclus' or 'nmf'")

    return A_list, X_list, D_list


def node_feat_norm(feat_list, norm='ohe'):
    """
    Apply one-hot encoding or z-score to a list of node features
    """
    if norm == 'ohe':
        fnorm = OneHotEncoder(sparse=False, categories='auto')
    elif norm == 'zscore':
        fnorm = StandardScaler()
    else:
        raise ValueError('Possible feat_norm: ohe, zscore')
    fnorm.fit(np.vstack(feat_list))
    feat_list = [fnorm.transform(feat_.astype(np.float32)) for feat_ in feat_list]
    return feat_list


def create_batch(A_list, X_list, D_list):
    D_out, A_out = list(zip(*D_list)), list(zip(*A_list))
    for i, _ in enumerate(D_out):
        D_out[i] = sp.block_diag(list(D_out[i]))
        A_out[i] = sp.block_diag(list(A_out[i]))
    X_out = np.vstack(X_list)
    n_nodes = np.array([_[0].shape[0] for _ in A_list])
    I_out = np.repeat(np.arange(len(n_nodes)), n_nodes)
    return A_out, X_out, D_out, I_out
