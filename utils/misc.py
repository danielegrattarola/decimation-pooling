import itertools
import os
import re
from collections import OrderedDict
from urllib.request import Request, urlopen

import gensim
import numpy as np
import sklearn as sk
import sklearn.datasets
from scipy import sparse as sp
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import pooling


################################################################################
# UTILS FOR GRAPH CLASSIFICATION
################################################################################
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


###################################################################
# GRAPH AND LAPLACIAN METHODS
###################################################################
def eval_bw(X, Y):
    """
    Compute heuristically the bandwidth using class information
    Returns (d^2)/9, with d minimum distance of elements in X with different
    class Y. A small value is added to avoid returning bw=0.
    """
    classes = np.unique(Y)
    min_dist = np.inf
    for i in range(classes.shape[0] - 1):
        c_i = classes[i]
        X_i = X[Y == c_i, :]

        for j in range(i + 1, classes.shape[0]):
            c_j = classes[j]
            X_j = X[Y == c_j, :]
            dist_ij = np.min(cdist(X_i, X_j, metric='sqeuclidean'))
            if dist_ij < min_dist:
                min_dist = dist_ij

    return min_dist / 9.0 + 1e-6


def get_adj_from_data(X_l, Y_l=None, X_u=None, adj='knn', k=10, knn_mode='distance', metric='euclidean',
                      self_conn=True):
    """
    Compute the adj matrix
    - Numpy version -
    
    INPUTS:
    X_l: labelled data
    X_u: unlabelled data
    Y_l: labels relative to X_l
    adj: type of adjacency matrix. Options are
        'rbf' --> compute rbf with bandwidth evaluated heuristically with eval_bw
        'knn' --> kNN graph; parameter k must be specified
    k: number of neighbors in the kNN graph or in the linear neighborhood
    knn_mode: 'connectivity' (graph with 0 and 1) or 'distance'
    metric: metric to use to build the knn graph
    self_conn: if True, self connections are removed from adj matrix (A_ii = 0) 
    """

    if X_u is not None:
        X = np.concatenate((X_l, X_u), axis=0)
    else:
        X = X_l

    # Compute transition prob matrix
    if adj == 'rbf':
        # Estimate bandwidth
        if Y_l is None:
            bw = 0.01
        else:
            bw = eval_bw(X_l, np.argmax(Y_l, axis=1))

        # Compute adjacency matrix
        d = squareform(pdist(X, metric='sqeuclidean'))
        A = np.exp(-d / bw)

        # No self-connections (avoids self-reinforcement)
        if self_conn is False:
            np.fill_diagonal(A, 0.0)

    elif adj == 'knn':
        if k is None:
            raise RuntimeError('Specify the number of neighbors (k)!')
        # Compute adjacency matrix
        A = kneighbors_graph(
            X, n_neighbors=k,
            mode=knn_mode,
            metric=metric,
            include_self=self_conn
        ).toarray()
        A = sp.csr_matrix(np.maximum(A, A.T))
    else:
        raise RuntimeError('adj must be "rbf" or "knn"!')

    return A


def _grid(m, dtype=np.float32):
    """Return the embedding of a grid graph."""
    M = m ** 2
    x = np.linspace(0, 1, m, dtype=dtype)
    y = np.linspace(0, 1, m, dtype=dtype)
    xx, yy = np.meshgrid(x, y)
    z = np.empty((M, 2), dtype)
    z[:, 0] = xx.reshape(M)
    z[:, 1] = yy.reshape(M)
    return z


def grid_graph(m, corners=False):
    z = _grid(m)
    A = get_adj_from_data(z, adj='knn', k=8, metric='euclidean')

    # Connections are only vertical or horizontal on the grid.
    # Corner vertices are connected to 2 neightbors only.
    if corners:
        A = A.toarray()
        A[A < A.max() / 1.5] = 0
        A = sp.csr_matrix(A)
        print('{} edges'.format(A.nnz))

    return A


def replace_random_edges(A, noise_level):
    """Replace randomly chosen edges by random edges."""
    M, M = A.shape
    n = int(noise_level * A.nnz // 2)

    indices = np.random.permutation(A.nnz // 2)[:n]
    rows = np.random.randint(0, M, n)
    cols = np.random.randint(0, M, n)
    vals = np.random.uniform(0, 1, n)
    assert len(indices) == len(rows) == len(cols) == len(vals)

    A_coo = sp.triu(A, format='coo')
    assert A_coo.nnz == A.nnz // 2
    assert A_coo.nnz >= n
    A = A.tolil()

    for idx, row, col, val in zip(indices, rows, cols, vals):
        old_row = A_coo.row[idx]
        old_col = A_coo.col[idx]
        A[old_row, old_col] = 0
        A[old_col, old_row] = 0
        A[row, col] = 1
        A[col, row] = 1

    A.setdiag(0)
    A = A.tocsr()
    A.eliminate_zeros()
    return A


###################################################################
# TEXT DOCUMENTS DATASETS PROCESSING
###################################################################
class TextDataset(object):
    """
    Helpers to process text documents
    """

    def clean_text(self, num='substitute'):
        # TODO: stemming, lemmatisation
        for i, doc in enumerate(self.documents):
            # Digits.
            if num is 'spell':
                doc = doc.replace('0', ' zero ')
                doc = doc.replace('1', ' one ')
                doc = doc.replace('2', ' two ')
                doc = doc.replace('3', ' three ')
                doc = doc.replace('4', ' four ')
                doc = doc.replace('5', ' five ')
                doc = doc.replace('6', ' six ')
                doc = doc.replace('7', ' seven ')
                doc = doc.replace('8', ' eight ')
                doc = doc.replace('9', ' nine ')
            elif num is 'substitute':
                # All numbers are equal. Useful for embedding (countable words)?
                doc = re.sub('(\\d+)', ' NUM ', doc)
            elif num is 'remove':
                # Numbers are uninformative (they are all over the place).
                doc = re.sub('[0-9]', ' ', doc)
            # Remove everything except a-z characters and single space.
            doc = doc.replace('$', ' dollar ')
            doc = doc.lower()
            doc = re.sub('[^a-z]', ' ', doc)
            doc = ' '.join(doc.split())  # same as doc = re.sub('\s{2,}', ' ', doc)
            self.documents[i] = doc

    def vectorize(self, **params):
        vectorizer = sk.feature_extraction.text.CountVectorizer(**params)
        self.data = vectorizer.fit_transform(self.documents)
        self.vocab = vectorizer.get_feature_names()
        assert len(self.vocab) == self.data.shape[1]

    def data_info(self, show_classes=False):
        N, M = self.data.shape
        sparsity = self.data.nnz / N / M * 100
        print('N = {} documents, M = {} words, sparsity={:.4f}%'.format(N, M, sparsity))
        if show_classes:
            for i in range(len(self.class_names)):
                num = sum(self.labels == i)
                print('  {:5d} documents in class {:2d} ({})'.format(num, i, self.class_names[i]))

    def show_document(self, i):
        label = self.labels[i]
        name = self.class_names[label]
        try:
            text = self.documents[i]
            wc = len(text.split())
        except AttributeError:
            text = None
            wc = 'N/A'
        print('document {}: label {} --> {}, {} words'.format(i, label, name, wc))
        try:
            vector = self.data[i, :]
            for j in range(vector.shape[1]):
                if vector[0, j] != 0:
                    print('  {:.2f} "{}" ({})'.format(vector[0, j], self.vocab[j], j))
        except AttributeError:
            pass
        return text

    def keep_documents(self, idx):
        """Keep the documents given by the index, discard the others."""
        self.documents = [self.documents[i] for i in idx]
        self.labels = self.labels[idx]
        self.data = self.data[idx, :]

    def keep_words(self, idx):
        """Keep the documents given by the index, discard the others."""
        self.data = self.data[:, idx]
        self.vocab = [self.vocab[i] for i in idx]
        try:
            self.embeddings = self.embeddings[idx, :]
        except AttributeError:
            pass

    def remove_short_documents(self, nwords, vocab='selected'):
        """Remove a document if it contains less than nwords."""
        if vocab is 'selected':
            # Word count with selected vocabulary.
            wc = self.data.sum(axis=1)
            wc = np.squeeze(np.asarray(wc))
        elif vocab is 'full':
            # Word count with full vocabulary.
            wc = np.empty(len(self.documents), dtype=np.int)
            for i, doc in enumerate(self.documents):
                wc[i] = len(doc.split())
        idx = np.argwhere(wc >= nwords).squeeze()
        self.keep_documents(idx)
        return wc

    def keep_top_words(self, M, Mprint=20):
        """Keep in the vocaluary the M words who appear most often."""
        freq = self.data.sum(axis=0)
        freq = np.squeeze(np.asarray(freq))
        idx = np.argsort(freq)[::-1]
        idx = idx[:M]
        self.keep_words(idx)
        print('most frequent words')
        for i in range(Mprint):
            print('  {:3d}: {:10s} {:6d} counts'.format(i, self.vocab[i], freq[idx][i]))
        return freq[idx]

    def normalize(self, norm='l1'):
        """Normalize data to unit length."""
        # TODO: TF-IDF.
        data = self.data.astype(np.float64)
        self.data = sk.preprocessing.normalize(data, axis=1, norm=norm)

    def embed(self, filename=None, size=100):
        """Embed the vocabulary using pre-trained vectors."""
        if filename:
            if not (os.path.exists(filename)):
                gnews_url = 'https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz'
                req = Request(gnews_url, headers={'User-Agent': 'Mozilla/5.0'})
                data = urlopen(req).read()
                with open(filename, 'wb') as out_file:
                    out_file.write(data)

            model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)
            size = model.vector_size
        else:
            class Sentences(object):
                def __init__(self, documents):
                    self.documents = documents

                def __iter__(self):
                    for document in self.documents:
                        yield document.split()

            model = gensim.models.Word2Vec(Sentences(self.documents), size=size)
        self.embeddings = np.empty((len(self.vocab), size))
        keep = []
        not_found = 0
        for i, word in enumerate(self.vocab):
            try:
                self.embeddings[i, :] = model[word]
                keep.append(i)
            except KeyError:
                not_found += 1
        print('{} words not found in corpus'.format(not_found, i))
        self.keep_words(keep)


class Text20News(TextDataset):
    def __init__(self, **params):
        dataset = sklearn.datasets.fetch_20newsgroups(**params)
        self.documents = dataset.data
        self.labels = dataset.target
        self.class_names = dataset.target_names
        assert max(self.labels) + 1 == len(self.class_names)
        N, C = len(self.documents), len(self.class_names)
        print('N = {} documents, C = {} classes'.format(N, C))


def product_dict(kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    if len(keys) > 0:
        for instance in itertools.product(*vals):
            yield OrderedDict(zip(keys, instance))
    else:
        for _ in [dict(), ]:
            yield _


def get_sw_key(sess):
    return [op.name
            for op in sess.graph.get_operations()
            if op.type == "Placeholder" and op.name.endswith('sample_weights')][-1] \
           + ':0'

def create_batch(A_list, X_list, D_list):
    D_out, A_out = list(zip(*D_list)), list(zip(*A_list))
    for i, _ in enumerate(D_out):
        D_out[i] = sp.block_diag(list(D_out[i]))
        A_out[i] = sp.block_diag(list(A_out[i]))
    X_out = np.vstack(X_list)
    n_nodes = np.array([_[0].shape[0] for _ in A_list])
    I_out = np.repeat(np.arange(len(n_nodes)), n_nodes)
    return A_out, X_out, D_out, I_out
