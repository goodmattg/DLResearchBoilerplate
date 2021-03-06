import operator
import os
import sys
import yaml
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import tensorflow as tf

from functools import reduce
from dotmap import DotMap
from tensorflow.python.lib.io import file_io
from tensorflow.python.framework.errors_impl import NotFoundError
from scipy.sparse.linalg.eigen.arpack import eigsh


def get_from_dict(dictionary, key_list):
    """Get value from dictionary with arbitrary depth via descending list of keys."""
    return reduce(operator.getitem, key_list, dictionary)


def set_in_dict(dictionary, key_list, value):
    """Set value from dictionary with arbitrary depth via descending list of keys."""
    get_from_dict(dictionary, key_list[:-1])[key_list[-1]] = value


def cast_to_type(type_abbrev, val):
    """Convert string input value to explicitly denoted type. Types are as follows:
    "f" -> float
    "i" -> integer
    "b" -> boolean
    "s" -> string
    """
    if type_abbrev == "f":
        return float(val)
    elif type_abbrev == "i":
        return int(val)
    elif type_abbrev == "b":
        return val.lower() in ("yes", "true", "t", "1")
    else:
        return val


def override_dotmap(overrides, config):
    """Override DotMap dictionary with explicitly typed values."""
    for i in range(len(overrides) // 3):
        key, type_abbrev, val = overrides[i * 3 : (i + 1) * 3]
        set_in_dict(config, key.split("."), cast_to_type(type_abbrev, val))


# Base Utilities (standard to boilerplate repository)
def load_file(filepath, load_func, **kwargs):
    """Generic file loader with missing/error messages."""
    try:
        print("Loading data file from: {0}".format(filepath))
        return load_func(filepath, **kwargs)
    except NotFoundError as e:
        print("Data file not found: {0}".format(filepath))
        exit(2)
    except Exception as e:
        print("Unable to load data file: {0}".format(filepath))
        exit(3)


def pickle_loader(filepath, encoding=None):
    """Load pickle file"""
    with file_io.FileIO(filepath, mode="rb") as stream:
        if encoding:
            return pkl.load(stream, encoding=encoding)
        else:
            return pkl.load(stream)


def index_loader(filepath):
    """Parse index file."""
    index = []
    with file_io.FileIO(filepath, mode="r") as stream:
        for line in stream:
            index.append(int(line.strip()))
        return index


def yaml_loader(filepath, use_dotmap=True):
    """Load a yaml file into a dictionary. Optionally wrap with DotMap."""
    with file_io.FileIO(filepath, mode="r") as stream:
        if use_dotmap:
            return DotMap(yaml.load(stream))
        else:
            return yaml.load(stream)


def load_training_config_file(filename):
    """Load a training configuration yaml file into a DotMap dictionary."""
    config_file_path = os.path.join(
        os.path.dirname(__file__), "config", "{0}.yaml".format(filename)
    )
    return load_file(config_file_path, yaml_loader)


def load_data_index_file(filename):
    """Load an index data file/"""
    data_file_path = os.path.join(os.path.dirname(__file__), "data", filename)
    return load_file(data_file_path, index_loader)


def load_data_pickle_file(filename, encoding=None):
    """Load a data pickle file/"""
    data_file_path = os.path.join(os.path.dirname(__file__), "data", filename)
    return load_file(data_file_path, pickle_loader, encoding=encoding)


# Custom Utilities for GCN example


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ["x", "y", "tx", "ty", "allx", "ally", "graph"]
    objects = []
    for i in range(len(names)):
        filename = "ind.{}.{}".format(dataset_str, names[i])
        encoding = "latin1" if sys.version_info > (3, 0) else None
        objects.append(load_data_pickle_file(filename, encoding=encoding))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = load_data_index_file("ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == "citeseer":
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_matrix_to_sparse_tensor(sparse_mx):
    """Convert sparse matrix to SparseTensor representation."""

    def to_sparse_tensor(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return tf.SparseTensor(coords, values, shape)

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = tf.cast(to_sparse_tensor(sparse_mx[i]), tf.float32)
    else:
        sparse_mx = tf.cast(to_sparse_tensor(sparse_mx), tf.float32)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to SparseTensor representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_matrix_to_sparse_tensor(adj_normalized)


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of SparseTensors."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which="LM")
    scaled_laplacian = (2.0 / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_matrix_to_sparse_tensor(t_k)
