import numpy as np

from sklearn.model_selection import train_test_split

import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


######################################################################
##################    Input Form   :   .npz File    ##################
######################################################################
def load_npz(file_name):
  # load
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']), shape=loader['adj_shape'])

        if 'attr_data' in loader:
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'], loader['attr_indptr']), shape=loader['attr_shape'])
        else:
            attr_matrix = None

        labels = loader.get('labels')

    return adj_matrix, attr_matrix, labels


######################################################################
#############     Input Form  :   ind. File         ##################
######################################################################
def load_data(dataset_str): # {'pubmed', 'citeseer', 'cora'}
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        
        with open("data-Fng/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    
    test_idx_reorder = parse_index_file("data-Fng/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    #//labels = labels[:, 1]
    
    print(adj.shape)
    print(features.shape)
    print(labels.shape)
    return adj, features, labels







#* #######################################################################
#* ######################   Preprocess Data    ###########################
#* #######################################################################
def largest_connected_components(adj, n_components=1): # Keep one LCC
  # Get subgraph(adjMatrix) with only nodes in LCC
    _, component_indices = connected_components(adj)
    component_sizes = np.bincount(component_indices)
    #//print(component_sizes) # [2485    2    3    9  ... 8    3    2    2    2    2]
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep
    ]
    print("Selecting {0} largest connected components".format(n_components))
    return nodes_to_keep


# For Adj Matrix
def preprocess_graph(adj):
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = adj_.sum(1).A1
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5))
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).T.dot(degree_mat_inv_sqrt).tocsr()
    return adj_normalized


def train_val_test_split_tabular(*arrays, train_size=0.5, val_size=0.3, test_size=0.2, stratify=None, random_state=None):
  # Split Matrix into random train, validation and test
    #@ *arrays[lists, numpy arrays or scipy-sparse matrices] : with same length / shape[0]
    #@ stratify[array-like or None] : as the class labels
    #@ random_state[int or None] : the seed fed to random num generator
    #@# Return splitting[list] : len = 3*len(arrays) containing train-validation-test split

    if len(set(array.shape[0] for array in arrays)) != 1: raise ValueError("Arrays must have equal first dimension.")
    idx = np.arange(arrays[0].shape[0])
    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=random_state,
                                                   train_size=(train_size + val_size),
                                                   test_size=test_size,
                                                   stratify=stratify)
    if stratify is not None: # split train and validation
        stratify = stratify[idx_train_and_val]
    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=random_state,
                                          train_size=(train_size / (train_size + val_size)),
                                          test_size=(val_size / (train_size + val_size)),
                                          stratify=stratify)
    result = []
    for X in arrays:
        result.append(X[idx_train])
        result.append(X[idx_val])
        result.append(X[idx_test])
    return result