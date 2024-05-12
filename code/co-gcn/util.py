import random
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp
import torch
import numpy as np
import scipy.io as sio
import scipy.sparse as ss
from sklearn.preprocessing import normalize
import h5py
from sklearn import metrics
from sklearn.metrics import pairwise_distances

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def get_data(name):
    features = np.loadtxt('F:\wxh_work\py_semi\co_gcn_master\data_cogcn/{0}/features.csv'.format((name)), delimiter=',')
    targets = np.loadtxt('F:\wxh_work\py_semi\co_gcn_master\data_cogcn/{0}/targets.csv'.format((name))).astype(int)
    print('Dataset {0}:'.format(name), features.shape, targets.shape)
    return features, targets

def load_data(name):
    # features = np.loadtxt('F:\wxh_work\py_semi\co_gcn_master\data_cogcn/{0}/features.csv'.format((name)), delimiter=',')
    # targets = np.loadtxt('F:\wxh_work\py_semi\co_gcn_master\data_cogcn/{0}/targets.csv'.format((name))).astype(int)
    # print('Dataset {0}:'.format(name), features.shape, targets.shape)
    data = h5py.File('F:\wxh_work\datasets\MultiView_Dataset\{0}'.format((name))+'.mat','r');
    num_view=len(data['X'][0])
    fea=[]
    dimension = []
    for i in range(num_view):
        feature = [data[element[i]][:] for element in data['X']]
        feature = np.array(feature)
        feature=np.squeeze(feature)
        feature=feature.T
        # print(feature.shape)
        feature=normalize(feature)
        if ss.isspmatrix(feature):
            feature = feature.todense()
        fea.append(feature)
        dimension.append(feature.shape[1])
        del feature
    Y=np.array(data['Y'])
    Y=Y.T
    Y = Y - min(Y)
    return fea,Y,num_view,dimension
def construct_graph(feature, targets, k=3, metric='euclidean'):
    n = feature.shape[0]
    neigh = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1, metric=metric)
    neigh.fit(feature)
    neigh_dis, neigh_ind = neigh.kneighbors(feature, n_neighbors=k + 1)
    adj = sp.coo_matrix((np.exp(-neigh_dis[:, 1:]).reshape(-1), (neigh_ind[:, 0].repeat(k).reshape(-1), neigh_ind[:, 1:].reshape(-1))),
                        shape=(n, n),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    return adj


