import math
import torch
import numpy as np
from scipy.special import softmax
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class MixedGraphConvolution(nn.Module):
    """
    Mixed CoGCN layer
    """
    def __init__(self, in_features, out_features, adjs, n_view, lr_alpha, device, bias=True):
        super(MixedGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.alpha = np.ones(n_view) / n_view
        self.adjs = adjs
        self.n = self.adjs[0].size()[0]
        self.adj = torch.sparse.FloatTensor(self.n, self.n)
        self.H = None
        self.input = None
        self.lr_alpha = lr_alpha
        self.device = device

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
            nn.init.uniform_(self.bias)
        else:
            self.register_parameter('bias', None)
        self._reset_parameters()
        self._update_adj()
        self.n_view = n_view

    def _reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def _update_adj(self):
        np.clip(self.alpha, 0.0, 1.0)
        self.alpha = softmax(self.alpha)
        self.adj = torch.sparse.FloatTensor(self.n, self.n).to(self.device)
        for alpha, adj in zip(self.alpha, self.adjs):
            self.adj.add_(alpha * adj)

    def update_alpha(self):
        for i in range(self.n_view):
            support = torch.spmm(self.adjs[i], self.input)
            partial_H = torch.mm(support, self.weight)
            gHT = torch.transpose(self.H.grad, 0, 1)
            self.alpha[i] -= self.lr_alpha * torch.trace(torch.mm(gHT, partial_H)).item()
            self._update_adj()

    def forward(self, input):
        self.input = input
        support = torch.mm(input, self.weight)
        self.H = torch.sparse.mm(self.adj, support)
        self.H.retain_grad()
        if self.bias is not None:
            return self.H + self.bias
        else:
            return self.H

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class CoGCN(nn.Module):
    def __init__(self, nfeat, nclass, adjs, n_view, lr_alpha, dropout, device):
        super(CoGCN, self).__init__()
        nhid = max(nfeat // 1280 * 64, 64)
        self.gc1 = MixedGraphConvolution(nfeat, nhid, adjs, n_view, lr_alpha, device)
        self.gc2 = MixedGraphConvolution(nhid, nclass, adjs, n_view, lr_alpha, device)
        self.dropout = dropout

    def forward(self, x):
        x = F.relu(self.gc1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x)
        return F.log_softmax(x, dim=1)

