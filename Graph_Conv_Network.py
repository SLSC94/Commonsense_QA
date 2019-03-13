import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import math

class GCN(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, dictionary):
        
        
        
        
        input= dictionary['input']
        adj = dictionary['adj']
        
        #batch size will be in the first dimension. 
        self.batch_size = adj.size()[0]
        self.n_nodes = adj.size()[1]
        
        adj_row_sums = adj.sum(dim = 1)**(-0.5)
        
        D = (torch.zeros(self.batch_size, self.n_nodes, self.n_nodes))
        D.as_strided(adj_row_sums.size(), 
                     [D.stride(0), D.size(2) + 1]).copy_(adj_row_sums)
        #note: the adjacency matrix might have negative entries. 
        
        adj = torch.einsum('ijk,ikl,ilm->ijm',  D, adj, D)
        
        support = torch.einsum('ijk,kl->ijl', input, self.weight)
        output = torch.einsum('ijk,ikl->ijl', adj, support)
        if self.bias is not None:
            return {'input':output + self.bias, 'adj':dictionary['adj']}
        else:
            return {'input':output, 'adj':dictionary['adj']}
