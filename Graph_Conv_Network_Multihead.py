from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import math

class GCN_Multihead(Module):

    def __init__(self, heads, in_features, out_features, bias=True):
        super(GCN_Multihead, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.heads = heads
        self.weights = {}
        self.bias = {}
        for i in range(self.heads):
            self.weights[i] = Parameter(torch.FloatTensor(in_features, out_features))
            self.bias[i] = Parameter(torch.FloatTensor(out_features))

            
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weights[0].size(1))
        for i in range(self.heads):
            self.weights[i].data.uniform_(-stdv, stdv)
            self.bias[i].data.uniform_(-stdv, stdv)


    def forward(self, dictionary):
        
        input= dictionary['input'] # a list of inputs (12 4 *768 feature matrices)
        adj = dictionary['adj'] # a single adjacency matrix
        
        self.output = {'input':[], 'adj':adj}
      
        #batch size will be in the first dimension. 
        self.batch_size = adj.size()[0]
        self.n_nodes = adj.size()[1]
        
        adj_row_sums = adj.sum(dim = 1)**(-0.5)
        
        D = (torch.zeros(self.batch_size, self.n_nodes, self.n_nodes))
        D.as_strided(adj_row_sums.size(), 
                     [D.stride(0), D.size(2) + 1]).copy_(adj_row_sums)
        #note: the adjacency matrix might have negative entries. 
        
        adj = torch.einsum('ijk,ikl,ilm->ijm',  D, adj, D)
        
        for i in range(self.heads):
            support = torch.einsum('ijk,kl->ijl', input[i], self.weights[i])
            output = torch.einsum('ijk,ikl->ijl', adj, support) + self.bias[i]
                      
            self.output['input'].append(output)
            
                      
        return self.output
    
    
class unpack(Module):

    def __init__(self, heads):
        super(unpack, self).__init__()
        self.heads = heads
    def forward(self, dictionary):
        input= dictionary['input'] # a list of inputs (12 4 *768 feature matrices)
        # adj = dictionary['adj'] # a single adjacency matrix
        
        self.output = input[0]
        for i in range(1,self.heads):
            self.output = torch.cat((self.output, input[i]),dim=2)
                      
        return self.output.reshape([1,-1])
