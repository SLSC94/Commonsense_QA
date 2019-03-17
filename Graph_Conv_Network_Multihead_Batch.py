from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import math
class GCN_Multihead_batch(Module):

    def __init__(self, heads, in_features, out_features, final = False, bias=True):
        super(GCN_Multihead_batch, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.final = final
        
        self.heads = heads
        self.weights = Parameter(torch.FloatTensor(heads,in_features, out_features))
        self.bias = Parameter(torch.FloatTensor(heads,out_features))
        self.Tanh = nn.Tanh()
        self.reset_parameters()


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weights.size(1)) 
        self.weights.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, dictionary):
        
        input= dictionary['input'] 
        # a matrix of batch*heads*nodes*inputdim
        adj = dictionary['adj']
        # a matrix of batch*d*d
        
        self.output = {'adj':adj}
      
        #batch size will be in the first dimension. 
        self.batch_size = adj.size()[0]
        self.n_nodes = adj.size()[1]
        
        adj_row_sums = adj.sum(dim = 1)**(-0.5)
        
        D = (torch.zeros(self.batch_size, self.n_nodes, self.n_nodes))
        D.as_strided(adj_row_sums.size(), 
                     [D.stride(0), D.size(2) + 1]).copy_(adj_row_sums)
        #note: the adjacency matrix might have negative entries. 
        
        adj = torch.einsum('ijk,ikl,ilm->ijm',  D, adj, D)
        
        support = torch.einsum('ijkl,jlm->ijkm', input, self.weights)
#         print(self.bias.shape)
        self.output['input'] = self.Tanh(torch.einsum('ijk,ilkn->iljn',adj,support) + self.bias[:,None,:]) 
            
        self.output['pooled'] = dictionary['pooled']
        return self.output
      
class unpack(Module):

    def __init__(self, heads):
        super(unpack, self).__init__()
        self.heads = heads
    def forward(self, dictionary):
        input= dictionary['input'] 
        # a list of inputs (12 batch* 4 *768 feature matrices)
        pooled = dictionary['pooled']
        # adj = dictionary['adj'] # a single adjacency matrix

        self.batch_size = input.shape[0]
        
        self.output = torch.cat((pooled.view(self.batch_size, -1), 
                                 input.contiguous().view(self.batch_size,-1)),dim = 1)
        
        return self.output