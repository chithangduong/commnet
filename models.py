import torch 
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn 
from torch.nn import init
from gumbel import gumbel_softmax

class Model(nn.Module):
    def __init__(self, n_nodes, n_parts):

        super(Model, self).__init__()
        self.input_dim = n_nodes
        self.num_parts = n_parts

        self.params = nn.Parameter(init.xavier_normal_(torch.Tensor(self.input_dim, self.num_parts)), requires_grad=True)

    def forward(self, adj, temp=10, hard=False, beta=1):
        self.assign_tensor = gumbel_softmax(self.params, temp=temp, hard=hard, beta=beta)
        self.assign_tensor_t = torch.transpose(self.assign_tensor, 0, 1)

        super_adj = self.assign_tensor_t @ adj @ self.assign_tensor # A' = S^T*A*S
        return super_adj

    def loss(self, super_adj, balance_node=True, lam = 0.7):
        ncut_loss = torch.sum(torch.tril(super_adj, diagonal=-1) + torch.triu(super_adj, diagonal=1))

        if balance_node:
            balance_loss = torch.sum((torch.sum(self.assign_tensor, dim=0) - self.input_dim//self.num_parts)**2)
        else:
            balance_loss = torch.sum((torch.diagonal(super_adj) - torch.sum(torch.diagonal(super_adj))//self.num_parts)**2)
        loss = lam*ncut_loss + (1-lam)*torch.sqrt(balance_loss)
        return loss 
