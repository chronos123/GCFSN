import torch
from torch import nn
import torch.nn.functional as F


class Net(nn.Module):
    # 不行可以考虑multi-head attention的k-qv机制
    def __init__(self, dims):
        torch.set_default_tensor_type(torch.FloatTensor)
        super(Net, self).__init__()
        self.layers = nn.ModuleList()
        self.dims = dims
        for i in range(len(self.dims)-1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i+1]))
            # if i == 1:
            #     self.layers.append(nn.InstanceNorm1d(self.dims[i+1]))

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i+1 < len(self.layers):
                x = F.relu(x)
        return x


class Attention(nn.Module):
    """
    Simple Attention layer
    """
    def __init__(self, in_features, nhid, master_node):
        super(Attention, self).__init__()
        self.in_features = in_features
        self.nhid = nhid
        self.master_node = master_node
        self.fc1 = nn.Linear(in_features, nhid)
        self.fc2 = nn.Linear(nhid, 1, bias=False)
        self.fc3 = nn.Linear(nhid, in_features)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
    
    def forward(self, x_in):
        x = torch.tanh(self.fc1(x_in))
        x = torch.tanh(self.fc2(x))
        if self.master_node:
            t = self.softmax(x[:,:-1,:])
            t = t.unsqueeze(3)
            x = x_in[:,:-1,:].repeat(1, 1, 1)
            x = x.view(x.size()[0],x.size()[1], 1, self.nhid)
            t = t.repeat(1, 1, 1, x_in.size()[2])*x
            t = t.view(t.size()[0], t.size()[1], -1)
            t = t.sum(1)
            t = self.relu(self.fc3(t))
            out = torch.cat([t, x_in[:,-1,:].squeeze()], 1)
        else:
            t = self.softmax(x)
            t = t.unsqueeze(3)
            x = x_in.repeat(1, 1, 1)
            x = x.view(x.size()[0],x.size()[1], 1, self.nhid)
            t = t.repeat(1, 1, 1, x_in.size()[2])*x
            t = t.view(t.size()[0], t.size()[1], -1)
            out = self.relu(self.fc3(t))
            
        return out
