import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from GEN import GEN, GENVpred


class GENSoftNNVpred(GENVpred):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.repr_fn_log_strength = torch.nn.Parameter(torch.zeros(474))

    def repr_fn(self, node_pos, x_inp, **kwargs):
        return self.compute_coordinates_soft_nn(node_pos, x_inp, **kwargs)

    def set_repr_fn_log_strength(self, log_strength):
        self.repr_fn_log_strength = log_strength

    def compute_coordinates_soft_nn(self, node_pos, x, log_strength=None):
        if log_strength is None: log_strength = self.repr_fn_log_strength
        assert log_strength is not None
        #Take out batch dimension
        bs = 1 if len(x.shape) == 2 else x.shape[0]
        inps_per_elt, features = x.shape[-2], x.shape[-1]
        pos = x.reshape(-1,features)[:,:2]
        #Compute pseudo-Squared Error distance
        #Using (x-y)^2 = x^2-2xy+y^2 \equivalent (y ctt) x^2-2xy
        pseudo_dist = (torch.norm(node_pos, dim=1)**2 - 2*torch.mm(pos, node_pos.t()))
        # dist = torch.abs(node_pos - pos)
        
        if torch.cuda.is_available():
            log_strength = log_strength.cuda()
            pseudo_dist = pseudo_dist.cuda()
        scores = (F.softmax(-torch.exp(log_strength)*pseudo_dist, dim=1))
        # scores = F.softmax(-torch.exp(log_strength)*dist, dim=1)
        return scores.reshape((bs, inps_per_elt, -1))
  

class GENSoftNN(GEN):
    def __init__(self, nodes=None, **kwargs):
        super(GENSoftNN, self).__init__(**kwargs)
        if self.exp == "1":
            self.repr_fn_log_strength = torch.nn.Parameter(torch.zeros(474))
        elif self.exp == "2":
            self.repr_fn_log_strength = torch.nn.Parameter(torch.zeros(543))
        elif self.exp == "3":
            if nodes == None:
                self.repr_fn_log_strength = torch.nn.Parameter(torch.zeros(504))
            else:
                self.repr_fn_log_strength = torch.nn.Parameter(torch.zeros(nodes))
        elif self.exp == "4":
            self.repr_fn_log_strength = torch.nn.Parameter(torch.zeros(676))
        elif self.exp == "5":
            self.repr_fn_log_strength = torch.nn.Parameter(torch.zeros(546))
        elif self.exp == "6":
            self.repr_fn_log_strength = torch.nn.Parameter(torch.zeros(684))
        else:
            self.repr_fn_log_strength = torch.nn.Parameter(torch.zeros(1))
            
    def repr_fn(self, node_pos, x_inp, **kwargs):
        return self.compute_coordinates_soft_nn(node_pos, x_inp, **kwargs)

    def set_repr_fn_log_strength(self, log_strength):
        self.repr_fn_log_strength = log_strength

    def compute_coordinates_soft_nn(self, node_pos, x, log_strength=None):
        if log_strength is None: log_strength = self.repr_fn_log_strength
        assert log_strength is not None
        #Take out batch dimension
        bs = 1 if len(x.shape) == 2 else x.shape[0]
        inps_per_elt, features = x.shape[-2], x.shape[-1]
        pos = x.reshape(-1,features)[:,:2]
        #Compute pseudo-Squared Error distance
        #Using (x-y)^2 = x^2-2xy+y^2 \equivalent (y ctt) x^2-2xy
        pseudo_dist = (torch.norm(node_pos, dim=1)**2 - 2*torch.mm(pos, node_pos.t()))
        # dist = torch.abs(node_pos - pos)
        
        if torch.cuda.is_available():
            log_strength = log_strength.cuda()
            pseudo_dist = pseudo_dist.cuda()
        scores = (F.softmax(-torch.exp(log_strength)*pseudo_dist, dim=1))
        # scores = F.softmax(-torch.exp(log_strength)*dist, dim=1)
        return scores.reshape((bs, inps_per_elt, -1))
