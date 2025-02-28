import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GraphConv, GCNConv, GATConv
from utils import Attention
from multihead_attention import MultiHeadAttention
from my_dataset_64 import room


class Vpred(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pred_pos = nn.Sequential(
                nn.Linear(2, 8),
                nn.ReLU(),
                nn.Linear(8, 16),
                nn.ReLU(),
            )   
        self.pred_v = nn.Sequential(
                nn.Linear(1, 8),
                nn.ReLU(),
                nn.Linear(8, 16),
                nn.ReLU(),
            ) 
        self.pred_t = nn.Sequential(
                nn.Linear(1, 8),
                nn.ReLU(),
                nn.Linear(8, 16),
                nn.ReLU(),
            ) 
        self.multi_head = MultiHeadAttention(16, 2)
        # self.out_layer = nn.Linear(16, 1)
        self.out_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(16, 1),
        )
    
    def forward(self, t, pos, v):
        t = t.detach()
        v = v.detach()
        t = self.pred_t(t)
        v = self.pred_v(v)
        pos = self.pred_pos(pos)
        v = self.multi_head(pos, t, v)
        return self.out_layer(v)
        

class GENVpred(nn.Module):
    def __init__(self, encoders, decoders, G=None, msg_steps=None):
        super().__init__()
        self.encoders = encoders
        self.decoders = decoders
        self.G = G if G is not None else Data()
        self.msg_steps = msg_steps
        #should match w/ num_features = num_node_features
        self.G.num_feat = self.encoders[0].layers[-1].out_features
        self.G.num_dimensions = (
                self.G.pos.shape[-1] if self.G.pos is not None else
                self.decoders[0].layers[0].in_features - self.G.num_feat)
        for enc in self.encoders:
            assert enc.layers[-1].out_features == self.G.num_feat
        for dec in self.decoders:
            assert (dec.layers[0].in_features ==
                    self.G.num_feat + self.G.num_dimensions)
        self.conv = GCNConv(self.G.num_feat + self.G.num_dimensions,
                self.G.num_feat) #position shouldn't be touched
        # self.layer_norm = nn.modules.normalization.LayerNorm(self.G.num_feat)
        self.bn = nn.BatchNorm1d(474)
        self.attn = Attention(self.G.num_feat, nhid=self.G.num_feat, master_node=False)
        # self.g_attn = GATConv(in_channels=32, out_channels=32, heads=2)
        self.pred_v = Vpred()

    def set_node_pos(self, node_pos):
        self.G.pos = node_pos

    def set_msg_steps(self, msg_steps):
        self.msg_steps = msg_steps

    def forward(self, Inp, Q, G=None, repr_fn_args={}):
        '''
        Inp: list of input points (X, y_i) of function i, Inp[0][0] 和 Inp[0][1]
        Q:   list of queries X for function j, Q[0] is coordinate, Q[1] is temperature
        '''
        if G is None: G = self.G
        else:
            G.num_feat = self.G.num_feat
            G.num_dimensions = self.G.num_dimensions
        assert G.pos is not None
        if hasattr(G, 'msg_steps'): msg_steps = G.msg_steps
        if msg_steps is None:
            if self.msg_steps is not None: msg_steps = self.msg_steps
            else: msg_steps = G.num_nodes*2-1
        # Encode all inputs
        inputs = [] #(BS, #inp, feat)
        for (inp, enc) in zip(Inp, self.encoders):
            res = (enc(torch.cat((inp[0], inp[1]), dim=-1)))
            inputs.append(res)
        inputs = torch.cat(inputs, dim=1) 
        x_inp = torch.cat([inp[0] for inp in Inp], dim=1)
        # Initialize GNN node states with representation function
        inp_coord = self.repr_fn(G.pos, x_inp, **repr_fn_args)
        G.x = torch.bmm(torch.transpose(inp_coord, 1, 2), inputs)
        # bs, num_nodes, f = G.x.shape
        # # Create Batch to feed to GNN
        # data_list = [Data(x=x.squeeze(0), pos=G.pos, edge_index=G.edge_index)
        #         for x in torch.split(G.x,split_size_or_sections=1,dim=0)]
        # SG = Batch.from_data_list(data_list)

        # # Propagate GNN states with message passing
        # for step in range(msg_steps):
        #     SG.x = self.layer_norm(SG.x + self.conv(
        #         torch.cat((SG.pos, SG.x), dim=-1), SG.edge_index))
        
        # G.x = SG.x.reshape((SG.num_graphs,-1,f))
        
        for step in range(msg_steps):
            G.x = self.bn(G.x + self.conv(
                torch.cat((G.pos.repeat(room, 1, 1), G.x), dim=-1), G.edge_index))
            
        G.x = self.attn(G.x)
        
        queries = [] #(BS, #out, feat)
        # Decode hidden states to final outputs
        res = []
        for (q, dec) in zip(Q, self.decoders):
            q_coord = self.repr_fn(G.pos, q, **repr_fn_args)
            lat = torch.bmm(q_coord, G.x)
            out = dec(torch.cat((lat, q), dim=-1))
            t = out[:, :, 0]
            v_pre = out[:, :, 1]
            t = t.unsqueeze(-1)
            v_pre = v_pre.unsqueeze(-1)
            v = self.pred_v(t, q, v_pre)
            res.append(torch.cat((t, v), dim=-1))
        return res

    def repr_fn(self, **kwargs):
        raise NotImplementedError("the default GEN class doesn't have \
                the repr_fn implemented, a reasonable default is GENSoftNN")


class GEN(nn.Module):
    def __init__(self, encoders, decoders, exp, G=None, msg_steps=None):
        super(GEN, self).__init__()
        self.encoders = encoders
        self.t_decoders = decoders
        self.decoders = decoders
        self.exp = exp
        self.v_decoders = nn.Sequential(
                nn.Linear(34, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.LayerNorm(32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.ReLU()
        )
        self.G = G if G is not None else Data()
        self.msg_steps = msg_steps
        #should match w/ num_features = num_node_features
        self.G.num_feat = self.encoders[0].layers[-1].out_features
        self.G.num_dimensions = (
                self.G.pos.shape[-1] if self.G.pos is not None else
                self.decoders[0].layers[0].in_features - self.G.num_feat)
        for enc in self.encoders:
            assert enc.layers[-1].out_features == self.G.num_feat
        for dec in self.decoders:
            assert (dec.layers[0].in_features ==
                    self.G.num_feat + self.G.num_dimensions)
        self.conv = GCNConv(self.G.num_feat + self.G.num_dimensions,
                self.G.num_feat) #position shouldn't be touched
        self.layer_norm = nn.modules.normalization.LayerNorm(self.G.num_feat)
        self.conv1 = GCNConv(self.G.num_feat + self.G.num_dimensions,
                self.G.num_feat) #position shouldn't be touched
        self.layer_norm1 = nn.modules.normalization.LayerNorm(self.G.num_feat)
        self.attn = Attention(self.G.num_feat, nhid=self.G.num_feat, master_node=False)
        self.attn1 = Attention(self.G.num_feat, nhid=self.G.num_feat, master_node=False)


    def set_node_pos(self, node_pos):
        self.G.pos = node_pos

    def set_msg_steps(self, msg_steps):
        self.msg_steps = msg_steps

    def forward(self, Inp, Q, G=None, repr_fn_args={}):
        '''
        Inp: list of input points (X, y_i) of function i, Inp[0][0] 和 Inp[0][1]
        Q:   list of queries X for function j, Q[0] is coordinate, Q[1] is temperature
        '''
        if G is None: G = self.G
        else:
            G.num_feat = self.G.num_feat
            G.num_dimensions = self.G.num_dimensions
        assert G.pos is not None
        if hasattr(G, 'msg_steps'): msg_steps = G.msg_steps
        if msg_steps is None:
            if self.msg_steps is not None: msg_steps = self.msg_steps
            else: msg_steps = G.num_nodes*2-1
        # Encode all inputs
        inputs = [] #(BS, #inp, feat)
        for (inp, enc) in zip(Inp, self.encoders):
            res = (enc(torch.cat((inp[0], inp[1]), dim=-1)))
            inputs.append(res)
        inputs = torch.cat(inputs, dim=1)
        x_inp = torch.cat([inp[0] for inp in Inp], dim=1)
        # Initialize GNN node states with representation function
        inp_coord = self.repr_fn(G.pos, x_inp, **repr_fn_args)
        G.x = torch.bmm(torch.transpose(inp_coord, 1, 2), inputs)
        bs, num_nodes, f = G.x.shape
        # # Create Batch to feed to GNN
        # data_list = [Data(x=x.squeeze(0), pos=G.pos, edge_index=G.edge_index)
        #         for x in torch.split(G.x,split_size_or_sections=1,dim=0)]
        # SG = Batch.from_data_list(data_list)

        # # Propagate GNN states with message passing
        # for step in range(msg_steps):
        #     SG.x = self.layer_norm(SG.x + self.conv(
        #         torch.cat((SG.pos, SG.x), dim=-1), SG.edge_index))
        
        # G.x = SG.x.reshape((SG.num_graphs,-1,f))
        x_v = G.x.detach()
        data_list = [Data(x=x.squeeze(0), pos=G.pos, edge_index=G.edge_index)
                for x in torch.split(x_v, split_size_or_sections=1, dim=0)]
        
        SG = Batch.from_data_list(data_list)
        if self.exp == "6":
            msg_steps = 20
        if self.exp == "2":
            msg_steps = 15
        if self.exp == "3":
            msg_steps = 17
        for step in range(msg_steps):
            G.x = self.layer_norm(G.x + self.conv(
                torch.cat((G.pos.repeat(room, 1, 1), G.x), dim=-1), G.edge_index))
            if self.exp == "2":
                x_v = self.layer_norm1(x_v + self.conv1(
                        torch.cat((G.pos.repeat(room, 1, 1), x_v), dim=-1), G.edge_index))
            else:
                if msg_steps % 2 == 0:
                    SG.x = self.layer_norm1(SG.x + self.conv1(
                        torch.cat((SG.pos, SG.x), dim=-1), SG.edge_index))
            
        G.x = self.attn(G.x)
        x_v = SG.x.reshape((SG.num_graphs,-1,f))
        x_v = self.attn1(x_v)
        
        queries = [] #(BS, #out, feat)
        # Decode hidden states to final outputs
        res = []
        for (q, dec_t, dec_v) in zip(Q, self.t_decoders, [self.v_decoders]):
            q_coord = self.repr_fn(G.pos, q, **repr_fn_args)
            lat1 = torch.bmm(q_coord, G.x)
            lat2 = torch.bmm(q_coord, x_v)
            t = dec_t(torch.cat((lat1, q), dim=-1))
            v = dec_v(torch.cat((lat2, q), dim=-1))
            res.append(torch.cat((t, v), dim=-1))
        return res

    def repr_fn(self, **kwargs):
        raise NotImplementedError("the default GEN class doesn't have \
                the repr_fn implemented, a reasonable default is GENSoftNN")
