import torch
from gen_softnn import GENSoftNN, GENSoftNNVpred
from torch import nn
from utils import Net
import os
from scipy.io import loadmat
from scipy.io import savemat
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
from matplotlib import collections as mc
from create_graph import GraphLessMsgStep
from gen_datasets import FTDataset
from my_dataset_64 import HeatInpDataset, \
        HeatOutDataset
from torch.utils import data
from torch.utils.data import DataLoader
from tqdm import tqdm
from my_dataset_64 import room
import argparse


class AvgMetrics:
    def __init__(self):
        self.t_relative_err = 0
        self.v_relative_err = 0
        self.v_mean_err = 0
        self.cnt = 1

    def update(self, pred, truth):
        pred = pred.detach().cpu().numpy()
        truth = truth.detach().cpu().numpy()
        r = np.where(truth == 0, 1, np.abs(truth))
        self.t_relative_err += np.mean(np.abs(pred[:, 0] - truth[:, 0]) / r[:, 0])
        self.v_relative_err += np.mean(np.abs(pred[:, 1] - truth[:, 1]) / r[:, 1])
        self.v_mean_err += np.mean(np.abs(pred[:, 1] - truth[:, 1]))
        self.cnt += 1
    
    def get(self):
        return (self.t_relative_err / self.cnt), (self.v_relative_err / self.cnt), self.v_mean_err / self.cnt


colormap_show = 'viridis'
size_s = 25
marker = "."


def test_all(args):
    exp = args.exp
    cuda = torch.cuda.is_available()
    # cuda = False
    device = torch.device('cuda') if cuda else torch.device('cpu')
    # device = torch.device('cpu')
    
    k = 32
    encoders = nn.ModuleList([Net(dims=[3, k, k, k])])
    decoders = nn.ModuleList([Net(dims=[k+2, k, k, 1])])

    net = GENSoftNN(encoders=encoders, decoders=decoders, exp=args.exp)
    if cuda:
        net.cuda()
    # for param_tensor in net.state_dict():
    #     print(param_tensor, "\t", net.state_dict()[param_tensor].size())
    # print()
    state_dict = torch.load(args.ckpt)
    net.load_state_dict(state_dict)
    net.eval()

    file_args = {'file_path' : f'{args.data_path}'}
    file_args1 = {'file_path' : f'{args.data_path}'}
    
    full_dataset = FTDataset(inp_datasets=[HeatInpDataset],
            inp_datasets_args = [file_args],    
            out_datasets = [HeatOutDataset],
            out_datasets_args = [file_args1],  
            idx_list=None)
    
    train_size = int(0.8*len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = data.random_split(full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0,
            shuffle=False, drop_last=False)

    metrics = AvgMetrics()
    test_loaders = [test_loader] * room
    with torch.no_grad():
        if exp == "1":
            graph = GraphLessMsgStep(data_idx=-1).train_heat_graph(device=device)
        elif exp == '2':
            graph = GraphLessMsgStep(data_idx=-1).train_heat_graph_2(device=device)
        elif exp == "3":
            graph = GraphLessMsgStep(data_idx=-1).train_heat_graph_3(device=device)
        elif exp == "4":
            graph = GraphLessMsgStep(data_idx=-1).train_heat_graph_4(device=device)
        elif exp == "5":
            graph = GraphLessMsgStep(data_idx=-1).train_heat_graph_5(device=device)
        elif exp == "6":
            graph = GraphLessMsgStep(data_idx=-1).train_heat_graph_6(device=device)
        else:
            raise NotImplementedError()
         
        for cnt, it in tqdm(enumerate(test_loaders)):
            for ((Inp,Out), idx) in it:
                if cuda:
                    for d in Out:
                        d[0] = d[0].cuda()
                        d[1] = d[1].cuda()
                    for d in Inp:
                        d[0] = d[0].cuda()
                        d[1] = d[1].cuda()
                for d in Inp:
                    d[0] = d[0].view([-1] + list(d[0].shape[2:]))
                    d[1] = d[1].view([-1] + list(d[1].shape[2:]))
                for d in Out:
                    d[0] = d[0].view([-1] + list(d[0].shape[2:]))
                    d[1] = d[1].view([-1] + list(d[1].shape[2:]))
                Q = [o[0] for o in Out]
                targets = [o[1] for o in Out]
                for num in range(32):
                    index = num % room
                    test_inp = Inp[0][1][index, :, :]
                    Inp[0][1] = test_inp.repeat(room, 1, 1)
                    
                    preds = net(Inp, Q, G=graph)
                    test_truth = targets[0][index, :, :]
                    pred_one = torch.mean(preds[0], dim=0)
                    metrics.update(pred_one, test_truth)
    t_err, v_err, v_mean_e = metrics.get()
    print(f't: mean relative error is {t_err}')
    print(f'v: mean relative error is {v_err}')
    print(f'v: mean error is {v_mean_e}')
    

def get_query_coordinates_for_show(coordinates):
    coordinates_np = coordinates.detach().cpu().numpy()
    return coordinates_np[:, 0], coordinates_np[:, 1]
    

def relative_error_calc(pred, truth, G, x, y, test_num, label="t", show_graph=False, show=False):
    error = np.abs(truth - pred)
    r = np.abs(np.where(truth==0, 1, truth))
    
    print(f'{label}-{test_num}: error mean is {np.mean(abs(error))}')
    print(f'{label}-{test_num}: mean relative error is {np.mean(abs(error/r))}')
    print(f'{label}-{test_num}: max relative error is {np.max(abs(error/r))}')
    
    plt.figure(f"{label}_pred_mean")
    plt.scatter(x, y, s=size_s, c=pred, cmap=colormap_show, marker=marker)
    if show_graph:
        Pos = G.pos.clone().detach().cpu().numpy()
        Edges = G.edge_index.clone().detach().cpu().numpy()
        lines = []
        for i in range(Edges.shape[1]):
            a, b = Edges[0][i], Edges[1][i]
            lines.append([(Pos[a,0],Pos[a,1]),(Pos[b,0],Pos[b,1])])
        lc = mc.LineCollection(lines, linewidths=1, colors='g')
        plt.gca().add_collection(lc)
    plt.colorbar()
    if not show:
        plt.gcf()
        plt.savefig(f"result_images/{label}_pred_mean.png")
        plt.cla()
    
    plt.figure(f"{label}_true_error")
    plt.scatter(x, y, s=size_s, c=error, cmap=colormap_show, marker=marker)
    plt.colorbar()
    if not show:
        plt.gcf()
        plt.savefig(f"result_images/{label}_true_error.png")
        plt.cla()

    plt.figure(f"{label}_relative_error")
    plt.scatter(x, y, s=size_s, c=error/r, cmap=colormap_show, marker=marker)
    plt.colorbar()
    if not show:
        plt.gcf()
        plt.savefig(f"result_images/{label}_relative_error.png")
        plt.cla()
    
    plt.figure(f"{label}_truth")
    plt.scatter(x, y, s=size_s, c=truth, cmap=colormap_show, marker=marker)
    plt.colorbar()
    if not show:
        plt.gcf()
        plt.savefig(f"result_images/{label}_true_error.png")
        plt.cla()
    else:
        plt.show()


if __name__ == '__main__':
    paser = argparse.ArgumentParser()
    paser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="path to the checkpoint",
    )

    paser.add_argument(
        "--exp-name",
        type=str,
        required=True,
        help="type of the graph given by the experiment number",
    )

    paser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="path to the dataset",
    )

    args = paser.parse_args()

    print(f"exp {args.exp}", end="\n")
    test_all(args)
    
    