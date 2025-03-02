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


torch.backends.cudnn.benchmark = False 
torch.backends.cudnn.deterministic = True 


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


def _test_all(args):
    # global model_dir, exp
    exp = args.exp_name
    model_dir = args.ckpt

    cuda = torch.cuda.is_available()
    # cuda = False
    device = torch.device('cuda') if cuda else torch.device('cpu')
    # device = torch.device('cpu')
    data_dir = "datasets/heat/dataGNN.mat"
    
    k = 32
    encoders = nn.ModuleList([Net(dims=[3, k, k, k])])
    decoders = nn.ModuleList([Net(dims=[k+2, k, k, 1])])

    net = GENSoftNN(encoders=encoders, decoders=decoders, exp=exp)
    if cuda:
        net.cuda()
    # for param_tensor in net.state_dict():
    #     print(param_tensor, "\t", net.state_dict()[param_tensor].size())
    # print()
    state_dict = torch.load(model_dir)
    net.load_state_dict(state_dict)
    net.eval()

    file_args = {'file_path': args.data_path}
    file_args1 = {'file_path': args.data_path}
    
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
            # 加载实验的图片，实验顺序按word中的来
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
        
        for cnt, it in tqdm(enumerate(test_loaders), total=len(test_loaders)):
            # 对测试集中所有的结果进行测试
            in_i = 0
            for ((Inp,Out), idx) in it:
                index = cnt % room
                # enumerate 0, 1, ..., 32
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

                test_inp = Inp[0][1][index, :, :]
                Inp[0][1] = test_inp.repeat(room, 1, 1)
                
                preds = net(Inp, Q, G=graph)
                test_truth = targets[0][index, :, :]
                pred_one = torch.mean(preds[0], dim=0)
                metrics.update(pred_one, test_truth)
                in_i += 1

    # print(in_i)
    t_err, v_err, _ = metrics.get()
    print(f't: mean relative error is {t_err}')
    print(f'v: mean relative error is {v_err}')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="path to the checkpoint",
    )

    parser.add_argument(
        "--exp-name",
        type=str,
        required=True,
        help="type of the graph given by the experiment number",
    )

    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="path to the dataset",
    )

    args = parser.parse_args()

    print(f"exp {args.exp_name}", end="\n")
    _test_all(args)
    
    