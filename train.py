import time
import os
from datetime import datetime
from tqdm import tqdm as Tqdm
from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
from gen_datasets import FTDataset
from my_dataset_64 import HeatInpDataset, \
        HeatOutDataset
from poisson_square_experiments_utils import *
from neural_processes import NeuralProcesses
from GEN import GEN
from torch.utils import data
from gen_softnn import GENSoftNN
from utils import Net
from design_utils import create_new_mesh_list_1, create_new_mesh_list_heat
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import randperm
import warnings
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="path to the dataset",
    )
    
    torch.manual_seed(0)
    cuda = torch.cuda.is_available()
    device = torch.device('cuda') if cuda else torch.device('cpu')
    model_type = ['GENSoftNN', 'GENPlanarGrid', 'NP'][0]
    bs = 8
    k = 32
    
    _lambda = 10
    
    file_args = {'file_path' : args.data_path}
    file_args1 = {'file_path' : args.data_path}
    #
    node_train = 16 
    total_epoch = 3000

    sqrt_num_nodes_list = [16]

    copies_per_graph = 1
    less_loss = float('inf')
    opt_nodes = False
    slow_opt_nodes = False 
    do_tensorboard = True

    if model_type == 'NP':
        opt_nodes = False
    if not opt_nodes: slow_opt_nodes = False
    full_dataset = FTDataset(inp_datasets=[HeatInpDataset],
            inp_datasets_args = [file_args],    
            out_datasets = [HeatOutDataset],
            out_datasets_args = [file_args1],   
            idx_list=None)
    
    train_size = int(0.8*len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = data.random_split(full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=6,
            shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset,  batch_size=1, num_workers=6,
            shuffle=True, drop_last=False)

    encoders = nn.ModuleList([Net(dims=[3, k, k, k])])
    decoders = nn.ModuleList([Net(dims=[k+2, k, k, 2])])

    loss_fn = nn.MSELoss()

    if model_type == 'NP':
        model = NeuralProcesses(encoders, decoders)
        mesh_list = mesh_params = [[None] for _ in range(len(full_dataset))]
    else:
        assert min(sqrt_num_nodes_list) >= 1
        if model_type == 'GENSoftNN':
            model = GENSoftNN(encoders=encoders, decoders=decoders)
        else: raise NotImplementedError
        mesh_list, mesh_params, num_nodes_list = create_new_mesh_list_heat(
                num_datasets=len(full_dataset),
                sqrt_num_nodes_list=sqrt_num_nodes_list,
                initialization='random' if opt_nodes else 'uniform',
                copies_per_graph=copies_per_graph, device=device, perturb=0.001)
    max_mesh_list_elts = max([len(aux) for aux in mesh_list])
    if cuda: model.cuda()
    opt = torch.optim.Adam(params=model.parameters(), lr=3e-3)
    if model_type == 'NP':mesh_opt = None

    mesh_opt = None
    time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    if do_tensorboard: writer = SummaryWriter(log_dir=f"runs/{time_now}")
    else: writer = None

    save_dir = f"heat_models/v{_lambda}/{time_now}"
    if not isinstance(loss_fn, nn.MSELoss):
        save_dir = f"relative_loss_heat_models/{time_now}"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in Tqdm(range(total_epoch)):
        train_loss = 0. ;  test_loss = 0.
        train_graphs = 0 ; test_graphs = 0
        if model_type == 'NP': # A NP is equivalent to a GEN with 1 node
            train_loss_summ = {1:[0,0]}
            test_loss_summ = {1: [0,0]}
            pos_change_summ = {1: [0,0]}
        else:
            train_loss_summ = {num:[0,0] for num in num_nodes_list}
            test_loss_summ = {num:[0,0] for num in num_nodes_list}
            pos_change_summ = {num:[0,0] for num in num_nodes_list}
        for g_idx in range(max_mesh_list_elts):
            for cnt, ((Inp,Out),idx) in enumerate(train_loader):
                if len(mesh_list[idx]) <= g_idx: continue
                G = mesh_list[idx][g_idx]
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
                train_graphs += 1
                if model_type == 'NP': preds = model(Inp, Q)
                else:
                    preds = model(Inp, Q, G=G) 
                if slow_opt_nodes:
                    exec_losses = [loss_fn(pred[node_train:],
                        target[node_train:]).unsqueeze(0)
                        for (pred, target) in zip(preds, targets)]
                    loss = torch.sum(torch.cat(exec_losses))
                    loss.backward(retain_graph=True)

                    finetune_losses = [loss_fn(pred[:node_train],
                        target[:node_train]).unsqueeze(0)
                        for (pred, target) in zip(preds, targets)]
                    finetune_loss = torch.sum(torch.cat(finetune_losses))
                    mesh_opt.zero_grad()
                    finetune_loss.backward()
                    mesh_opt.step()
                    
                    graph_update_meshes_after_opt(mesh_list[idx][g_idx],
                            epoch=epoch, writer=writer)
                else:
                    losses = [loss_fn(pred, target * _lambda).unsqueeze(0)
                        for (pred, target) in zip(preds, targets)]
                    loss = torch.sum(torch.cat(losses))
                    loss.backward()
                train_loss += loss.item()
                num_nodes = 0 if model_type== 'NP' else G.num_nodes
                train_loss_summ[num_nodes][0] += loss.item()
                train_loss_summ[num_nodes][1] += 1
                if model_type != 'NP':
                    pos_change_summ[num_nodes][0] += (
                            torch.max(torch.abs(G.pos - G.ini_pos)).item())
                    pos_change_summ[num_nodes][1] += 1
                if (cnt % bs == bs-1) or (cnt == len(train_loader)-1):
                    opt.step()
                    opt.zero_grad()
        if do_tensorboard:
            it_list = [1] if model_type=='NP' else num_nodes_list
            for num in it_list:
                writer.add_scalar('train/loss-'+str(num),
                        train_loss_summ[num][0]/train_loss_summ[num][1],
                        epoch)

        for cnt, ((Inp,Out),idx) in enumerate(test_loader):
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
            for g_idx, G in enumerate(mesh_list[idx]):
                if model_type == 'NP': preds = model(Inp, Q)
                else: preds = model(Inp, Q, G=G)
                if opt_nodes:
                    finetune_losses = [loss_fn(pred[:node_train],
                        target[:node_train]).unsqueeze(0)
                        for (pred, target) in zip(preds, targets)]
                    finetune_loss = torch.sum(torch.cat(finetune_losses))
                    exec_losses = [loss_fn(pred[node_train:],
                        target[node_train:]).unsqueeze(0)
                        for (pred, target) in zip(preds, targets)]
                    exec_loss = torch.sum(torch.cat(exec_losses))
                    finetune_loss.backward()
                    loss = exec_loss
                else:
                    losses = [loss_fn(pred, target * _lambda).unsqueeze(0)
                        for (pred, target) in zip(preds, targets)]
                    loss = torch.sum(torch.cat(losses))
                test_loss += loss.item()
                test_graphs += 1
                num_nodes = 0 if model_type == 'NP' else G.num_nodes
                test_loss_summ[num_nodes][0] += loss.item()
                test_loss_summ[num_nodes][1] += 1
        

        opt.zero_grad() 
        if mesh_opt is not None:
            mesh_opt.step()
            mesh_opt.zero_grad()
            update_meshes_after_opt(mesh_list, epoch=epoch, writer=writer)
        if do_tensorboard:
            it_list = [1] if model_type=='NP' else num_nodes_list
            for num in it_list:
                writer.add_scalar('test-loss-'+str(num),
                        test_loss_summ[num][0]/test_loss_summ[num][1], epoch)
        

                if opt_nodes:
                    writer.add_scalar('pos_change-'+str(num**2),
                            pos_change_summ[num**2][0]/pos_change_summ[num**2][1],
                            epoch)
            print(f'\ntest_loss_summ is {test_loss_summ}\ntest_loss is {test_loss}\n')
            if train_loss < less_loss:
                less_loss = train_loss
                torch.save(model.state_dict(), f'{save_dir}/local_optium_model.pkl')

        else:
            if train_loss < less_loss:
                less_loss = train_loss
                torch.save(model, f'{save_dir}/local_optium_model.pkl')
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    torch.save(model.state_dict(), f'{save_dir}/final_model.pkl')

