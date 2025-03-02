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
from gen_softnn import GENSoftNN, GENSoftNNVpred
from utils import Net
from design_utils import create_new_mesh_list_1, create_new_mesh_list_heat
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, CyclicLR, StepLR
from torch import randperm
import warnings
import argparse


if __name__ == '__main__':
    # torch.manual_seed(0)
    parser = argparse.ArgumentParser()
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
    
    cuda = torch.cuda.is_available()
    device = torch.device('cuda') if cuda else torch.device('cpu')
    model_type = ['GENSoftNN', 'GENPlanarGrid', 'NP'][0]
    
    exp = args.exp_name
        
    print(f"exp is {exp}\n")
    bs = 8
    k = 32
    
    file_args = {'file_path' : args.data_path}
    file_args1 = {'file_path' : args.data_path}

    node_train = 16 
    total_epoch = 10000

    sqrt_num_nodes_list = [16]

    # ratio = 5

    copies_per_graph = 1
    less_loss = float('inf')
    opt_nodes = False
    slow_opt_nodes = False # Train node_pos only in part of each "house" data;slower
    do_tensorboard = True
    # Changed the random initialization because GeneralizedHalton
    # doesn't install well on a Docker. We use another simple random initialization.

    if model_type == 'NP':
        opt_nodes = False
    if not opt_nodes: slow_opt_nodes = False
    full_dataset = FTDataset(inp_datasets=[HeatInpDataset],
            inp_datasets_args = [file_args],    # DBR: new file direction
            out_datasets = [HeatOutDataset],
            out_datasets_args = [file_args1],   # DBR: new file direction
            idx_list=None)
    
    train_size = int(0.8*len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = data.random_split(full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=0,
            shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset,  batch_size=1, num_workers=0,
            shuffle=True, drop_last=False)

    encoders = nn.ModuleList([Net(dims=[3, k, k, k])])
    decoders = nn.ModuleList([Net(dims=[k+2, k, k, 1])])

    loss_fn = nn.MSELoss()
    # loss_fn = log_loss
    if model_type == 'NP':
        model = NeuralProcesses(encoders, decoders)
        mesh_list = mesh_params = [[None] for _ in range(len(full_dataset))]
    else:
        assert min(sqrt_num_nodes_list) >= 1
        if model_type == 'GENSoftNN':
            model = GENSoftNN(encoders=encoders, decoders=decoders, exp=exp)
        else: raise NotImplementedError
        mesh_list, mesh_params, num_nodes_list = create_new_mesh_list_heat(
                num_datasets=len(full_dataset),
                sqrt_num_nodes_list=sqrt_num_nodes_list,
                initialization='random' if opt_nodes else 'uniform',
                copies_per_graph=copies_per_graph, device=device, perturb=0.001, exp=exp)
    max_mesh_list_elts = max([len(aux) for aux in mesh_list])
    if cuda: model.cuda()
    opt = torch.optim.Adam(params=model.parameters(), lr=3e-3)
    if model_type == 'NP':mesh_opt = None
#    else: mesh_opt = torch.optim.Adam(params=mesh_params, lr=3e-4)
    mesh_opt = None
    time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    if do_tensorboard: writer = SummaryWriter(log_dir=f"runs/{time_now}")
    else: writer = None 
    
    lr_scheduler = StepLR(opt, step_size=15000, gamma=0.9)
    save_dir = f"exp_{exp}_transform_room_independent_model/{time_now}"
    os.makedirs(save_dir, exist_ok=True)
    
    # if exp == "2":
    #     state_dict = torch.load(r"exp_4_transform_room_independent_model\2023-06-18_16-01-41\final_model.pkl")
    #     model.load_state_dict(state_dict)

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
                    preds = model(Inp, Q, G=G) # 计算网络的预测值
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
                    # project back to square
                    graph_update_meshes_after_opt(mesh_list[idx][g_idx],
                            epoch=epoch, writer=writer)
                else:
                    losses = [loss_fn(pred, target).unsqueeze(0)
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
            # print(f"\ntrain loss is {train_loss_summ}\ntrain_loss is {train_loss}\n")
# test below
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
                    losses = [loss_fn(pred, target).unsqueeze(0)
                        for (pred, target) in zip(preds, targets)]
                    loss = torch.sum(torch.cat(losses))
                test_loss += loss.item()
                test_graphs += 1
                num_nodes = 0 if model_type == 'NP' else G.num_nodes
                test_loss_summ[num_nodes][0] += loss.item()
                test_loss_summ[num_nodes][1] += 1
        
        # lr_scheduler.step(test_loss)
        lr_scheduler.step()
        print(f"lr is {lr_scheduler._last_lr}")
        opt.zero_grad() #Don't train Theta on finetune test set when optmizing nodes
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
                # for param_tensor in model.state_dict():
                #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
                # print("save")
        else:
            # print(round(train_loss/(max_mesh_list_elts * train_size), 3),
            #     round(test_loss/(max_mesh_list_elts * test_size), 3))
            if train_loss < less_loss:
                less_loss = train_loss
                torch.save(model, f'{save_dir}/local_optium_model.pkl')
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    torch.save(model.state_dict(), f'{save_dir}/final_model.pkl')


