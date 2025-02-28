import torch
from create_graph import GraphLessMsgStep


def create_new_mesh_list_1(num_datasets, sqrt_num_nodes_list=[3],
        initialization='uniform', copies_per_graph=1, device='cpu', perturb=0.):
    graph = GraphLessMsgStep(data_idx=-1).train_polygon_mesh(device=device)
    num_nodes_p = graph.num_nodes
    print(f'nodes num is: {num_nodes_p}')
    print(f'message step = {graph.msg_steps}')
    L = []
    param_list = torch.nn.ParameterList()
    for dataset in range(num_datasets):
        aux = []
        for c in range(copies_per_graph):
            for (e_nn, num_nodes) in enumerate(sqrt_num_nodes_list):
                data_idx = (dataset, c*len(sqrt_num_nodes_list)+e_nn)
                if initialization == 'uniform':
                    aux.append(GraphLessMsgStep(data_idx=data_idx).train_polygon_mesh(device=device))
                else: raise NotImplementedError
        L.append(aux)
    return L, param_list, [num_nodes_p]


def create_new_mesh_list_heat(num_datasets, sqrt_num_nodes_list=[3],
        initialization='uniform', copies_per_graph=1, device='cpu', perturb=0., exp="", size=None):
    if exp == "1":
        graph = GraphLessMsgStep(data_idx=-1).train_heat_graph(device=device)
    elif exp == "2":
        graph = GraphLessMsgStep(data_idx=-1).train_heat_graph_2(device=device)
    elif exp == "3":
        if size == None:
            graph = GraphLessMsgStep(data_idx=-1).train_heat_graph_3(device=device)
        else:
            graph = GraphLessMsgStep(data_idx=-1).train_heat_graph_3(device=device, size=size)
    elif exp == "4":
        graph = GraphLessMsgStep(data_idx=-1).train_heat_graph_4(device=device)
    elif exp == "5":
        graph = GraphLessMsgStep(data_idx=-1).train_heat_graph_5(device=device)
    elif exp == "6":
        graph = GraphLessMsgStep(data_idx=-1).train_heat_graph_6(device=device)
    else:
        raise NotImplementedError()
    
    num_nodes_p = graph.num_nodes
    print(f'nodes num is: {num_nodes_p}')
    print(f'message step = {graph.msg_steps}')
    L = []
    param_list = torch.nn.ParameterList()
    for dataset in range(num_datasets):
        aux = []
        for c in range(copies_per_graph):
            for (e_nn, num_nodes) in enumerate(sqrt_num_nodes_list):
                data_idx = (dataset, c*len(sqrt_num_nodes_list)+e_nn)
                if initialization == 'uniform':
                    if exp == "":
                        aux.append(GraphLessMsgStep(data_idx=data_idx).train_heat_graph(device=device))
                    else:
                        if exp == "1":
                            aux.append(GraphLessMsgStep(data_idx=data_idx).train_heat_graph(device=device))
                        elif exp == "2":
                            aux.append(GraphLessMsgStep(data_idx=data_idx).train_heat_graph_2(device=device))
                        elif exp == "3":
                            if size == None:
                                aux.append(GraphLessMsgStep(data_idx=-1).train_heat_graph_3(device=device))
                            else:
                                aux.append(GraphLessMsgStep(data_idx=-1).train_heat_graph_3(device=device, size=size))
                        elif exp == "4":
                            aux.append(GraphLessMsgStep(data_idx=data_idx).train_heat_graph_4(device=device))
                        elif exp == "5":
                            aux.append(GraphLessMsgStep(data_idx=data_idx).train_heat_graph_5(device=device))
                        elif exp == "6":
                            aux.append(GraphLessMsgStep(data_idx=data_idx).train_heat_graph_6(device=device))
                        else:
                            raise NotImplementedError()
                else: raise NotImplementedError
        L.append(aux)
    return L, param_list, [num_nodes_p]
