import numpy as np
import torch
from torch_geometric.data import Data
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from torch.autograd import Variable
import time
from scipy.io import loadmat
from create_graph import GraphCreater


def create_uniform_grid_dbr(n, device='cpu', perturb=0., learnable=False, data_idx=-1):
    node_pos = []
    for i in range(n):
        for j in range(n):
            x = (i*(i+1.)/128.) * (i <= 8) + ((9/16.)+(23-i)*(i-8)/128.) * (i > 8)
            y = (j*(j+1.)/128.) * (j <= 8) + ((9/16.)+(23-j)*(j-8)/128.) * (j > 8)
            node_pos.append([x, y])
    node_pos = torch.FloatTensor(np.stack(node_pos, 0)).cuda()
    if learnable: node_pos = torch.nn.Parameter(node_pos)
    edges = []
    N = n*n
    data = Data()
    data.data_idx = data_idx
    data.num_nodes = N
    data.pos = node_pos
    data.grid = {'X': 1, 'Y': 1, 'min_X' : 0, 'min_Y': 0,
            'dx':1/(n-1), 'dy':1/(n-1), 'n_x': n, 'n_y': n}
    data.msg_steps = 2*n-1
    if perturb == 0.:
        for i in range(n):
            for j in range(n):
                if i:
                    edges.append(torch.LongTensor([n*(i-1)+j, n*i+j]))
                    edges.append(torch.LongTensor([n*i+j, n*(i-1)+j]))
                if j:
                    edges.append(torch.LongTensor([n*i+(j-1), n*i+j]))
                    edges.append(torch.LongTensor([n*i+j, n*i+(j-1)]))
        data.edge_index = torch.stack(edges).t().contiguous()
    else:
        graph_update_meshes_after_opt(data)
    data.ini_pos = node_pos.clone()
    data = data.to(device)
    return data


def create_ghalton_grid(n, gh, device='cpu', learnable=False, data_idx=-1):
    '''
    Creates randomly positioned nodes.
    '''
    ##################################################################
    ## CURRENTLY UNUSED BECAUSE IT DOESN'T INSTALL WELL ON A DOCKER ##
    ## WE INSTEAD USE perturb > 0 WITH create_uniform_grid.         ##
    ##################################################################
    N = n*n
    #Get nodes positions
    pos = gh.get(N)
    node_positions = []
    for r in range(N):
      node_positions.append(np.array([
        data.grid['min_X'] + pos[r][0]*data.grid['X'],
        data.grid['min_Y'] + pos[r][1]*data.grid['Y']]))

    if learnable:
        node_pos = torch.nn.Parameter(torch.FloatTensor(
            np.vstack(node_positions)).to(device=device))
    else:
        node_pos = torch.FloatTensor(
                np.vstack(node_positions)).to(device=device)

    data = Data()
    data.data_idx = data_idx
    data.pos = node_pos
    data.num_nodes = N
    data.grid = {'X': 1, 'Y': 1, 'min_X' : 0, 'min_Y': 0,
            'dx':1/(n-1), 'dy':1/(n-1), 'n_x': n, 'n_y': n}
    data.msg_steps = 2*n-1
    compute_Delaunay_edges(data)
    data.ini_pos = node_pos.clone()
    data = data.to(device)
    return data


def create_uniform_grid(n, device='cpu', perturb=0.,
        learnable=False, data_idx=-1):
    node_pos = []
    for i in range(n):
        for j in range(n):
            x = i/(n-1.)
            y = j/(n-1.)
            node_pos.append([x, y])
    node_pos = torch.FloatTensor(np.stack(node_pos, 0)).cuda()
    if learnable: node_pos = torch.nn.Parameter(node_pos)
    edges = []
    N = n*n
    data = Data()
    data.data_idx = data_idx
    data.num_nodes = N
    data.pos = node_pos
    data.grid = {'X': 1, 'Y': 1, 'min_X' : 0, 'min_Y': 0,
            'dx':1/(n-1), 'dy':1/(n-1), 'n_x': n, 'n_y': n}
    data.msg_steps = 2*n-1
    if perturb == 0.:
        for i in range(n):
            for j in range(n):
                if i:
                    edges.append(torch.LongTensor([n*(i-1)+j, n*i+j]))
                    edges.append(torch.LongTensor([n*i+j, n*(i-1)+j]))
                if j:
                    edges.append(torch.LongTensor([n*i+(j-1), n*i+j]))
                    edges.append(torch.LongTensor([n*i+j, n*i+(j-1)]))
        data.edge_index = torch.stack(edges).t().contiguous()
    else:
        graph_update_meshes_after_opt(data)
    data.ini_pos = node_pos.clone()
    data = data.to(device)
    return data


def create_mesh_list(num_datasets, sqrt_num_nodes_list=[3],
        initialization='uniform', copies_per_graph=1, device='cpu', perturb=0.):
    L = []
    param_list = torch.nn.ParameterList()
    for dataset in range(num_datasets):
        aux = []
        for c in range(copies_per_graph):
            for (e_nn, num_nodes) in enumerate(sqrt_num_nodes_list):
                data_idx = (dataset, c*len(sqrt_num_nodes_list)+e_nn)
                if initialization == 'uniform':
                    aux.append(create_uniform_grid(num_nodes, device=device,
                        data_idx=data_idx, perturb=perturb))
                    # variable = Variable(aux[-1].pos, requires_grad=True)
                    # param_list.append(variable)
                elif initialization == 'random':
                    aux.append(create_uniform_grid(num_nodes, device=device,
                        perturb=0.2/num_nodes, learnable=True,
                        data_idx=data_idx))
                    param_list.append(aux[-1].pos)
                else: raise NotImplementedError
        L.append(aux)
    return L, param_list

def compute_Delaunay_edges(G):
    '''
    Computes the Delaunay triangulation to get the edges given a grid.
    '''
    if G.num_nodes < 3: #not enough for Delaunay
        raise NotImplementedError("Graph with less than 3 nodes are not implemented")
    
    points = G.pos.detach().cpu().numpy()
    G.Delaunay_triangles = (
        Delaunay(points, qhull_options='QJ Pp').simplices)
    aux_map = {} #from ordered edge to complimentary vertex in triangle
    for i in range(G.Delaunay_triangles.shape[0]):
      v = G.Delaunay_triangles[i].tolist()
      for j in range(3):
        key = str(min(v[j], v[(j+1)%3]))+ '_'+str(max(v[j],v[(j+1)%3]))
        if key in aux_map: aux_map[key].append(v[(j+2)%3])
        else: aux_map[key] = [v[(j+2)%3]]
    edges = []
    for s in aux_map:
      a,b = s.split('_')
      a = int(a) ; b = int(b)
      edges.append(torch.LongTensor([a, b]))
      edges.append(torch.LongTensor([b, a]))
    if G.pos.is_cuda:
        G.edge_index = torch.stack(edges).t().contiguous().cuda()
    else: G.edge_index = torch.stack(edges).t().contiguous()

def show_structure_mesh(G, writer=None, epoch=0):
    '''
    Shows mesh on top of image
    '''
    try:
        path_to_image = './3200_electric_data/result/'+ str(G.data_idx[0]) +'.mat'
        values = loadmat(path_to_image)['V'].astype(np.float32)
    except:
        print('Image not found')
        return
    if len(values.shape) == 3 and values.shape[-1] not in [1,3]:
      values = values[0] #multiple images
    fig = plt.figure()
    plt.imshow(values)
    Pos = G.pos.clone().detach().cpu().numpy()*100
    Edges = G.edge_index.clone().detach().cpu().numpy()
    lines = []
    for i in range(Edges.shape[1]):
        a, b = Edges[0][i], Edges[1][i]
        lines.append([(Pos[a,0],Pos[a,1]),(Pos[b,0],Pos[b,1])])
    lc = mc.LineCollection(lines, linewidths=3, colors='w')
    plt.gca().add_collection(lc)
    if writer is None:
        pass
    else:
        writer.add_figure('node_pos/'+str(G.num_nodes)+'/'+
                str(G.data_idx[1])+'/'+str(G.data_idx[0]),
                fig, epoch)
    plt.clf()

def graph_update_meshes_after_opt(G, writer=None, epoch=None):
    #Clip node positions to fall inside grid
    G.pos.data.clamp_(G.grid['min_X'], G.grid['min_X']+G.grid['X'])
    compute_Delaunay_edges(G)

def update_meshes_after_opt(L, writer=None, epoch=None):
    for l in L:
        for G in l:
            graph_update_meshes_after_opt(G, writer=writer, epoch=epoch)


def create_new_mesh_list(num_datasets, sqrt_num_nodes_list=[3],
        initialization='uniform', copies_per_graph=1, device='cpu', perturb=0.):
    graph = GraphCreater(data_idx=-1).train_polygon_mesh(device=device)
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
                    aux.append(GraphCreater(data_idx=data_idx).train_polygon_mesh(device=device))
                elif initialization == 'random':
                    aux.append(create_uniform_grid(num_nodes, device=device,
                        perturb=0.2/num_nodes, learnable=True,
                        data_idx=data_idx))
                    param_list.append(aux[-1].pos)
                else: raise NotImplementedError
        L.append(aux)
    return L, param_list, [num_nodes_p]
    