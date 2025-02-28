
import pygmsh
from pygmsh.occ.geometry import Disk
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from matplotlib import collections as mc
from torch_geometric.data import Data
from typing import List, Tuple
from scipy.io import loadmat


class GraphCreater:
    def __init__(self, data_idx=-1) -> None:
        self.data = Data()
        self.data.data_idx = data_idx
        self.data.grid = {'X': 1, 'Y': 1, 'min_X' : 0, 'min_Y': 0,
            'dx':None, 'dy':None, 'n_x': None, 'n_y': None}
        self.data.num_nodes = None
        self.max_area_meshpy = 0.004
        self.max_area_pygmsh = 0.08

    def __repr__(self) -> str:
        return f'<node num in graph is {self.data.num_nodes}>'

    @classmethod
    def rotate_xy(cls, points):
        degree = np.pi / 2
        x = points[:, 0]
        y = points[:, 1]
        new_points = np.zeros(points.shape[:2])
        new_points[:, 0] = np.sin(degree) * (x-y)
        new_points[:, 1] = np.sin(degree) * (x+y)
        return new_points

    @classmethod
    def connect_triangle(cls, node_pos, cells):
        edges = []
        for cell in cells:
            n1 = cell[0]
            n2 = cell[1]
            n3 = cell[2]
            edges.append(torch.LongTensor([n1, n2]))
            edges.append(torch.LongTensor([n2, n1]))
            edges.append(torch.LongTensor([n1, n3]))
            edges.append(torch.LongTensor([n3, n1]))
            edges.append(torch.LongTensor([n2, n3]))
            edges.append(torch.LongTensor([n3, n2]))
        return edges

    @classmethod
    def regular_points(cls, points):
        points_new = np.zeros(points[:, :2].shape)
        max_x = max(points[:, 0])
        min_x = min(points[:, 0])
        max_y = max(points[:, 1])
        min_y = min(points[:, 1])
        delta_x = max_x - min_x
        delta_y = max_y - min_y
        points_new[:, 0] = (points[:, 0] - min_x) / delta_x
        points_new[:, 1] = (points[:, 1] - min_y) / delta_y
        return points_new

    @classmethod
    def compute_Delaunay_edges(cls, G):
        '''
        Computes the Delaunay triangulation to get the edges given a grid.
        '''
        if G.num_nodes < 3: #not enough for Delaunay
            raise ValueError("the point is not enough for the delaunay edge")
        points = G.pos.detach().cpu().numpy()
        G.Delaunay_triangles = (
            Delaunay(points, qhull_options='QJ Pp').simplices
            )
        aux_map = {} #from ordered edge to complimentary vertex in triangle
        for i in range(G.Delaunay_triangles.shape[0]):
            v = G.Delaunay_triangles[i].tolist()
            for j in range(3):
                key = str(min(v[j], v[(j+1)%3])) +  '_' + str(max(v[j],v[(j+1)%3]))
                if key in aux_map: 
                    aux_map[key].append(v[(j+2)%3])
                else: 
                    aux_map[key] = [v[(j+2)%3]]
            edges = []
        for s in aux_map:
            a,b = s.split('_')
            a = int(a) ; b = int(b)
            edges.append(torch.LongTensor([a, b]))
            edges.append(torch.LongTensor([b, a]))
        if G.pos.is_cuda:
            G.edge_index = torch.stack(edges).t().contiguous().cuda()
        else: 
            G.edge_index = torch.stack(edges).t().contiguous()
        return
    
    @property
    def msg_steps(self) -> int:
        value = round(np.sqrt(self.data.num_nodes))*2 -1 
        return value

    def pygmsh_polygen(self, device='cpu'):
        with pygmsh.geo.Geometry() as geom:
            # add_polygon
            geom.add_polygon(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 1.0]
                ],
                mesh_size = self.max_area_pygmsh,
            )
            mesh = geom.generate_mesh()
            points = mesh.points
            tri_cells = mesh.cells_dict['triangle']
            line_cells = mesh.cells_dict['line']
            node_pos = points[:, :2]
            node_pos = torch.FloatTensor(np.stack(node_pos, 0)).cuda()
            self.data.num_nodes = len(points)
            data = self.data
            data.pos = node_pos
            data.msg_steps = self.msg_steps
            data.ini_pos = node_pos.clone()
            edges = self.connect_triangle(node_pos, tri_cells)
            for line in line_cells:
                edges.append(torch.LongTensor([line[0], line[1]]))
                edges.append(torch.LongTensor([line[0], line[1]]))

            data = data.to(device)
            data.edge_index = torch.stack(edges).t().contiguous()
        return data

    @classmethod
    def get_poly_coorinate(cls) -> List[Tuple[float, float]]:
        points = []
        x1_s = [75.0, 0.0, 0.0, 75.0]
        x2_s = [175.0, 250.0, 250.0, 175.0]
        y1 = 0.0
        k = 0
        height = [40.0, 30.0, 40.0, 40.0, 40.0, 50.0, 40.0, 40.0, 40.0, 30.0, 30.0]
        
        for step in range(48):
            points.append([x1_s[step%4], y1])
            if step%4 == 2:
                y1 += 10.0
            if step%4 == 0 and step != 0:
                y1 += height[k]
                k += 1

        k = len(height) - 1

        for step in range(48):
            points.append([x2_s[step%4], y1])
            if step%4 == 2:
                y1 -= 10.0
            if step%4 == 0 and step != 0:
                y1 -= height[k]
                k -= 1

        return points

    def train_polygon_mesh(self, device=torch.device('cuda')):
        maxium_size = 0.1
        with pygmsh.geo.Geometry() as geom:
            geom.add_polygon(    
                [
                    [0.0, 0.0],
                    [0.0, 0.3],
                    [0.75, 0.5],
                    [0.75, 1.0],
                    [0.0, 1.2],
                    [0.0, 1.5],
                    [2.0, 1.5],
                    [2.0, 1.2],
                    [1.25, 1.0],
                    [1.25, 0.5],
                    [2.0, 0.3],
                    [2.0, 0.0]
                ],
                mesh_size = maxium_size,
            )
            mesh = geom.generate_mesh()
            points = mesh.points
            tri_cells = mesh.cells_dict['triangle']
            line_cells = mesh.cells_dict['line']
            node_pos = points[:, :2]
            node_pos = torch.FloatTensor(np.stack(node_pos, 0)).cuda()
            self.data.num_nodes = len(points)
            data = self.data
            data.pos = node_pos
            data.msg_steps = self.msg_steps
            data.ini_pos = node_pos.clone()
            edges = self.connect_triangle(node_pos, tri_cells)
            for line in line_cells:
                edges.append(torch.LongTensor([line[0], line[1]]))
                edges.append(torch.LongTensor([line[0], line[1]]))

            data.edge_index = torch.stack(edges).t().contiguous()
            data = data.to(device)
        return data

    def train_square_without_circle(self, device=torch.device('cuda')):
        maxium_size = 0.1
        flag = False
        with pygmsh.occ.Geometry() as geom:
            geom.characteristic_length_max = maxium_size
            circle = geom.add_disk([0.75, 0.75], 0.5)
            rec = geom.add_rectangle([0.0, 0.0, 0.0], 1.5, 1.5)
            geom.boolean_difference(rec, circle)
           
            mesh = geom.generate_mesh()
            points = mesh.points
            if mesh.cells_dict.get('triangle', None).any():
                tri_cells = mesh.cells_dict['triangle']
                flag = True
            if mesh.cells_dict.get('line', None).any():
                line_cells = mesh.cells_dict['line']
                flag = True
            if not flag:
                raise ValueError("no edge")
            node_pos = points[:, :2]
            node_pos = torch.FloatTensor(np.stack(node_pos, 0)).cuda()
            self.data.num_nodes = len(points)
            data = self.data
            data.pos = node_pos
            data.msg_steps = self.msg_steps
            data.ini_pos = node_pos.clone()
            edges = self.connect_triangle(node_pos, tri_cells)
            for line in line_cells:
                edges.append(torch.LongTensor([line[0], line[1]]))
                edges.append(torch.LongTensor([line[0], line[1]]))

            data.edge_index = torch.stack(edges).t().contiguous()
            data = data.to(device)
        return data

    def train_circle(self, device=torch.device('cuda')):
        maxium_size = 0.1
        with pygmsh.geo.Geometry() as geom:
            geom.add_circle([0.75, 0.75], 0.75, maxium_size)
            mesh = geom.generate_mesh()
            points = mesh.points
            tri_cells = mesh.cells_dict['triangle']
            line_cells = mesh.cells_dict['line']
            node_pos = points[:, :2]
            node_pos = torch.FloatTensor(np.stack(node_pos, 0)).cuda()
            self.data.num_nodes = len(points)
            data = self.data
            data.pos = node_pos
            data.msg_steps = self.msg_steps
            data.ini_pos = node_pos.clone()
            edges = self.connect_triangle(node_pos, tri_cells)
            for line in line_cells:
                edges.append(torch.LongTensor([line[0], line[1]]))
                edges.append(torch.LongTensor([line[0], line[1]]))

            data.edge_index = torch.stack(edges).t().contiguous()
            data = data.to(device)
        return data

    def train_oval(self, device=torch.device('cuda')):
        maxium_size = 0.1
        flag = False
        with pygmsh.occ.Geometry() as geom:
            geom.add_disk([0.75, 0.75], 0.75, 0.5, mesh_size=maxium_size)
            mesh = geom.generate_mesh()
            points = mesh.points
            if mesh.cells_dict.get('triangle', None).any():
                tri_cells = mesh.cells_dict['triangle']
                flag = True
            if mesh.cells_dict.get('line', None).any():
                line_cells = mesh.cells_dict['line']
                flag = True
            if not flag:
                raise ValueError("no edge")
            node_pos = points[:, :2]
            node_pos = torch.FloatTensor(np.stack(node_pos, 0)).cuda()
            self.data.num_nodes = len(points)
            data = self.data
            data.pos = node_pos
            data.msg_steps = self.msg_steps
            data.ini_pos = node_pos.clone()
            edges = self.connect_triangle(node_pos, tri_cells)
            for line in line_cells:
                edges.append(torch.LongTensor([line[0], line[1]]))
                edges.append(torch.LongTensor([line[0], line[1]]))

            data.edge_index = torch.stack(edges).t().contiguous()
            data = data.to(device)
        return data

    def train_heat_graph(self, device=torch.device('cuda')):
        maxium_size = 3e-3
        flag = False
        with pygmsh.occ.Geometry() as geom:
            geom.characteristic_length_max = maxium_size
            circle_large = geom.add_disk([0, 0], 0.05)
            mask_below = geom.add_polygon(
                [
                    [-0.05, 0],
                    [0.05, 0],
                    [0.05, -0.05],
                    [-0.05, -0.05]
                ]
            )
            out = geom.boolean_difference(circle_large, mask_below)
            circle_small = geom.add_disk([0, 0], 0.03)
            out = geom.boolean_difference(out, circle_small)
            circle_right = Disk([-0.04, 0], 0.01)
            mask_up = geom.add_polygon(
                [
                    [-0.05, 0],
                    [0.05, 0],
                    [0.05, 0.05],
                    [-0.05, 0.05]
                ]
            )
            geom.boolean_difference(circle_right, mask_up)
            mask_up = geom.add_polygon(
                [
                    [-0.05, 0],
                    [0.05, 0],
                    [0.05, 0.05],
                    [-0.05, 0.05]
                ]
            )
            circle_left = Disk([0.04, 0], 0.01)
            geom.boolean_difference(circle_left, mask_up)
            
            mesh = geom.generate_mesh()
            points = mesh.points
            if mesh.cells_dict.get('triangle', None).any():
                tri_cells = mesh.cells_dict['triangle']
                flag = True
            if mesh.cells_dict.get('line', None).any():
                line_cells = mesh.cells_dict['line']
                flag = True
            if not flag:
                raise ValueError("no edge")
            node_pos = points[:, :2]
            node_pos = torch.FloatTensor(np.stack(node_pos, 0)).cuda()
            self.data.num_nodes = len(points)
            data = self.data
            data.pos = node_pos
            data.msg_steps = self.msg_steps
            data.ini_pos = node_pos.clone()
            edges = self.connect_triangle(node_pos, tri_cells)
            for line in line_cells:
                edges.append(torch.LongTensor([line[0], line[1]]))
                edges.append(torch.LongTensor([line[0], line[1]]))

            data.edge_index = torch.stack(edges).t().contiguous()
            data = data.to(device)
        return data

    def train_heat_graph_2(self, device=torch.device('cuda')):
        maxium_size = 4.1e-3
        flag = False
        with pygmsh.occ.Geometry() as geom:
            geom.characteristic_length_max = maxium_size
            circle_large = geom.add_disk([0, 0], 0.05)
            mask_below = geom.add_disk([0.02, 0], 0.015)
            out = geom.boolean_difference(circle_large, mask_below)
            circle_small = geom.add_disk([-0.02, 0], 0.015)
            out = geom.boolean_difference(out, circle_small)    
            
            mesh = geom.generate_mesh()
            points = mesh.points
            if mesh.cells_dict.get('triangle', None).any():
                tri_cells = mesh.cells_dict['triangle']
                flag = True
            if mesh.cells_dict.get('line', None).any():
                line_cells = mesh.cells_dict['line']
                flag = True
            if not flag:
                raise ValueError("no edge")
            node_pos = points[:, :2]
            node_pos = torch.FloatTensor(np.stack(node_pos, 0)).cuda()
            self.data.num_nodes = len(points)
            data = self.data
            data.pos = node_pos
            data.msg_steps = self.msg_steps
            data.ini_pos = node_pos.clone()
            edges = self.connect_triangle(node_pos, tri_cells)
            for line in line_cells:
                edges.append(torch.LongTensor([line[0], line[1]]))
                edges.append(torch.LongTensor([line[0], line[1]]))

            data.edge_index = torch.stack(edges).t().contiguous()
            data = data.to(device)
        return data

    def train_heat_graph_3(self, device=torch.device('cuda'), size=2e-2):
        maxium_size = size
        flag = False
        with pygmsh.occ.Geometry() as geom:
            geom.characteristic_length_max = maxium_size
            circle_large = geom.add_disk([0, 0], 0.3, 0.2)
            mask_below = geom.add_disk([0, 0], 0.15, 0.1)
            out = geom.boolean_difference(circle_large, mask_below)
            
            mesh = geom.generate_mesh()
            points = mesh.points
            if mesh.cells_dict.get('triangle', None).any():
                tri_cells = mesh.cells_dict['triangle']
                flag = True
            if mesh.cells_dict.get('line', None).any():
                line_cells = mesh.cells_dict['line']
                flag = True
            if not flag:
                raise ValueError("no edge")
            node_pos = points[:, :2]
            node_pos = torch.FloatTensor(np.stack(node_pos, 0)).cuda()
            self.data.num_nodes = len(points)
            data = self.data
            data.pos = node_pos
            data.msg_steps = self.msg_steps
            data.ini_pos = node_pos.clone()
            edges = self.connect_triangle(node_pos, tri_cells)
            for line in line_cells:
                edges.append(torch.LongTensor([line[0], line[1]]))
                edges.append(torch.LongTensor([line[0], line[1]]))

            data.edge_index = torch.stack(edges).t().contiguous()
            data = data.to(device)
        return data

    def train_heat_graph_4(self, device=torch.device('cuda')):
        maxium_size = 0.042
        points_need = 25
        with pygmsh.geo.Geometry() as geom:
            theta = np.linspace(0, 0.5*np.pi, points_need)
            x = 0.5*np.sin(2*theta) * np.cos(theta)
            y = 0.5*np.sin(2*theta) * np.sin(theta)
            coor_list = [[i,j] for i, j in zip(x, y)]
            geom.add_polygon(    
                coor_list,
                mesh_size = maxium_size,
            )
            
            theta = np.linspace(0.5*np.pi, np.pi, points_need)
            x = 0.5*np.sin(2*theta) * np.cos(theta)
            y = 0.5*np.sin(2*theta) * np.sin(theta)
            coor_list = [[i,j] for i, j in zip(x, y)]
            geom.add_polygon(    
                coor_list,
                mesh_size = maxium_size,
            )
            
            theta = np.linspace(np.pi, 1.5*np.pi, points_need)
            x = 0.5*np.sin(2*theta) * np.cos(theta)
            y = 0.5*np.sin(2*theta) * np.sin(theta)
            coor_list = [[i,j] for i, j in zip(x, y)]
            geom.add_polygon(    
                coor_list,
                mesh_size = maxium_size,
            )
            
            theta = np.linspace(1.5*np.pi, 2*np.pi, points_need)
            x = 0.5*np.sin(2*theta) * np.cos(theta)
            y = 0.5*np.sin(2*theta) * np.sin(theta)
            coor_list = [[i,j] for i, j in zip(x, y)]
            geom.add_polygon(    
                coor_list,
                mesh_size = maxium_size,
            )
            
            mesh = geom.generate_mesh()
            points = mesh.points
            tri_cells = mesh.cells_dict['triangle']
            line_cells = mesh.cells_dict['line']
            node_pos = points[:, :2]
            node_pos = torch.FloatTensor(np.stack(node_pos, 0)).cuda()
            self.data.num_nodes = len(points)
            data = self.data
            data.pos = node_pos
            data.msg_steps = self.msg_steps
            data.ini_pos = node_pos.clone()
            edges = self.connect_triangle(node_pos, tri_cells)
            for line in line_cells:
                edges.append(torch.LongTensor([line[0], line[1]]))
                edges.append(torch.LongTensor([line[0], line[1]]))

            data.edge_index = torch.stack(edges).t().contiguous()
            data = data.to(device)
        return data

    def train_heat_graph_5(self, device=torch.device('cuda')):
        maxium_size = 0.031
        points_need = 30
        with pygmsh.occ.Geometry() as geom:
            theta = np.linspace(0, np.pi, points_need)
            x = 0.3*np.cos(theta) * (1 - np.cos(theta))
            y = 0.3*np.sin(theta) * (1 - np.cos(theta))
            coor_list = [[i,j] for i, j in zip(x, y)]
            poly1 = geom.add_polygon(    
                coor_list,
                mesh_size = maxium_size,
            )
            circle_small = Disk([-0.3, 0], 0.3, 0.2)
            geom.rotate(circle_small, (-0.3, 0, 0), np.pi/2, (0, 0, 1))
            out = geom.boolean_difference(poly1, circle_small) 
            
            theta = np.linspace(np.pi, 2*np.pi, points_need)
            x = 0.3*np.cos(theta) * (1 - np.cos(theta))
            y = 0.3*np.sin(theta) * (1 - np.cos(theta))
            coor_list = [[i,j] for i, j in zip(x, y)]
            poly2 = geom.add_polygon(    
                coor_list,
                mesh_size = maxium_size,
            )
            
            circle_small = Disk([-0.3, 0], 0.3, 0.2)
            geom.rotate(circle_small, (-0.3, 0, 0), np.pi/2, (0, 0, 1))
            out = geom.boolean_difference(poly2, circle_small) 
            
            geom.characteristic_length_max = maxium_size
            mesh = geom.generate_mesh()
            points = mesh.points
            tri_cells = mesh.cells_dict['triangle']
            line_cells = mesh.cells_dict['line']
            node_pos = points[:, :2]
            node_pos = torch.FloatTensor(np.stack(node_pos, 0)).cuda()
            self.data.num_nodes = len(points)
            data = self.data
            data.pos = node_pos
            data.msg_steps = self.msg_steps
            data.ini_pos = node_pos.clone()
            edges = self.connect_triangle(node_pos, tri_cells)
            for line in line_cells:
                edges.append(torch.LongTensor([line[0], line[1]]))
                edges.append(torch.LongTensor([line[0], line[1]]))

            data.edge_index = torch.stack(edges).t().contiguous()
            data = data.to(device)
        return data

    def train_heat_graph_6(self, device=torch.device('cuda')):
        maxium_size = 0.29
        points_need = 30
        with pygmsh.occ.Geometry() as geom:
            datas = loadmat("datasets/heat/dataGNN6.mat")
            xs = datas['x'].flatten()[0:200:5]
            ys = datas['y'].flatten()[0:200:5]
            coor_list = [[i,j] for i, j in zip(xs, ys)]
            geom.add_polygon(    
                coor_list,
                mesh_size = maxium_size,
            )
            xs = datas['x'].flatten()[200:400:5]
            ys = datas['y'].flatten()[200:400:5]
            coor_list = [[i,j] for i, j in zip(xs, ys)]
            geom.add_polygon(    
                coor_list,
                mesh_size = maxium_size,
            )
            xs = datas['x'].flatten()[400:600:5]
            ys = datas['y'].flatten()[400:600:5]
            coor_list = [[i,j] for i, j in zip(xs, ys)]
            geom.add_polygon(    
                coor_list,
                mesh_size = maxium_size,
            )
            xs = datas['x'].flatten()[600:800:5]
            ys = datas['y'].flatten()[600:800:5]
            coor_list = [[i,j] for i, j in zip(xs, ys)]
            geom.add_polygon(    
                coor_list,
                mesh_size = maxium_size,
            )
        
            mesh = geom.generate_mesh()
            points = mesh.points
            tri_cells = mesh.cells_dict['triangle']
            line_cells = mesh.cells_dict['line']
            node_pos = points[:, :2]
            node_pos = torch.FloatTensor(np.stack(node_pos, 0)).cuda()
            self.data.num_nodes = len(points)
            data = self.data
            data.pos = node_pos
            data.msg_steps = self.msg_steps
            data.ini_pos = node_pos.clone()
            edges = self.connect_triangle(node_pos, tri_cells)
            for line in line_cells:
                edges.append(torch.LongTensor([line[0], line[1]]))
                edges.append(torch.LongTensor([line[0], line[1]]))

            data.edge_index = torch.stack(edges).t().contiguous()
            data = data.to(device)
        return data

    def export_graph_6(self):
        maxium_size = 0.29
        points_need = 30
        with pygmsh.occ.Geometry() as geom:
            datas = loadmat("datasets\heat\dataGNN6.mat")
            xs = datas['x'].flatten()[0:200:5]
            ys = datas['y'].flatten()[0:200:5]
            coor_list = [[i,j] for i, j in zip(xs, ys)]
            geom.add_polygon(    
                coor_list,
                mesh_size = maxium_size,
            )
            xs = datas['x'].flatten()[200:400:5]
            ys = datas['y'].flatten()[200:400:5]
            coor_list = [[i,j] for i, j in zip(xs, ys)]
            geom.add_polygon(    
                coor_list,
                mesh_size = maxium_size,
            )
            xs = datas['x'].flatten()[400:600:5]
            ys = datas['y'].flatten()[400:600:5]
            coor_list = [[i,j] for i, j in zip(xs, ys)]
            geom.add_polygon(    
                coor_list,
                mesh_size = maxium_size,
            )
            xs = datas['x'].flatten()[600:800:5]
            ys = datas['y'].flatten()[600:800:5]
            coor_list = [[i, j, 0] for i, j in zip(xs, ys)]
            coor_list1 = [[i, j, 1] for i, j in zip(xs, ys)]
            geom.add_polygon(    
                coor_list,
                mesh_size = maxium_size,
            )
            mesh = geom.generate_mesh()
            return mesh
            

    def plot_polynomial(self, point_list: List[List[int]], device=torch.device('cuda')):
        """
        point_list: connect from the first, one by one
        """
        maxium_size = 0.1
        with pygmsh.geo.Geometry() as geom:
            geom.add_polygon(
                point_list,
                mesh_size=maxium_size,
            )
            mesh = geom.generate_mesh()
            points = mesh.points
            tri_cells = mesh.cells_dict['triangle']
            line_cells = mesh.cells_dict['line']
            node_pos = points[:, :2]
            node_pos = torch.FloatTensor(np.stack(node_pos, 0)).cuda()
            self.data.num_nodes = len(points)
            data = self.data
            data.pos = node_pos
            data.msg_steps = self.msg_steps
            data.ini_pos = node_pos.clone()
            edges = self.connect_triangle(node_pos, tri_cells)
            for line in line_cells:
                edges.append(torch.LongTensor([line[0], line[1]]))
                edges.append(torch.LongTensor([line[0], line[1]]))

            data.edge_index = torch.stack(edges).t().contiguous()
            data = data.to(device)
        return data


class GraphLessMsgStep(GraphCreater):
    def __init__(self, data_idx=-1) -> None:
        super().__init__(data_idx)

    @property
    def msg_steps(self) -> int:
        value = round(np.sqrt(self.data.num_nodes))*2 -1 
        value = round(value/2)
        return value


def show_graph(G, writer=None, epoch=0):
    '''
    Shows mesh on top of image
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    Pos = G.pos.clone().detach().cpu().numpy()
    Edges = G.edge_index.clone().detach().cpu().numpy()
    lines = []
    for i in range(Edges.shape[1]):
        a, b = Edges[0][i], Edges[1][i]
        lines.append([(Pos[a,0],Pos[a,1]),(Pos[b,0],Pos[b,1])])
    lc = mc.LineCollection(lines, linewidths=1, colors='r')
    ax.add_collection(lc)
    if writer is None:
        ax.axis("equal")
        plt.show()
        pass
    else:
        pass
    plt.clf()


def show_graph_2(writer=None, epoch=0):
    '''
    Shows mesh on top of image
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    grids = [
        [0.0, 0.0],
        [0.0, 0.3],
        [0.75, 0.5],
        [0.75, 1.0],
        [0.0, 1.2],
        [0.0, 1.5],
        [2.0, 1.5],
        [2.0, 1.2],
        [1.25, 1.0],
        [1.25, 0.5],
        [2.0, 0.3],
        [2.0, 0.0]
    ]
    lines = []
    for i in range(len(grids)):
        lines.append([(grids[i][0], grids[i][1]),(grids[(i+1)%len(grids)][0], grids[(i+1)%len(grids)][1])])
    lc = mc.LineCollection(lines, linewidths=1, colors='r')
    ax.add_collection(lc)
    if writer is None:
        plt.xlim(-1, 4.0)
        plt.ylim(-1, 3.0)
        ax.axis('equal')
        ax.axis("off")
        plt.show()
        pass
    else:
        pass
    plt.clf()

