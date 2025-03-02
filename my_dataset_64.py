import torch
import numpy as np
from torch.utils.data import Dataset
from copy import deepcopy
from scipy.io import loadmat

group_num = 125
room = 32

class PoissonSquareRoomInpDataset(Dataset):

    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.group_num = group_num
        self.room_size = room
        self.data_key = 'boundary'

    def __len__(self):
        return self.group_num

    def __getitem__(self, idx):
        """
        return [0,1] for coordinates, [2, 3, 4] for boundary and 0, 0
        """
        if type(idx) is not int:
            idx = idx.item()

        if idx >= self.__len__():
            raise StopIteration

        all_inputs = []
        for j in range(idx * self.room_size + 1, (idx + 1) * self.room_size + 1):
            file_dir = self.dir_path + '/' + str(j) + '.mat'
            data = np.transpose(loadmat(file_dir)[self.data_key])
            data = np.hstack((data, np.zeros((np.shape(data)[0], 2))))
            all_inputs.append(deepcopy(data))
        all_inputs = np.array(all_inputs)
        all_inputs = torch.FloatTensor(all_inputs)
        all_inputs.to(torch.float32)
        return (all_inputs[:, :, :2].contiguous(),
                all_inputs[:, :, 2].unsqueeze_(-1).contiguous())


class PoissonSquareRoomOutDataset(Dataset):

    def __init__(self, file_path):
        self.dir_path = file_path 
        self.room_size = room
        self.group_num = group_num
        self.data_key = "Vdata"

    def __len__(self):
        return self.group_num

    def __getitem__(self, idx):
        """
        """
        if type(idx) is not int:
            idx = idx.item()

        if idx >= self.__len__():
            raise StopIteration

        coordinates = []
        values = []
        for j in range(idx * self.room_size + 1, (idx + 1) * self.room_size + 1):
            file_dir = self.dir_path + '/' + str(j) + '.mat'
            data = loadmat(file_dir)[self.data_key]
            coordinates.append(deepcopy(data[:, 0:2]))
            values.append(deepcopy(data[:, 2]))
        coordinates = np.array(coordinates)
        coordinates = torch.FloatTensor(coordinates)
        values = np.array(values)
        values = torch.FloatTensor(values)
        coordinates.to(torch.float32)
        values.to(torch.float32)
        values.unsqueeze_(2)
        return (coordinates, values)


class HeatInpDataset(Dataset):

    def __init__(self, file_path):
        self.file_path = file_path
        self.group_num = group_num
        self.room_size = room
        self.datas = loadmat(file_path)
        self.xs = self.datas['x']
        self.ys = self.datas['y']

    def __len__(self):
        return self.group_num

    def __getitem__(self, idx):
        """
        return [0,1] for coordinates, [2, 3, 4] for boundary and 0, 0
        """
        if type(idx) is not int:
            idx = idx.item()

        if idx >= self.__len__():
            raise StopIteration

        temperatures = self.datas['Bnorm'][idx * self.room_size:(idx + 1) * self.room_size, :]
        temperatures = torch.FloatTensor(temperatures.astype(np.float32))
        
        x_coordinates = torch.FloatTensor(self.xs.astype(np.float32))
        y_coordinates = torch.FloatTensor(self.ys.astype(np.float32))
        x_coordinates = x_coordinates.repeat(32, 1).unsqueeze_(-1)
        y_coordinates = y_coordinates.repeat(32, 1).unsqueeze_(-1)
        coordinates = torch.cat((x_coordinates, y_coordinates), dim=2)

        return (coordinates.contiguous(),
                temperatures.unsqueeze_(-1).contiguous())


class HeatInpDatasetNoise(Dataset):

    def __init__(self, file_path, noise_level):
        self.file_path = file_path
        self.group_num = group_num
        self.room_size = room
        self.datas = loadmat(file_path)
        self.xs = self.datas['x']
        self.ys = self.datas['y']
        self.noise_level = noise_level

    def __len__(self):
        return self.group_num

    def __getitem__(self, idx):
        """
        return [0,1] for coordinates, [2, 3, 4] for boundary and 0, 0
        """
        if type(idx) is not int:
            idx = idx.item()

        if idx >= self.__len__():
            raise StopIteration

        temperatures = self.datas['Bnorm'][idx * self.room_size:(idx + 1) * self.room_size, :]
        temperatures = temperatures + self.noise_level * np.random.randn(*temperatures.shape)
        temperatures = torch.FloatTensor(temperatures.astype(np.float32))
        
        x_coordinates = torch.FloatTensor(self.xs.astype(np.float32))
        y_coordinates = torch.FloatTensor(self.ys.astype(np.float32))
        x_coordinates = x_coordinates.repeat(32, 1).unsqueeze_(-1)
        y_coordinates = y_coordinates.repeat(32, 1).unsqueeze_(-1)
        coordinates = torch.cat((x_coordinates, y_coordinates), dim=2)

        return (coordinates.contiguous(),
                temperatures.unsqueeze_(-1).contiguous())


class HeatOutDataset(Dataset):

    def __init__(self, file_path):
        self.file_path = file_path
        self.group_num = group_num
        self.room_size = room
        self.datas = loadmat(file_path)
        self.xs = self.datas['xo']
        self.ys = self.datas['yo']
        if self.xs.shape[0] != 1:
            self.xs = np.transpose(self.xs, [1, 0])
        if self.ys.shape[0] != 1:
            self.ys = np.transpose(self.ys, [1, 0])

    def __len__(self):
        return self.group_num

    def __getitem__(self, idx):
        """
        """
        if type(idx) is not int:
            idx = idx.item()

        if idx >= self.__len__():
            raise StopIteration
        
        temperatures = self.datas['Tnorm'][idx * self.room_size:(idx + 1) * self.room_size]
        
        # fix the processing error (normlization bias) of the matlab datafile 
        if self.file_path == "dataset/dataGNN4.mat" or  self.file_path == "dataset/dataGNN3.mat":
            temperatures = temperatures + 0.2
        if self.file_path in [
            "dataset/dataGNN1.mat",
            "dataset/dataGNN2.mat",
            "dataset/dataGNN3.mat",
            "dataset/dataGNN4.mat"
        ]:
            velocities = self.datas['Vnorm'][idx * self.room_size:(idx + 1) * self.room_size] + 0.5
        else:
            velocities = self.datas['Vnorm'][idx * self.room_size:(idx + 1) * self.room_size] 

        temperatures = torch.FloatTensor(temperatures.astype(np.float32)).unsqueeze_(-1)
        velocities = torch.FloatTensor(velocities.astype(np.float32)).unsqueeze_(-1)
        
        values = torch.cat((temperatures, velocities), dim=2)

        x_coordinates = torch.FloatTensor(self.xs.astype(np.float32))
        y_coordinates = torch.FloatTensor(self.ys.astype(np.float32))
        x_coordinates = x_coordinates.repeat(32, 1).unsqueeze_(-1)
        y_coordinates = y_coordinates.repeat(32, 1).unsqueeze_(-1)
        coordinates = torch.cat((x_coordinates, y_coordinates), dim=2)

        return (coordinates, values)
