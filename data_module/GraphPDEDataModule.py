import numpy as np
import torch

import scipy
from scipy.spatial.distance import cdist
import h5py

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from data_module.PDEDataModule import PDEDataModule

class GraphPDEDataModule(PDEDataModule):
    def __init__(self, config:dict):
        super().__init__(config)
        self.edge_radius = config['EDGE_RADIUS']

    def generate_edge_info(self, grid:np.ndarray):
        distances = cdist(grid, grid)
        connections = np.where(np.logical_and(distances < self.edge_radius, distances > 0))
        edges = np.vstack(connections).astype(np.int32)
        edge_features = np.concatenate((
            distances[connections[0], connections[1]].reshape(-1, 1),
            grid[connections[0], :],
            grid[connections[1], :] 
        ), axis=-1)
        return edges, edge_features

    def get_dataset(self, data:np.ndarray, grid:np.ndarray, edges:np.ndarray, edge_features:np.ndarray, image_size:list):
        dataset = []
        for example_idx in range(data.shape[0]):
            dataset.append(Data(
                x=torch.tensor(data[example_idx,:,:self.time_steps_in], dtype=torch.float32),
                y=torch.tensor(data[example_idx,:,self.time_steps_in:self.time_steps_in+self.time_steps_out], dtype=torch.float32),
                edge_index=torch.tensor(edges, dtype=torch.int64),
                edge_attr=torch.tensor(edge_features, dtype=torch.float32),
                grid=torch.tensor(grid, dtype=torch.float32),
                image_size=torch.tensor(image_size, dtype=torch.int16)
            ))
        return dataset

    def get_data_loader(self, dataset, shuffle:bool, inference:bool=False):
        # if(inference):
        #     DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=4, persistent_workers=False)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)#, num_workers=4, persistent_workers=True)

    def flatten_nodes(self, data:np.ndarray, grid:np.ndarray):
        shape = data.shape
        data = data.reshape(shape[0], shape[1]*shape[2], shape[3])
        grid = grid.reshape(shape[1]*shape[2], -1)
        return data, grid

    def pipeline(self, data:np.ndarray, split:str, shuffle:bool, downsample_ratio:int=None, inference:bool=False):
        assert shuffle == True or split != 'train'
        grid = self.generate_grid(nx=data.shape[1], ny=data.shape[2])
        data, grid = self.downsample_data(data, grid, ratio=downsample_ratio)
        data, grid = self.cut_data(data, grid)
        image_size = grid.shape[:-1]
        if(split == 'train' and self.image_size is None):
            self.image_size = image_size
        data, grid = self.flatten_nodes(data, grid)
        edges, edge_features = self.generate_edge_info(grid)
        dataset = self.get_dataset(data, grid, edges, edge_features, image_size)
        return self.get_data_loader(dataset, shuffle=shuffle, inference=inference)