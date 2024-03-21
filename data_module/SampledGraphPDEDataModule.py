import numpy as np
import torch

import scipy
from scipy.spatial.distance import cdist
import h5py

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from data_module.PDEDataModule import PDEDataModule

class SampledGraphPDEDataModule(PDEDataModule):
    def __init__(self, config:dict):
        super().__init__(config)
        self.edge_radius = 0.05 #config['EDGE_RADIUS']

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

    def generate_boundary_distance_mask(self, grid:np.ndarray):
        # define the boundaries grid mask
        x_bounds = np.stack((grid[:, 0], 1.0 - grid[:, 0]), axis=-1).min(axis=-1)
        y_bounds = np.stack((grid[:, 1], 1.0 - grid[:, 1]), axis=-1).min(axis=-1)
        bounds = np.stack((x_bounds, y_bounds), axis=-1).min(axis=-1)
        return bounds

    def get_dataset(self, data:np.ndarray, grid:np.ndarray, edges:np.ndarray, edge_features:np.ndarray):
        dataset = []
        for example_idx in range(data.shape[0]):
            dataset.append(Data(
                x=torch.tensor(data[example_idx,:,:self.time_steps_in], dtype=torch.float32),
                y=torch.tensor(data[example_idx,:,self.time_steps_in:self.time_steps_in+self.time_steps_out], dtype=torch.float32),
                edge_index=torch.tensor(edges, dtype=torch.int64),
                edge_attr=torch.tensor(edge_features, dtype=torch.float32),
                grid=torch.tensor(grid, dtype=torch.float32)
            ))
        return dataset

    def get_data_loader(self, dataset, shuffle:bool):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)#, num_workers=4, persistent_workers=True)

    def flatten_nodes(self, data:np.ndarray, grid:np.ndarray):
        shape = data.shape
        data = data.reshape(shape[0], shape[1]*shape[2], shape[3])
        grid = grid.reshape(shape[1]*shape[2], -1)
        return data, grid

    def pipeline(self, data:np.ndarray, split:str, shuffle:bool, downsample_ratio:int=None):
        assert shuffle == True or split != 'train'
        grid = self.generate_grid(nx=data.shape[1], ny=data.shape[2])
        data, grid = self.downsample_data(data, grid, ratio=downsample_ratio)
        data, grid = self.cut_data(data, grid)
        data, grid = self.flatten_nodes(data, grid)
        edges, edge_features = self.generate_edge_info(grid)
        dataset = self.get_dataset(data, grid, edges, edge_features)
        if(split == 'train' and self.image_size is None):
            self.image_size = grid.shape[:-1]
        return self.get_data_loader(dataset, shuffle=shuffle)


class SampledGraphPDEDataset(torch.data.utils.Dataset):
    def __init__(self, nodes, grid, distances):
        super().__init__()
        # save info we need
        self.nodes = nodes
        self.grid = grid
        self.distances = distances
        self.prior = np.repeat(1, repeats=self.nodes.shape[1])
    def __len__(self):
        return self.nodes.shape[0]
    def __getitem__(self, idx):
        # select the nodes to keep
        random = np.random.uniform(0, 1, size=self.prior.shape) * self.prior
        # keep the top M nodes in the graph
        indecies = np.argpartition(random, -self.m)[-self.m:]
        indecies = np.sort(indecies)
        nodes = self.nodes[indecies]
        grid = self.grid[indecies]
        arr[indecies]
        selected_indecies = np.argpartition(random, -self.m)
        selected_idx = np.where(random <= prior)[0]
        # get the selected nodes and stuff
        


