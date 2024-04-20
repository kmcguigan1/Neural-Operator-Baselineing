import numpy as np
import torch

import scipy
from scipy.spatial.distance import cdist
import h5py

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from data_module.GraphPDEDataModule import GraphPDEDataModule

class BoundaryGraphPDEDataModule(GraphPDEDataModule):
    def __init__(self, config:dict):
        super().__init__(config)
        
    def generate_edge_info(self, grid:np.ndarray):
        # boundary mask
        boundary_mask = np.zeros((grid.shape[0], grid.shape[1]))
        boundary_mask[0, :] = 1
        boundary_mask[-1, :] = 1
        boundary_mask[:, 0] = 1
        boundary_mask[:, -1] = 1
        print(boundary_mask)
        boundary_mask = boundary_mask.reshape(boundary_mask.shape[0]*boundary_mask.shape[1])
        boundary_nodes_index = np.where(boundary_mask==1)[0]
        # get the rest of the edge information
        this_grid = this_grid.reshape(this_grid.shape[0]*this_grid.shape[1], -1)
        distances = cdist(this_grid, this_grid)
        connections = np.where(np.logical_and(distances < self.edge_radius, distances > 0))
        edges = np.vstack(connections).astype(np.int32)
        edge_features = np.concatenate((
            distances[connections[0], connections[1]].reshape(-1, 1),
            this_grid[connections[0], :],
            this_grid[connections[1], :],
            boundary_mask[connections[0]],
            boundary_mask[connections[1]],
        ), axis=-1)
        return [edges, boundary_nodes_index], [edge_features,]

    def get_dataset(self, data:np.ndarray, grid:np.ndarray, edges:list, edge_features:list, image_size:list):
        dataset = []
        for example_idx in range(data.shape[0]):
            dataset.append(Data(
                x=torch.tensor(data[example_idx,:,:self.time_steps_in], dtype=torch.float32),
                y=torch.tensor(data[example_idx,:,self.time_steps_in:self.time_steps_in+self.time_steps_out], dtype=torch.float32),
                grid=torch.tensor(grid, dtype=torch.float32),
                image_size=torch.tensor(image_size, dtype=torch.int16),
                edge_index=torch.tensor(edges[0], dtype=torch.int64),
                boundary_edge_index=torch.tensor(edges[1], dtype=torch.int64),
                edge_attr=torch.tensor(edge_features[0], dtype=torch.float32),
            ))
        return dataset
    
    def pipeline(self, data:np.ndarray, split:str, shuffle:bool, downsample_ratio:int=None, inference:bool=False):
        assert shuffle == True or split != 'train'
        grid = self.generate_grid(nx=data.shape[1], ny=data.shape[2])
        data, grid = self.downsample_data(data, grid, ratio=downsample_ratio)
        data, grid = self.cut_data(data, grid)
        image_size = grid.shape[:-1]
        if(split == 'train' and self.image_size is None):
            self.image_size = image_size
        print("grid shape ", grid.shape)
        edges, edge_features = self.generate_edge_info(grid)
        data, grid = self.flatten_nodes(data, grid)
        dataset = self.get_dataset(data, grid, edges, edge_features, image_size)
        return self.get_data_loader(dataset, shuffle=shuffle, inference=inference)