import numpy as np
import torch

import scipy
from scipy.spatial.distance import cdist
import h5py

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from data_module.GraphPDEDataModule import GraphPDEDataModule

class PairData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'boundary_edge_index':
            return self.boundary_node_index.size(0)
        return super().__inc__(key, value, *args, **kwargs)


class BoundaryGraphPDEDataModule(GraphPDEDataModule):
    def __init__(self, config:dict):
        super().__init__(config)
        
    def generate_edge_info(self, grid:np.ndarray):
        # boundary mask
        boundary_node_mask = np.zeros((grid.shape[0], grid.shape[1]))
        boundary_node_mask[0, :] = 1
        boundary_node_mask[-1, :] = 1
        boundary_node_mask[:, 0] = 1
        boundary_node_mask[:, -1] = 1
        print(boundary_node_mask)
        # get the boundary node mask and the index of the boundary nodes
        boundary_node_mask = boundary_node_mask.reshape(boundary_node_mask.shape[0]*boundary_node_mask.shape[1])
        boundary_node_index = np.where(boundary_node_mask==1)[0]
        boundary_node_mask = boundary_node_mask.reshape(-1, 1)
        # get the grid for this data
        this_grid = grid.copy()
        this_grid = this_grid.reshape(this_grid.shape[0]*this_grid.shape[1], -1)
        # get the edge distances and create connections between nodes
        distances = cdist(this_grid, this_grid)
        connections = np.where(np.logical_and(distances < self.edge_radius, distances > 0))
        edge_index = np.vstack(connections).astype(np.int32)
        # get the edge attributes
        edge_attr = np.concatenate((
            distances[connections[0], connections[1]].reshape(-1, 1),
            this_grid[connections[0], :],
            this_grid[connections[1], :],
            boundary_node_mask[connections[0], :],
            boundary_node_mask[connections[1], :],
        ), axis=-1)
        # pretending we cut out only the boundary nodes
        # we need to create a new edge index that connects all the nodes that are on the boundary
        # maker sure we don't include self loops
        boundary_node_count = np.arange(boundary_node_index.shape[0])
        boundary_edge_index = np.array(np.meshgrid(boundary_node_count, boundary_node_count))
        boundary_edge_index = boundary_edge_index.T.reshape(-1, 2)
        boundary_edge_mask = np.where(boundary_edge_index[:, 0] != boundary_edge_index[:, 1])[0]
        boundary_edge_index = boundary_edge_index[boundary_edge_mask].T
        # next we need the information of the original nodes
        # first step is to convert boundary edge index to replace the indecies with the indecies
        # of each node in the original set
        boundary_edge_index_src_nodes = boundary_node_index[boundary_edge_index[0]]
        boundary_edge_index_dst_nodes = boundary_node_index[boundary_edge_index[1]]
        # next we should bu
        boundary_edge_attr = np.concatenate((
            distances[boundary_edge_index_src_nodes, boundary_edge_index_dst_nodes].reshape(-1,1),
            this_grid[boundary_edge_index_src_nodes, :],
            this_grid[boundary_edge_index_dst_nodes, :],
        ), axis=-1)
        return [edge_index, boundary_edge_index, boundary_node_index, boundary_node_mask], [edge_attr, boundary_edge_attr]

    def get_dataset(self, data:np.ndarray, grid:np.ndarray, edges:list, edge_features:list, image_size:list):
        dataset = []
        for example_idx in range(data.shape[0]):
            dataset.append(PairData(
                x=torch.tensor(data[example_idx,:,:self.time_steps_in], dtype=torch.float32),
                y=torch.tensor(data[example_idx,:,self.time_steps_in:self.time_steps_in+self.time_steps_out], dtype=torch.float32),
                grid=torch.tensor(grid, dtype=torch.float32),
                image_size=torch.tensor(image_size, dtype=torch.int16),
                edge_index=torch.tensor(edges[0], dtype=torch.int64),
                boundary_edge_index=torch.tensor(edges[1], dtype=torch.int64),
                boundary_node_index=torch.tensor(edges[2], dtype=torch.int64),
                boundary_node_mask=torch.tensor(edges[3], dtype=torch.int64),
                edge_attr=torch.tensor(edge_features[0], dtype=torch.float32),
                boundary_edge_attr=torch.tensor(edge_features[1], dtype=torch.float32),
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