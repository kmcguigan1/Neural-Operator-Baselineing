import numpy as np
import torch

import scipy
from scipy.spatial.distance import cdist
import h5py

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from data_module.GraphPDEDataModule import GraphPDEDataModule

class MPGraphPDEDataModule(GraphPDEDataModule):
    def __init__(self, config:dict):
        super().__init__(config)
        self.levels = len(self.edge_radius)

    def generate_edge_info_subset(self, grid:np.array, resolution:int, edge_radius:float):
        this_grid = grid[::resolution,::resolution,:].copy()
        distances = cdist(this_grid, this_grid)
        connections = np.where(np.logical_and(distances < edge_radius, distances > 0))
        edges = np.vstack(connections).astype(np.int32)
        edge_features = np.concatenate((
            distances[connections[0], connections[1]].reshape(-1, 1),
            this_grid[connections[0], :],
            this_grid[connections[1], :] 
        ), axis=-1)
        return edges, edge_features

    def generate_edge_info(self, grid:np.ndarray):
        edge_index_1, edge_features_1 = self.generate_edge_info_subset(grid, 1, self.edge_radius[0])
        edge_index_2, edge_features_2 = self.generate_edge_info_subset(grid, 2, self.edge_radius[1])
        edge_index_3, edge_features_3 = self.generate_edge_info_subset(grid, 4, self.edge_radius[2])
        return [edge_index_1, edge_index_2, edge_index_3], [edge_features_1, edge_features_2, edge_features_3]

    def get_dataset(self, data:np.ndarray, grid:np.ndarray, edges:list, edge_features:list, image_size:list):
        dataset = []
        for example_idx in range(data.shape[0]):
            dataset.append(Data(
                x=torch.tensor(data[example_idx,:,:self.time_steps_in], dtype=torch.float32),
                y=torch.tensor(data[example_idx,:,self.time_steps_in:self.time_steps_in+self.time_steps_out], dtype=torch.float32),
                grid=torch.tensor(grid, dtype=torch.float32),
                image_size=torch.tensor(image_size, dtype=torch.int16),
                edge_index_1=torch.tensor(edges[0], dtype=torch.int64),
                edge_index_2=torch.tensor(edges[1], dtype=torch.int64),
                edge_index_3=torch.tensor(edges[2], dtype=torch.int64),
                edge_features_1=torch.tensor(edge_features[0], dtype=torch.float32),
                edge_features_2=torch.tensor(edge_features[1], dtype=torch.float32),
                edge_features_3=torch.tensor(edge_features[2], dtype=torch.float32),
            ))
        return dataset
    
class MPGraphPDEDataModuleCust(GraphPDEDataModule):
    def __init__(self, config:dict):
        super().__init__(config)
    
    def generate_edges(self, grid:np.ndarray, distances:np.ndarray, edge_radius:float, src_mask:np.ndarray=None, dst_mask:np.ndarray=None):
        connections = np.where(np.logical_and(distances < edge_radius, distances > 0))
        edges = np.vstack(connections).astype(np.int32) # shape is [src] -> [dst]
        if(src_mask is not None):
            mask = np.where(edges[0, :] in src_mask)[0]
            edges = edges[mask, :]
        if(dst_mask is not None):
            mask = np.where(edges[1, :] in dst_mask)[0]
            edges = edges[mask, :]
        edge_features = np.concatenate((
            distances[edges[0, :], edges[1, :]].reshape(-1, 1),
            grid[edges[0, :], :],
            grid[edges[1, :], :] 
        ), axis=-1)
        return edges, edge_features

    def generate_edge_info(self, grid:np.ndarray):
        # get the node masks
        mask2 = np.zeros((grid.shape[0], grid.shape[1]))
        mask2[::2, ::2] = 1
        mask3 = np.zeros((grid.shape[0], grid.shape[1]))
        mask3[::4, ::4] = 1
        # flatten the grid 
        grid = grid.reshape(grid.shape[1]*grid.shape[2], -1)
        mask2 = mask2.reshape(mask2.shape[1]*mask2.shape[2])
        mask3 = mask3.reshape(mask3.shape[1]*mask3.shape[2])
        # get the node indecies of the node masks
        mask2 = np.where(mask2 == 1)[0]
        mask3 = np.where(mask3 == 1)[0]
        # get the distances between all nodes
        distances = cdist(grid, grid)
        # get the edges and features for the different directions
        edge_index_11, edge_features_11 = self.generate_edges(grid, distances, self.edge_radius[0])
        edge_index_12, edge_features_12 = self.generate_edges(grid, distances, self.edge_radius[0], dst_mask=mask2)
        
        edge_index_22, edge_features_22 = self.generate_edges(grid, distances, self.edge_radius[1], src_mask=mask2, dst_mask=mask2)
        edge_index_23, edge_features_23 = self.generate_edges(grid, distances, self.edge_radius[1], src_mask=mask2, dst_mask=mask3)

        edge_index_33, edge_features_33 = self.generate_edges(grid, distances, self.edge_radius[2], src_mask=mask3, dst_mask=mask3)

        edge_index_32, edge_features_32 = self.generate_edges(grid, distances, self.edge_radius[1], src_mask=mask3, dst_mask=mask2)
        edge_index_21, edge_features_21 = self.generate_edges(grid, distances, self.edge_radius[0], src_mask=mask2)
        return (
            [edge_index_11, edge_index_12, edge_index_22, edge_index_23, edge_index_33, edge_index_32, edge_index_21], 
            [edge_features_11, edge_features_12, edge_features_22, edge_features_23, edge_features_33, edge_features_32, edge_features_21]
        )

    def get_dataset(self, data:np.ndarray, grid:np.ndarray, edges:list, edge_features:list, image_size:list):
        dataset = []
        for example_idx in range(data.shape[0]):
            dataset.append(Data(
                x=torch.tensor(data[example_idx,:,:self.time_steps_in], dtype=torch.float32),
                y=torch.tensor(data[example_idx,:,self.time_steps_in:self.time_steps_in+self.time_steps_out], dtype=torch.float32),
                grid=torch.tensor(grid, dtype=torch.float32),
                image_size=torch.tensor(image_size, dtype=torch.int16),
                edge_index_11=torch.tensor(edges[0], dtype=torch.int64), 
                edge_index_12=torch.tensor(edges[1], dtype=torch.int64), 
                edge_index_22=torch.tensor(edges[2], dtype=torch.int64), 
                edge_index_23=torch.tensor(edges[3], dtype=torch.int64), 
                edge_index_33=torch.tensor(edges[4], dtype=torch.int64), 
                edge_index_32=torch.tensor(edges[5], dtype=torch.int64), 
                edge_index_21=torch.tensor(edges[6], dtype=torch.int64),
                edge_attr_11=torch.tensor(edge_features[0], dtype=torch.float32), 
                edge_attr_12=torch.tensor(edge_features[1], dtype=torch.float32), 
                edge_attr_22=torch.tensor(edge_features[2], dtype=torch.float32), 
                edge_attr_23=torch.tensor(edge_features[3], dtype=torch.float32), 
                edge_attr_33=torch.tensor(edge_features[4], dtype=torch.float32), 
                edge_attr_32=torch.tensor(edge_features[5], dtype=torch.float32), 
                edge_attr_21=torch.tensor(edge_features[6], dtype=torch.float32),
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
            data, grid = self.flatten_nodes(data, grid)
            edges, edge_features = self.generate_edge_info(grid)
            dataset = self.get_dataset(data, grid, edges, edge_features, image_size)
            return self.get_data_loader(dataset, shuffle=shuffle, inference=inference)