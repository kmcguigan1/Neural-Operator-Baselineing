import numpy as np
import torch

from scipy.spatial.distance import cdist
from data_module.utils.data_processor import DataContainer, GraphDataContainer

class PDEDataset(torch.utils.data.Dataset):
    def __init__(self, data_container:DataContainer, time_steps_in:int, time_steps_out:int, time_interval:int):
        super().__init__()
        # save the data that we need
        self.array = data_container.data.copy().astype(np.float32) # (example, x, y, time)
        self.grid = data_container.grid.copy().astype(np.float32)
        self.image_shape = self.array.shape[-2:]
        self.time_steps_in = time_steps_in
        self.time_steps_out = time_steps_out
        self.time_interval = time_interval
        # generate the indecies for where we can take samples from
        self.indecies_map = []
        for example_idx in range(self.array.shape[0]):
            if(self.time_interval == -1):
                self.indecies_map.append((example_idx, 0))
            else:
                for time_idx in range(0, self.array.shape[-1], self.time_interval):
                    if(time_idx < self.array.shape[-1] - self.time_steps_in - self.time_steps_out):
                        self.indecies_map.append((example_idx, time_idx))
    
    def __len__(self):
        return len(self.indecies_map)

    def __getitem__(self, idx):
        (exmaple_idx, time_idx) = self.indecies_map[idx]
        # get the observations
        X = self.array[exmaple_idx, ..., time_idx:time_idx+self.time_steps_in]
        y = self.array[exmaple_idx, ...,  time_idx+self.time_steps_in:time_idx+self.time_steps_in+self.time_steps_out]
        return X, y, self.grid

class GraphPDEDataset(torch.utils.data.Dataset):
    def __init__(self, data_container:GraphDataContainer, time_steps_in:int, time_steps_out:int, time_interval:int, neighbors_method:str):
        super().__init__()
        # save the data that we need
        self.array = data_container.data.copy().astype(np.float32) # (example, x, y, time)
        self.grid = data_container.grid.copy().astype(np.float32)
        self.edges = data_container.edges.copy().astype(np.int32)
        self.edge_attrs = data_container.edge_attrs.copy().astype(np.float32)
        self.image_shape = self.array.shape[1:-1]
        self.time_steps_in = time_steps_in
        self.time_steps_out = time_steps_out
        self.time_interval = time_interval
        self.neighbors_method = neighbors_method
        # generate the indecies for where we can take samples from
        self.indecies_map = []
        for example_idx in range(self.array.shape[0]):
            if(self.time_interval == -1):
                self.indecies_map.append((example_idx, 0))
            else:
                for time_idx in range(0, self.array.shape[-1], self.time_interval):
                    if(time_idx < self.array.shape[-1] - self.time_steps_in - self.time_steps_out):
                        self.indecies_map.append((example_idx, time_idx))
    def __len__(self):
        return len(self.indecies_map)
    def __getitem__(self, idx):
        (exmaple_idx, time_idx) = self.indecies_map[idx]
        # get the observations
        X = self.array[exmaple_idx, ..., time_idx:time_idx+self.time_steps_in]
        y = self.array[exmaple_idx, ..., time_idx+self.time_steps_in:time_idx+self.time_steps_in+self.time_steps_out]
        # return the data
        return X, y, self.grid, self.edges, self.edge_attrs