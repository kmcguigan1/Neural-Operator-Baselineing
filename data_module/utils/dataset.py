import numpy as np
import torch

from scipy.spatial.distance import cdist

class PDEDataset(torch.utils.data.Dataset):
    def __init__(self, array:np.ndarray, grid:np.ndarray, time_steps_in:int, time_steps_out:int, time_int:int):
        super().__init__()
        # save the data that we need
        self.array = array.copy().astype(np.float32) # (example, x, y, time)
        self.grid = grid.copy().astype(np.float32)
        self.image_shape = self.array.shape[-2:]
        self.time_steps_in = time_steps_in
        self.time_steps_out = time_steps_out
        self.time_int = time_int
        # generate the indecies for where we can take samples from
        self.indecies_map = []
        for example_idx in range(self.array.shape[0]):
            for time_idx in range(0, self.array.shape[-1], self.time_int):
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
    def __init__(self, array:np.ndarray, edges:np.ndarray, edge_features:np.ndarray, time_steps_in:int, time_steps_out:int, time_int:int, neighbors_method:str):
        super().__init__()
        # save the data that we need
        self.array = array.copy().astype(np.float32) # (example, x, y, time)
        self.grid = grid.copy().astype(np.float32)
        self.edges = edges.copy().astype(np.int32)
        self.edge_features = edge_features.copy().astype(np.float32)
        self.image_shape = self.array.shape[1:-1]
        self.time_steps_in = time_steps_in
        self.time_steps_out = time_steps_out
        self.time_int = time_int
        self.neighbors_method = neighbors_method
        # generate the indecies for where we can take samples from
        self.indecies_map = []
        for example_idx in range(self.array.shape[0]):
            for time_idx in range(0, self.array.shape[-1], self.time_int):
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
        return X, y, self.grid, self.edges, self.edge_features

class SingleSamplePDEDataset(torch.utils.data.Dataset):
    def __init__(self, x:np.ndarray, y:np.ndarray, grid:np.ndarray):
        super().__init__()
        # save the data that we need
        self.x = x.copy().astype(np.float32) # (example, x, y, time)
        self.y = y.copy().astype(np.float32) # (example, x, y, time)
        self.grid = grid.copy().astype(np.float32)
        self.image_shape = self.array.shape[1:-1]
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, idx):
        return self.x[idx, ...], self.y[idx, ...], self.grid

class SingleSampleGraphPDEDataset(torch.utils.data.Dataset):
    def __init__(self, x:np.ndarray, y:np.ndarray, edges:np.ndarray, edge_features:np.ndarray, time_steps_in:int, time_steps_out:int, time_int:int, neighbors_method:str):
        super().__init__()
        # save the data that we need
        self.x = x.copy().astype(np.float32) # (example, x, y, time)
        self.y = y.copy().astype(np.float32) # (example, x, y, time)
        self.grid = grid.copy().astype(np.float32)
        self.edges = edges.copy().astype(np.int32)
        self.edge_features = edge_features.copy().astype(np.float32)
        self.image_shape = self.x.shape[1:-1]
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, idx):
        return self.x[idx, ...], self.y[idx, ...], self.grid, self.edges, self.edge_features