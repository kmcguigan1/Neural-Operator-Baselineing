import numpy as np
import torch

from scipy.spatial.distance import cdist

class PDEDataset(torch.utils.data.Dataset):
    def __init__(self, array:np.ndarray, time_steps_in:int, time_steps_out:int, time_int:int):
        super().__init__()
        # save the data that we need
        self.array = array.copy().astype(np.float32) # (example, time, x, y)
        self.image_shape = self.array.shape[-2:]
        self.time_steps_in = time_steps_in
        self.time_steps_out = time_steps_out
        self.time_int = time_int
        # get the grid
        grid_x, grid_y = np.meshgrid(
            np.linspace(start=0, stop=1.0, num=array.shape[-2]),
            np.linspace(start=0, stop=1.0, num=array.shape[-1])
        )
        self.grid = np.concatenate(
            (
                np.expand_dims(grid_x, axis=-1),
                np.expand_dims(grid_y, axis=-1)
            ), axis=-1
        ).astype(np.float32)
        # generate the indecies for where we can take samples from
        self.indecies_map = []
        for example_idx in range(self.array.shape[0]):
            for time_idx in range(0, self.array.shape[1], self.time_int):
                if(time_idx < self.array.shape[1] - self.time_steps_in - self.time_steps_out):
                    self.indecies_map.append((example_idx, time_idx))
        # transpose the array to not do it every time we read data
        self.array = self.array.transpose([0, 2, 3, 1]) # (example, x, y, time)
    
    def __len__(self):
        return len(self.indecies_map)

    def __getitem__(self, idx):
        (exmaple_idx, time_idx) = self.indecies_map[idx]
        # get the observations
        X = self.array[exmaple_idx, ..., time_idx:time_idx+self.time_steps_in]
        y = self.array[exmaple_idx, ...,  time_idx+self.time_steps_in:time_idx+self.time_steps_in+self.time_steps_out]
        return X, y, self.grid

class GraphPDEDataset(torch.utils.data.Dataset):
    def __init__(self, array:np.ndarray, time_steps_in:int, time_steps_out:int, time_int:int, neighbors_method:str):
        super().__init__()
        # save the data that we need
        self.array = array.copy().astype(np.float32) # (example, time, x, y)
        self.image_shape = self.array.shape[-2:]
        self.time_steps_in = time_steps_in
        self.time_steps_out = time_steps_out
        self.time_int = time_int
        self.neighbors_method = neighbors_method
        # flatten the array so that we have nodes
        self.array = self.array.reshape(self.array.shape[0], self.array.shape[1], self.image_shape[0]*self.image_shape[1])
        # generate our grid
        grid_x, grid_y = np.meshgrid(
            np.linspace(start=0, stop=1.0, num=self.image_shape[0]),
            np.linspace(start=0, stop=1.0, num=self.image_shape[1])
        )
        self.grid = np.concatenate((
            np.expand_dims(grid_x, axis=-1),
            np.expand_dims(grid_y, axis=-1)
        ), axis=-1).astype(np.float32).reshape(-1, 2)
        # get the distances between nodes and use that to get the node connections
        distances = cdist(self.grid, self.grid)
        # get the distance cuttoff
        single_dist = 1.0 / (self.image_shape[0] - 1)
        if(self.neighbors_method == "radial"):
            distance_cuttoff = 1.9 * (single_dist)
        else:
            raise Exception(f"Neighbors method {self.neighbors_method} not implemented yet")
        connections = np.where(np.logical_and(distances < distance_cuttoff, distances > 0))
        self.edges = np.vstack(connections).astype(np.int32)
        # get the features of each edge such as the distance and the cos and sin of the angle
        self.edge_features = distances[connections[0], connections[1]].reshape(-1, 1)
        self.edge_features = np.concatenate((
            self.edge_features,
            self.grid[connections[0], :],
            self.grid[connections[1], :] 
        ), axis=-1).astype(np.float32)
        # generate the indecies for where we can take samples from
        self.indecies_map = []
        for example_idx in range(self.array.shape[0]):
            for time_idx in range(0, self.array.shape[1], self.time_int):
                if(time_idx < self.array.shape[1] - self.time_steps_in - self.time_steps_out):
                    self.indecies_map.append((example_idx, time_idx))
        # transpose the array so that time is last
        self.array = self.array.transpose([0, 2, 1]) # (example, x y, time)

    def __len__(self):
        return len(self.indecies_map)

    def __getitem__(self, idx):
        (exmaple_idx, time_idx) = self.indecies_map[idx]
        # get the observations
        X = self.array[exmaple_idx, ..., time_idx:time_idx+self.time_steps_in]
        y = self.array[exmaple_idx, ..., time_idx+self.time_steps_in:time_idx+self.time_steps_in+self.time_steps_out]
        # return the data
        return X, y, self.grid, self.edges, self.edge_features

    