import numpy as np
import torch

from scipy.spatial.distance import cdist

class GraphPDEDataset(torch.utils.data.Dataset):
    def __init__(self, array: np.ndarray, config: dict):
        super().__init__()
        # save the data that we need
        self.array = array.copy().astype(np.float32) # (example, time, x, y)
        self.nx = array.shape[-2]
        self.ny = array.shape[-1]
        # flatten the array so that we have nodes
        self.array = self.array.reshape(array.shape[0], array.shape[1], self.nx * self.ny)
        # generate our grid
        grid_x, grid_y = np.meshgrid(
            np.linspace(start=0, stop=1.0, num=self.nx),
            np.linspace(start=0, stop=1.0, num=self.ny)
        )
        self.grid = np.concatenate((
            np.expand_dims(grid_x, axis=-1),
            np.expand_dims(grid_y, axis=-1)
        ), axis=-1).astype(np.float32).reshape(-1, 2)
        # get the distances between nodes and use that to get the node connections
        self.distances = cdist(self.grid)
        distance_cuttoff = 1.0 / (self.nx - 1)
        connections = np.where(np.logical_and(self.distances <= distance_cuttoff, self.distances > 0))
        self.edges = np.vstack(connections)
        # save info from config
        self.time_steps_in = config["TIME_STEPS_IN"]
        self.time_steps_out = config["TIME_STEPS_OUT"]
        # generate the indecies for where we can take samples from
        self.indecies_map = []
        for example_idx in range(self.array.shape[0]):
            for time_idx in range(0, self.array.shape[1], config["TIME_INT"]):
                if(time_idx < self.array.shape[1] - self.time_steps_in - self.time_steps_out):
                    self.indecies_map.append((example_idx, time_idx))
    
    def __len__(self):
        return len(self.indecies_map)

    def generate_example_shape(self):
        X, y, grid = self.__getitem__(0)
        return (X.shape, grid.shape)

    def __getitem__(self, idx):
        (exmaple_idx, time_idx) = self.indecies_map[idx]
        # get the observations
        X = self.array[exmaple_idx, time_idx:time_idx+self.time_steps_in, ...]
        y = self.array[exmaple_idx, time_idx+self.time_steps_in:time_idx+self.time_steps_in+self.time_steps_out, ...]
        # reshape the array so that time is last
        X = X.transpose([1, 2, 0])
        y = y.transpose([1, 2, 0])
        # return the data
        return X, y, self.grid, self.edges

def create_data_loader(array: np.ndarray, config: dict, shuffle: bool = True):
    dataset = PDEDataset(array, config)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=config['BATCH_SIZE'], shuffle=shuffle, num_workers=3, persistent_workers=True, pin_memory=False)
    return data_loader, len(dataset), dataset.generate_example_shape()