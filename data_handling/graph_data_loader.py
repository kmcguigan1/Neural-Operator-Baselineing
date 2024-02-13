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
        self.array = self.array.reshape(array.shape[0], array.shape[1], -1)
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
        distances = cdist(self.grid, self.grid)
        # get the distance cuttoff
        single_dist = 1.0 / (self.nx - 1)
        if(config["NEIGHBORS"] == "radial"):
            distance_cuttoff = 1.9 * (single_dist)
        connections = np.where(np.logical_and(distances < distance_cuttoff, distances > 0))
        self.edges = np.vstack(connections).astype(np.int32)
        # get the features of each edge such as the distance and the cos and sin of the angle
        self.edge_features = distances[connections[0], connections[1]].reshape(-1, 1)
        self.edge_features = np.concatenate((
            self.edge_features,
            self.grid[connections[0], :],
            self.grid[connections[1], :] 
        ), axis=-1)
        # save info from config
        self.time_steps_in = config["TIME_STEPS_IN"]
        self.time_steps_out = config["TIME_STEPS_OUT"]
        # generate the indecies for where we can take samples from
        self.indecies_map = []
        for example_idx in range(self.array.shape[0]):
            for time_idx in range(0, self.array.shape[1], config["TIME_INT"]):
                if(time_idx < self.array.shape[1] - self.time_steps_in - self.time_steps_out):
                    self.indecies_map.append((example_idx, time_idx))
        # # make everything tensors
        # self.array = torch.tensor(self.array, dtype=torch.float32)
        # self.grid = torch.tensor(self.grid, dtype=torch.float32)
        # self.edges = torch.tensor(self.edges, dtype=torch.int32)
        # self.edge_features = torch.tensor(self.edge_features, dtype=torch.float32)

    def __len__(self):
        return len(self.indecies_map)

    def generate_example_shape(self):
        outputs = self.__getitem__(0)
        X, y, grid, edges, edge_features = outputs
        return (X.shape, grid.shape, edges.shape, edge_features.shape)

    def __getitem__(self, idx):
        (exmaple_idx, time_idx) = self.indecies_map[idx]
        # get the observations
        X = self.array[exmaple_idx, time_idx:time_idx+self.time_steps_in, ...]
        y = self.array[exmaple_idx, time_idx+self.time_steps_in:time_idx+self.time_steps_in+self.time_steps_out, ...]
        # reshape the array so that time is last
        X = X.transpose([1, 0])
        y = y.transpose([1, 0])
        # return the data
        return X, y, self.grid, self.edges, self.edge_features

def create_graph_data_loader(array: np.ndarray, config: dict, shuffle: bool = True):
    dataset = GraphPDEDataset(array, config)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=config['BATCH_SIZE'], shuffle=shuffle, num_workers=3, persistent_workers=True, pin_memory=False)
    return data_loader, len(dataset), dataset.generate_example_shape(), (dataset.nx, dataset.ny)

def test():
    # Ex, Time, X, Y
    example_data = np.random.normal(loc=0, scale=1, size=(4, 20, 4, 4))
    config = {"TIME_STEPS_IN":2, "TIME_STEPS_OUT":3, "TIME_INT":1, "NEIGHBORS":'radial'}
    dataset = GraphPDEDataset(example_data, config)
    X, y, grid, edges, edge_features = dataset.__getitem__(idx=0)
    print(X.shape, y.shape, grid.shape, edges.shape, edge_features.shape)

if __name__ == '__main__':
    test()


# print("Edge Features")
#         print(self.edge_features.shape)
#         print(self.edge_features)
#         dx = grid[connections[1], 0] - grid[connections[0], 0]
#         dy = grid[connections[1], 1] - grid[connections[0], 1]
#         tan_theta = dy / dx
#         theta = np.arctan(tan_theta)
#         print(theta)