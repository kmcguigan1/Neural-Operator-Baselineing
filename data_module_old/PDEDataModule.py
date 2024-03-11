import numpy as np
import scipy
from data_module.DataModule import GausNorm, RangeNorm, PassNorm

class PDEDataModule(object):
    def __init__(self, config:dict):
        super().__init__()
        # figure out patch cutting
        self.patch_size = None
        if(config.get("CUT_PATCHES", False) == True):
            self.patch_size = config['PATCH_SIZE']

    def load_data(self):
        data = scipy.io.loadmat(self.data_file)['u']

    def get_normalizer(self, config):
        if(config['NORMALIZATION'] == 'gaussian'):
            self.normalizer = GausNorm()
        elif(config['NORMALIZATION'] == 'range'):
            self.normalizer = RangeNorm()
        else:
            self.normalizer = PassNorm()

    def generate_grid(self, nx:int, ny:int):
        grid_x, grid_y = np.meshgrid(
            np.linspace(start=0, stop=1.0, num=nx),
            np.linspace(start=0, stop=1.0, num=ny)
        )
        grid = np.concatenate(
            (
                np.expand_dims(grid_x, axis=-1),
                np.expand_dims(grid_y, axis=-1)
            ), axis=-1
        ).astype(np.float32)
        return grid

    def downsample_data(self, data:np.ndarray, grid:np.ndarray, ratio:int=1):
        if(ratio > 1): 
            data = data[:, ::ratio, ::ratio, :]
            grid = grid[::ratio, ::ratio, :]
        return data, grid

    def cut_data(self, data:np.ndarray, grid:np.ndarray, patch_size:int):
        x_cut = data.shape[1] % patch_size
        y_cut = data.shape[2] % patch_size
        if(x_cut > 0):
            data = data[:, :-x_cut, :, :]
            grid = grid[:-x_cut, :, :]
        if(y_cut > 0):
            data = data[:, :, :-y_cut, :]
            grid = grid[:, :-y_cut, :]
        return data, grid


class GraphPDEDataModule(PDEDataModule):
    def __init__(self, config:dict):
        super().__init__(config)

    def generate_edges_and_features(self, grid:np.ndarray):
        # get the shape of the image grid before we do anything
        image_shape = grid.shape[:-1]
        # reshape the grid to be token space, we don't need it to be in a grid
        grid = grid.reshape(-1, 2)
        # get the distances between the grid cells
        distances = cdist(grid, grid)
        # get the distance cuttoff for neighbors
        single_dist = 1.0 / ((image_shape[0] + image_shape[1]) * 0.5 - 1)
        distance_cuttoff = (self.n_neighbors + 0.9) * single_dist
        # get the connections within our distance matrix
        connections = np.where(np.logical_and(distances < distance_cuttoff, distances > 0))
        edges = np.vstack(connections).astype(np.int32)
        # get the features of each edge such as the distance and the cos and sin of the angle
        edge_features = distances[connections[0], connections[1]].reshape(-1, 1)
        edge_features = np.concatenate((
            edge_features,
            grid[connections[0], :],
            grid[connections[1], :] 
        ), axis=-1).astype(np.float32)
        return grid, edges, edge_features