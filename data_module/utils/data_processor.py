from abc import ABC, abstractmethod
import numpy as np 

from scipy.spatial.distance import cdist
from data_module.data_module import DataContainer, GraphDataContainer

EPSILON = 1e-5

class DataProcessor(object):
    def __init__(self, config:dict):
        self.get_normalizer(config)
        # default image shape
        # this is the grid size to build since we can't realistically upsample in experiments
        self.image_size = None
        # figure out downsampling
        self.downsampling_ratios = {
            'train': config.get("TRAIN_DOWNSAMP_RATIO", 1),
            'val': config.get("VAL_DOWNSAMP_RATIO", 1),
            'test': config.get("TEST_DOWNSAMP_RATIO", 1),
        }
        # figure out patch cutting
        self.patch_size = None
        if(config.get("CUT_PATCHES", False) == True):
            self.patch_size = config['PATCH_SIZE']

    def get_normalizer(self, config):
        if(config['NORMALIZATION'] == 'gaussian'):
            self.normalizer = GausNorm()
        elif(config['NORMALIZATION'] == 'range'):
            self.normalizer = RangeNorm()
        else:
            self.normalizer = PassNorm()

    def inverse_predictions(self, preds:np.ndarray, split:str):
        return self.normalizer.inverse_transform(preds, split=split)

    def transform(self, data:np.ndarray, split:str, fit:bool=False, inference:bool=False, downsampling_ratio:int=None):
        if(fit):
            assert split == 'train'
            assert inference == False
            self.image_size = (data.shape[1:-1])
            data = self.normalizer.fit_transform(data, split=split)
        else:
            data = self.normalizer.transform(data, split=split)
        grid = generate_grid(*self.image_size)
        data, grid = downsample_data(data, grid, ratio=self.downsampling_ratios[split] if downsampling_ratio is None else downsampling_ratio)
        if(self.patch_size is not None):
            data, grid = cut_data(data, grid, self.patch_size)
        return DataContainer(data, grid)

class GraphDataProcessor(DataProcessor):
    def __init__(self, config:dict):
        super().__init__(config)
        self.n_neighbors = config['N_NEIGHBORS']

    def transform(self, data:np.ndarray, split:str, fit:bool=False, inference:bool=False, downsampling_ratio:int=None):
        if(fit):
            assert split == 'train'
            assert inference == False
            self.image_size = (data.shape[1:-1])
            data = self.normalizer.fit_transform(data, split=split)
        else:
            data = self.normalizer.transform(data, split=split)
        grid = generate_grid(*self.image_size)
        data, grid = downsample_data(data, grid, ratio=self.downsampling_ratios[split] if downsampling_ratio is None else downsampling_ratio)
        if(self.patch_size is not None):
            data, grid = cut_data(data, grid, self.patch_size)
        data, grid, edges, edge_features = generate_edges_and_features(grid, n_neighbors=self.n_neighbors)
        return GraphDataContainer(data, grid, edges, edge_features)


def generate_grid(nx, ny):
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

def downsample_data(data:np.ndarray, grid:np.ndarray, ratio:int=1):
    if(ratio > 1): 
        data = data[:, ::ratio, ::ratio, :]
        grid = grid[::ratio, ::ratio, :]
    return data, grid

def cut_data(data:np.ndarray, grid:np.ndarray, patch_size:int):
    x_cut = data.shape[1] % patch_size
    y_cut = data.shape[2] % patch_size
    if(x_cut > 0):
        data = data[:, :-x_cut, :, :]
        grid = grid[:-x_cut, :, :]
    if(y_cut > 0):
        data = data[:, :, :-y_cut, :]
        grid = grid[:, :-y_cut, :]
    return data, grid

def generate_edges_and_features(grid:np.ndarray, n_neighbors:int=1):
    # get the shape of the image grid before we do anything
    image_shape = grid.shape[:-1]
    # reshape the grid to be token space, we don't need it to be in a grid
    grid = grid.reshape(-1, 2)
    # get the distances between the grid cells
    distances = cdist(grid, grid)
    # get the distance cuttoff for neighbors
    single_dist = 1.0 / ((image_shape[0] + image_shape[1]) * 0.5 - 1)
    distance_cuttoff = (n_neighbors + 0.9) * single_dist
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

"""
Data transformations to be used
"""
class DataTransform(ABC):
    def __init__(self):
        self.pointwise = False
    @abstractmethod
    def fit(self, array:np.array, split:str) -> None:
        pass
    @abstractmethod
    def transform(self, array:np.array, split:str) -> np.array:
        pass
    @abstractmethod
    def inverse_transform(self, array:np.array, split:str) -> np.array:
        pass
    def fit_transform(self, array:np.array, split:str) -> np.array:
        self.fit(array, split)
        return self.transform(array, split)

class GausNorm(DataTransform):
    def __init__(self):
        super().__init__()
        self.mean = None
        self.var = None
    def fit(self, array:np.array, split:str) -> None:
        self.mean = np.mean(array)
        self.var = np.std(array)
    def transform(self, array:np.array, split:str) -> np.array:
        return (array - self.mean) / self.var
    def inverse_transform(self, array:np.array, split:str) -> np.array:
        return array * self.var + self.mean

class RangeNorm(DataTransform):
    def __init__(self):
        super().__init__()
        self.lower = None
        self.upper = None
    def fit(self, array:np.array, split:str) -> None:
        self.lower = np.min(array)
        self.upper = np.max(array)
    def transform(self, array:np.array, split:str) -> np.array:
        return (array - self.lower) / (self.upper - self.lower)
    def inverse_transform(self, array:np.array, split:str) -> np.array:
        return array * (self.upper - self.lower) + self.lower

class PassNorm(DataTransform):
    def __init__(self):
        super().__init__()
    def fit(self, array:np.array, split:str) -> None:
        pass
    def transform(self, array:np.array, split:str) -> np.array:
        return array
    def inverse_transform(self, array:np.array, split:str) -> np.array:
        return array