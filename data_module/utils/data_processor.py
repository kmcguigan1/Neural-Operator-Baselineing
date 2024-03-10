from abc import ABC, abstractmethod
import numpy as np 

from scipy.spatial.distance import cdist

EPSILON = 1e-5

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

def downsample_data(data:np.ndarray, grid:np.ndarray, ratio:int=1, targets:np.ndarray=None):
    if(ratio > 1): 
        data = data[:, ::ratio, ::ratio, :]
        grid = grid[::ratio, ::ratio, :]
        if(targets is not None):
            targets = targets[:, ::ratio, ::ratio, :]
    if(targets is not None):
        return data, targets, grid
    return data, grid

def cut_data(data:np.ndarray, grid:np.ndarray, patch_size:int, targets:np.ndarray=None):
    x_cut = data.shape[1] % patch_size
    y_cut = data.shape[2] % patch_size
    if(x_cut > 0):
        data = data[:, :-x_cut, :, :]
        grid = grid[:-x_cut, :, :]
        if(targets is not None):
            targets = targets[:, :-x_cut, :, :]
    if(y_cut > 0):
        data = data[:, :, :-y_cut, :]
        grid = grid[:, :-y_cut, :]
        if(targets is not None):
            targets = targets[:, :, :-y_cut, :]
    if(targets is not None):
        return data, targets, grid
    return data, grid

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

    def fit(self, data:np.ndarray, split:str, inference:bool=False):
        assert split == 'train'
        assert inference == False
        self.image_size = (data.shape[1:-1])
        data = self.normalizer.fit_transform(data, split=split)
        grid = generate_grid(*self.image_size)
        data, grid = downsample_data(data, grid, ratio=self.downsampling_ratios[split])
        if(self.patch_size is not None):
            data, grid = cut_data(data, grid, self.patch_size)
        return data, grid 

    def transform(self, data:np.ndarray, split:str, downsampling_ratio=None, inference:bool=False):
        data = self.normalizer.transform(data, split=split)
        grid = generate_grid(*self.image_size)
        data, grid = downsample_data(data, grid, ratio=self.downsampling_ratios[split] if downsampling_ratio is None else downsampling_ratio)
        if(self.patch_size is not None):
            data, grid = cut_data(data, grid, self.patch_size)
        return data, grid 

    def inverse_predictions(self, preds:np.ndarray, split:str):
        return self.normalizer.inverse_transform(preds, split=split)

class SingleSampleDataProcessor(DataProcessor):
    def __init__(self, config:dict):
        super().__init__(config)
        self.time_steps_in = config["TIME_STEPS_IN"]
        self.time_steps_out = config["TIME_STEPS_OUT"]

    def get_normalizer(self, config):
        if(config['NORMALIZATION'] == 'gaussian'):
            self.normalizer = GausNorm()
        elif(config['NORMALIZATION'] == 'range'):
            self.normalizer = RangeNorm()
        elif(config['NORMALIZATION'] == 'pointwise_gaussian'):
            self.normalizer = PointGausNorm()
        elif(config['NORMALIZATION'] == 'pointwise_range'):
            self.normalizer = PointRangeNorm()
        else:
            self.normalizer = PassNorm()

    def split_data(self, data:np.ndarray):
        x = data[..., :self.time_steps_in]
        y = data[..., self.time_steps_in:self.time_steps_in+self.time_steps_out]
        return x, y

    def fit(self, data:np.ndarray, split:str, inference:bool=False):
        assert split == 'train'
        assert inference == False
        self.image_size = (data.shape[1:-1])
        if(self.normalizer.pointwise == False):
            data = self.normalizer.fit_transform(data, split=split)
            x, y = self.split_data(data)
        else:
            x, y = self.split_data(data)
            x, y = self.normalizer.fit_transform(x, y, split=split)
        grid = generate_grid(*self.image_size)
        x, y, grid = downsample_data(x, grid, ratio=self.downsampling_ratios[split], targets=y)
        if(self.patch_size is not None):
            x, y, grid = cut_data(x, grid, self.patch_size, targets=y)
        return x, y, grid 

    def transform(self, data:np.ndarray, split:str, inference:bool=False, downsampling_ratio=None):
        if(self.normalizer.pointwise == False):
            data = self.normalizer.transform(data, split=split)
            x, y = self.split_data(data)
        else:
            x, y = self.split_data(data)
            x, y = self.normalizer.transform(x, y, split=split)
        grid = generate_grid(*self.image_size)
        x, y, grid = downsample_data(x, grid, targets=y, ratio=self.downsampling_ratios[split] if downsampling_ratio is None else downsampling_ratio)
        if(self.patch_size is not None):
            x, y, grid = cut_data(x, grid, self.patch_size, targets=y)
        return x, y, grid 

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

class GraphDataProcessor(DataProcessor):
    def __init__(self, config:dict):
        super().__init__(config)
        self.n_neighbors = config['N_NEIGHBORS']

    def fit(self, data:np.ndarray, split:str, inference:bool=False):
        assert split == 'train'
        assert inference == False
        self.image_size = (data.shape[1:-1])
        data = self.normalizer.fit_transform(data, split=split)
        grid = generate_grid(*self.image_size)
        data, grid = downsample_data(data, grid, ratio=self.downsampling_ratios[split])
        if(self.patch_size is not None):
            data, grid = cut_data(data, grid, self.patch_size)
        data, grid, edges, edge_features = generate_edges_and_features(grid, n_neighbors=self.n_neighbors)
        return data, grid, edges, edge_features

    def transform(self, data:np.ndarray, split:str, inference:bool=False, downsampling_ratio=None):
        data = self.normalizer.transform(data, split=split)
        grid = generate_grid(*self.image_size)
        data, grid = downsample_data(data, grid, ratio=self.downsampling_ratios[split] if downsampling_ratio is None else downsampling_ratio)
        if(self.patch_size is not None):
            data, grid = cut_data(data, grid, self.patch_size)
        data, grid, edges, edge_features = generate_edges_and_features(grid, n_neighbors=self.n_neighbors)
        return data, grid, edges, edge_features

class GraphSingleSampleDataProcessor(SingleSampleDataProcessor):
    def __init__(self, config:dict):
        super().__init__(config)
        self.n_neighbors = config['N_NEIGHBORS']

    def fit(self, data:np.ndarray, split:str, inference:bool=False):
        assert split == 'train'
        assert inference == False
        self.image_size = (data.shape[1:-1])
        if(self.normalizer.pointwise == False):
            data = self.normalizer.fit_transform(data, split=split)
            x, y = self.split_data(data)
        else:
            x, y = self.split_data(data)
            x, y = self.normalizer.fit_transform(x, y, split=split)
        grid = generate_grid(*self.image_size)
        x, y, grid = downsample_data(x, grid, ratio=self.downsampling_ratios[split], targets=y)
        if(self.patch_size is not None):
            x, y, grid = cut_data(x, grid, self.patch_size, targets=y)
        grid, edges, edge_features = generate_edges_and_features(grid, n_neighbors=self.n_neighbors)
        return x, y, grid, edges, edge_features

    def transform(self, data:np.ndarray, split:str, inference:bool=False, downsampling_ratio=None):
        if(self.normalizer.pointwise == False):
            data = self.normalizer.transform(data, split=split)
            x, y = self.split_data(data)
        else:
            x, y = self.split_data(data)
            x, y = self.normalizer.transform(x, y, split=split)
        grid = generate_grid(*self.image_size)
        x, y, grid = downsample_data(x, grid, targets=y, ratio=self.downsampling_ratios[split] if downsampling_ratio is None else downsampling_ratio)
        if(self.patch_size is not None):
            x, y, grid = cut_data(x, grid, self.patch_size, targets=y)
        grid, edges, edge_features = generate_edges_and_features(grid, n_neighbors=self.n_neighbors)
        return x, y, grid, edges, edge_features

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

class PointGausNorm(DataTransform):
    def __init__(self):
        super().__init__()
        self.pointwise = True
        self.mean = {}
        self.var = {}
    def fit(self, array:np.array, split:str) -> None:
        self.mean[split] = np.mean(array, axis=(1,2,3), keepdims=True)
        self.var[split] = np.std(array, axis=(1,2,3), keepdims=True)
    def transform(self, array:np.array, split:str) -> np.array:
        return (array - self.mean[split]) / self.var[split]
    def inverse_transform(self, array:np.array, split:str) -> np.array:
        return array * self.var[split] + self.mean[split]

class PointRangeNorm(DataTransform):
    def __init__(self):
        super().__init__()
        self.pointwise = True
        self.lower = {}
        self.upper = {}
    def fit(self, array:np.array, split:str) -> None:
        self.lower[split] = np.minimum(array, axis=(1,2,3), keepdims=True)
        self.upper[split] = np.maximum(array, axis=(1,2,3), keepdims=True)
    def transform(self, array:np.array, split:str) -> np.array:
        return (array - self.lower[split]) / (self.upper[split] - self.lower[split])
    def inverse_transform(self, array:np.array, split:str) -> np.array:
        return array * (self.upper[split] - self.lower[split]) + self.lower[split]