from abc import ABC, abstractmethod
import numpy as np
import torch

class DataModule(object):
    def __init__(self, config:dict):
        super().__init__()
        self.normalizer = self.get_normalizer(config)
        self.time_steps_in = config['TIME_STEPS_IN']
        self.time_steps_out = config['TIME_STEPS_OUT']
        self.batch_size = config['BATCH_SIZE']

    def get_normalizer(self, config):
        if(config['NORMALIZATION'] == 'gaussian'):
            self.normalizer = GausNorm()
        elif(config['NORMALIZATION'] == 'range'):
            self.normalizer = RangeNorm()
        else:
            self.normalizer = None

    def apply_normalizer(self, data:np.ndarray, split:str, fit:bool=False):
        if(self.normalizer is None):
            return data
        if(fit):
            return self.normalizer.fit_transform(data, split=split)
        return self.normalizer.transform(data, split=split)

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
        if(patch_size == -1):
            return data, grid
        x_cut = data.shape[1] % patch_size
        y_cut = data.shape[2] % patch_size
        if(x_cut > 0):
            data = data[:, :-x_cut, :, :]
            grid = grid[:-x_cut, :, :]
        if(y_cut > 0):
            data = data[:, :, :-y_cut, :]
            grid = grid[:, :-y_cut, :]
        return data, grid
    
    def create_data_loader(self, dataset, shuffle:bool=True):
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=1, persistent_workers=True, pin_memory=False)
    
    def transform_predictions(self, data:np.array):
        if(self.normalizer is not None):
            return self.normalizer.inverse_transform(data)
        return data
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