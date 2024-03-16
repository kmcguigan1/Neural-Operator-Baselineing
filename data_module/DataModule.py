import os
from abc import ABC, abstractmethod

import torch
import numpy as np

from constants import DATA_PATH

class DataModule(ABC):
    def __init__(self, config:dict):
        self.data_file = os.path.join(DATA_PATH, config['DATA_FILE'])
        self.time_steps_in = config['TIME_STEPS_IN']
        self.time_steps_out = config['TIME_STEPS_OUT']
        self.batch_size = config['BATCH_SIZE']
        self.downsample_ratio = config.get('DOWNSAMPLE_RATIO', 1)
        self.patch_size = config['PATCH_SIZE'] if config.get('CUT_TO_PATCH', False) else None
        self.normalizer = self.get_normalizer(config)
        self.image_size = None
        self.test_data = None
        self.test_indecies = None
        self.train_example_count = None

    def get_normalizer(self, config:dict):
        if(config['NORMALIZATION'] == 'gaussian'):
            return GausNorm()
        elif(config['NORMALIZATION'] == 'range'):
            return RangeNorm()
        else:
            return PassNorm()

    def split_data(self, data:np.ndarray):
        # get the data splits
        example_count = data.shape[0]
        train_split = int(0.75 * example_count)
        val_split = train_split + int(0.15 * example_count)
        # get the data paritions
        train_data = data[:train_split, ...]
        val_data = data[train_split:val_split, ...]
        test_data = data[val_split:, ...]
        self.test_indecies = np.arange(data.shape[0])[val_split:]
        return train_data, val_data, test_data

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

    def downsample_data(self, data:np.ndarray, grid:np.ndarray, ratio:int=None):
        if(ratio is None):
            ratio = self.downsample_ratio
        if(ratio > 1): 
            data = data[:, ::ratio, ::ratio, :]
            grid = grid[::ratio, ::ratio, :]
        return data, grid

    def cut_data(self, data:np.ndarray, grid:np.ndarray):
        if(self.patch_size is None):
            return data, grid
        x_cut = data.shape[1] % self.patch_size
        y_cut = data.shape[2] % self.patch_size
        if(x_cut > 0):
            data = data[:, :-x_cut, :, :]
            grid = grid[:-x_cut, :, :]
        if(y_cut > 0):
            data = data[:, :, :-y_cut, :]
            grid = grid[:, :-y_cut, :]
        return data, grid

    def get_data_loader(self, dataset, shuffle:bool):
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)#, num_workers=4, persistent_workers=True)

    def get_training_data(self):
        # load and split the data
        data = self.load_data()
        train_data, val_data, self.test_data = self.split_data(data)
        self.train_example_count = train_data.shape[0]
        # fit the normalizer
        train_data = self.normalizer.fit_transform(train_data)
        val_data = self.normalizer.transform(val_data)
        self.test_data = self.normalizer.transform(self.test_data)
        # get the data loaders
        train_loader = self.pipeline(train_data, split='train', shuffle=True)
        val_loader = self.pipeline(val_data, split='val', shuffle=False)
        return train_loader, val_loader

    def get_testing_data(self, downsample_ratio:int=None):
        return self.pipeline(self.test_data, split='test', shuffle=False, downsample_ratio=downsample_ratio), self.test_indecies

    def inverse_transform(self, array:np.ndarray):
        return self.normalizer.inverse_transform(array)

    def load_data(self):
        raise NotImplementedError('This is the base class, extend this method in a subclass')

    def get_dataset(self, *args):
        raise NotImplementedError('This is the base class, extend this method in a subclass')

    def pipeline(self, data:np.ndarray, split:str, shuffle:bool, downsample_ratio:int=None):
        raise NotImplementedError('This is the base class, extend this method in a subclass')

"""
Data transformations to be used
"""
class DataTransform(ABC):
    def __init__(self):
        self.pointwise = False
    @abstractmethod
    def fit(self, array:np.array) -> None:
        pass
    @abstractmethod
    def transform(self, array:np.array) -> np.array:
        pass
    @abstractmethod
    def inverse_transform(self, array:np.array) -> np.array:
        pass
    def fit_transform(self, array:np.array) -> np.array:
        self.fit(array)
        return self.transform(array)

class GausNorm(DataTransform):
    def __init__(self):
        super().__init__()
        self.mean = None
        self.var = None
    def fit(self, array:np.array) -> None:
        self.mean = np.mean(array)
        self.var = np.std(array)
    def transform(self, array:np.array) -> np.array:
        return (array - self.mean) / self.var
    def inverse_transform(self, array:np.array) -> np.array:
        return array * self.var + self.mean

class RangeNorm(DataTransform):
    def __init__(self):
        super().__init__()
        self.lower = None
        self.upper = None
    def fit(self, array:np.array) -> None:
        self.lower = np.min(array)
        self.upper = np.max(array)
    def transform(self, array:np.array) -> np.array:
        return (array - self.lower) / (self.upper - self.lower)
    def inverse_transform(self, array:np.array) -> np.array:
        return array * (self.upper - self.lower) + self.lower
    
class PassNorm(DataTransform):
    def __init__(self):
        super().__init__()
    def fit(self, array:np.array) -> None:
        pass
    def transform(self, array:np.array) -> np.array:
        return array
    def inverse_transform(self, array:np.array) -> np.array:
        return array