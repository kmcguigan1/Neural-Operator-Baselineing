import numpy as np
import torch

import scipy
import h5py

from data_module.PDEDataModule import PDEDataModule

class GraphPDEDataModule(PDEDataModule):
    def __init__(self, config:dict):
        super().__init__(config)

    def get_dataset(self, data:np.ndarray, grid:np.ndarray):
        return PDEDataset(data, grid, self.time_steps_in, self.time_steps_out)

    def pipeline(self, data:np.ndarray, split:str, shuffle:bool, downsample_ratio:int=None):
        assert shuffle == True or split != 'train'
        grid = self.generate_grid(nx=data.shape[1], ny=data.shape[2])
        data, grid = self.downsample_data(data, grid, ratio=downsample_ratio)
        data, grid = self.cut_data(data, grid)
        dataset = self.get_dataset(data, grid)
        if(split == 'train' and self.image_size is None):
            self.image_size = dataset.image_size
        return self.get_data_loader(dataset, shuffle=shuffle)