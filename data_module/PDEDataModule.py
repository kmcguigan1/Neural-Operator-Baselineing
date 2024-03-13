import numpy as np
import torch

import scipy
import h5py

from data_module.DataModule import DataModule

class PDEDataModule(DataModule):
    def __init__(self, config:dict):
        super().__init__(config)
    
    def load_data(self):
        # shape (example, dim, dim, time)
        try:    
            data = scipy.io.loadmat(self.data_file)['u']
        except:
            data = h5py.File(self.data_file)['u']
            data = data[()]
            data = np.transpose(data, axes=range(len(data.shape) - 1, -1, -1))
        # make the data floats
        data = data.astype(np.float32)
        # get the data in the shape that we want it
        data = data[..., :self.time_steps_in+self.time_steps_out]
        data = data[:100, ...]
        return data

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

class PDEDataset(torch.utils.data.Dataset):
    def __init__(self, data:np.ndarray, grid:np.ndarray, time_steps_in:int, time_steps_out:int):
        super().__init__()
        # save the data that we need
        self.time_steps_in = time_steps_in
        self.time_steps_out = time_steps_out
        self.data = data.copy().astype(np.float32)
        self.grid = grid.copy().astype(np.float32)
        self.image_size = self.data.shape[1:-1]
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # get the observations
        X = self.data[idx, ..., :self.time_steps_in]
        y = self.data[idx, ...,  self.time_steps_in:self.time_steps_in+self.time_steps_out]
        return X, y, self.grid