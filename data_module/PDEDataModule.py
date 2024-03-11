import numpy as np
import scipy
import h5py
import torch

from data_module.DataModule import DataModule


class PDEDataModule(DataModule):
    def __init__(self, config:dict):
        super().__init__(config)
    
    def load_data(self):
        # shape (example, dim, dim, time)
        try:    
            data = scipy.io.loadmat(self.data_file)['u']
        except:
            data = h5py.File(self.file_path)['u']
            data = data[()]
            data = np.transpose(data, axes=range(len(data.shape) - 1, -1, -1))
        # make the data floats
        data = data.astype(np.float32)
        # get the data in the shape that we want it
        data = data[..., :self.time_steps_in+self.time_steps_out]
        return data
    
    def split_data(self, data:np.ndarray):
        # get the data splits
        example_count = data.shape[0]
        train_split = int(0.75 * example_count)
        val_split = int(0.15 * example_count)
        # get the data paritions
        train_data = data[:train_split, ...]
        val_data = data[train_split:val_split, ...]
        test_data = data[val_split:, ...]
        return train_data, val_data, test_data
    
    def pipeline(self, data:np.ndarray, split:str, fit:bool=False, shuffle:bool=True, downsample_ratio:int=1):
        data = self.apply_normalizer(data, split=split, fit=fit)
        grid = self.generate_grid(data.shape[1], data.shape[2])
        data, grid = self.downsample_data(data, grid, ratio=downsample_ratio)
        data, grid = self.cut_data(self, data, grid, patch_size=self.patch_size)
        dataset = PDEDataset(data, grid, self.time_steps_in, self.time_steps_out)
        return self.create_data_loader(dataset, shuffle=shuffle)
    
    def get_data_loaders(self):
        data = self.load_data()
        train_data, val_data, test_data = self.split_data(data)
        train_loader = self.pipeline(train_data, split='train', fit=True, shuffle=True)
        val_loader = self.pipeline(val_data, split='val', shuffle=False)
        test_loader = self.pipeline(test_data, split='test', shuffle=False)
        return train_loader, val_loader, test_loader
    
class PDEDataset(torch.utils.data.Dataset):
    def __init__(self, data:np.ndarry, grid:np.ndarray, time_steps_in:int, time_steps_out:int):
        super().__init__()
        # save the data that we need
        self.time_steps_in = time_steps_in
        self.time_steps_out = time_steps_out
        self.data = data.astype(np.float32)
        self.grid = grid.astype(np.float32)
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # get the observations
        X = self.data[idx, ..., :self.time_steps_in]
        y = self.data[idx, ...,  self.time_steps_in:self.time_steps_in+self.time_steps_out]
        return X, y, self.grid

    
