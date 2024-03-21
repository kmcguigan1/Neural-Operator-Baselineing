import numpy as np
import torch

import scipy
import h5py

from data_module.PDEDataModule import PDEDataModule, PDEDataset

class EfficientPDEDataModule(PDEDataModule):
    def __init__(self, config:dict):
        super().__init__(config)
    
    def load_data(self, split:str):
        # shape (example, time, dim, dim)
        with h5py.File(self.data_file, "r") as f:
            data = f[f'{split}_u'][:]
        data = np.transpose(data, axes=(0,2,3,1))
        data = data[..., :self.time_steps_in+self.time_steps_out]
        return data
    
    def pipeline(self, split:str, shuffle:bool, inference:bool=False, downsample_ratio:int=None):
        assert shuffle == True or split != 'train' or inference == True
        # load and normalize the data
        data = self.load_data(split)
        if(split == 'train' and inference == False):
            self.train_example_count = data.shape[0]
            data = self.normalizer.fit_transform(data)
        else:
            data = self.normalizer.transform(data)
        # run the remaining things
        grid = self.generate_grid(nx=data.shape[1], ny=data.shape[2])
        data, grid = self.downsample_data(data, grid, ratio=downsample_ratio)
        data, grid = self.cut_data(data, grid)
        dataset = self.get_dataset(data, grid)
        print(f"{split} data shape is {data.shape}")
        if(split == 'train' and self.image_size is None):
            self.image_size = dataset.image_size
        return self.get_data_loader(dataset, shuffle=shuffle)
    
    def get_data_loader(self, dataset, shuffle:bool, inference:bool=False):
        if(inference):
            return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size*4, shuffle=False, num_workers=4, persistent_workers=False)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=4, persistent_workers=True)

    def get_training_data(self):
        train_loader = self.pipeline(split='train', shuffle=True)
        val_loader = self.pipeline(split='val', shuffle=False)
        return train_loader, val_loader

    def get_testing_data(self, split:str='test', downsample_ratio:int=None):
        return self.pipeline(split=split, shuffle=False, inference=True, downsample_ratio=downsample_ratio), None