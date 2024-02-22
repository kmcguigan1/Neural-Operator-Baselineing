import os
import numpy as np 
import torch

from constants import DATA_PATH
from data_handling.datasets import PDEDataset
from data_handling.transforms import GausNorm, RangeNorm, DataTransform

class DataModule(object):
    def __init__(self, config:dict):
        self.data_file = os.path.join(DATA_PATH, config['DATA_FILE'])
        self.batch_size = config['BATCH_SIZE']
        self.time_steps_in = config["TIME_STEPS_IN"]
        self.time_steps_out = config["TIME_STEPS_OUT"]
        self.time_int = config["TIME_INT"]
        # transform
        if(config['NORMALIZATION'] == 'gaus'):
            self.transform = GausNorm()
        elif(config['NORMALIZATION'] == 'range'):
            self.transform = RangeNorm()
        else:
            self.transform = None
        # patching
        self.patch_cutting = None
        if(config['EXP_KIND'] in ['AFNO', 'VIT']):
            self.patch_cutting = config['PATCH_SIZE']
    
    def cut_data(self, array:np.array):
        if(self.patch_cutting is not None):
            x_cut = array.shape[-2] % config['PATCH_SIZE']
            y_cut = array.shape[-1] % config['PATCH_SIZE']
            if(x_cut > 0):
                array = array[..., :-x_cut, :]
            if(y_cut > 0):
                array = array[..., :-y_cut]
        return array
            
    def load_data(self, split:str='train'):
        file_data = np.load(self.data_file)
        array = file_data[f'{split}_data']
        array = self.cut_data(array)
        return array

    def create_data_loader(self, data:np.array, shuffle:bool=True):
        dataset = PDEDataset(data, self.time_steps_in, self.time_steps_out, self.time_int)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=4, persistent_workers=True, pin_memory=False)

    def get_training_data(self):
        train_data = self.load_data()
        val_data = self.load_data(split='val')
        # fit the transfrom if we have that
        if(self.transform is not None):
            train_data = self.transform.fit_transform(train_data)
            val_data = self.transform.transform(val_data)
        # make the data loader
        train_loader = self.create_data_loader(train_data, shuffle=True)
        val_loader = self.create_data_loader(val_data, shuffle=False)
        return train_loader, val_loader

    def get_test_data(self, split:str='test'):
        data = self.load_data(split=split)
        if(self.transform is not None):
            data = self.transform.transform(data)
        return self.create_data_loader(data, shuffle=False)

    def transform_predictions(self, data:np.array):
        if(self.transform is not None):
            return self.transform.inverse_transform(data)
        return data

class GraphDataModule(DataModule):
    def __init__(self, config:dict):
        super().__init__(config)