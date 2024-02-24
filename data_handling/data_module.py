import os
import numpy as np 
import torch

from einops import rearrange

from constants import DATA_PATH
from data_handling.datasets import PDEDataset, GraphPDEDataset
from data_handling.transforms import GausNorm, RangeNorm, DataTransform

class DataModule(object):
    def __init__(self, config:dict):
        self.data_file = os.path.join(DATA_PATH, config['DATA_FILE'])
        self.batch_size = config['BATCH_SIZE']
        self.time_steps_in = config["TIME_STEPS_IN"]
        self.time_steps_out = config["TIME_STEPS_OUT"]
        self.time_int = config["TIME_INT"]
        # create a storage for the image sizes
        self.image_sizes = {}
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

    def get_image_shape(self):
        """This function provides the default image shape. This is 
        needed for Conv LSTM models to setup. Most models should be shape agnostic
        if they are operator based. This can map to files or things.

        Returns:
            int: The default image size.
        """
        return (64,64)
    
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

    def _create_dataset(self, data:np.array):
        return PDEDataset(data, self.time_steps_in, self.time_steps_out, self.time_int)

    def create_data_loader(self, data:np.array, shuffle:bool=True, split:str=None, get_image_shape:bool=False):
        dataset = self._create_dataset(data)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=4, persistent_workers=True, pin_memory=False)
        # save the split if we need it
        if(split is not None):
            self.image_sizes[split] = dataset.image_shape
        # see if we get the image shape too
        if(get_image_shape):
            return data_loader, dataset.image_shape
        return data_loader

    def get_training_data(self):
        train_data = self.load_data()
        val_data = self.load_data(split='val')
        # fit the transfrom if we have that
        if(self.transform is not None):
            train_data = self.transform.fit_transform(train_data)
            val_data = self.transform.transform(val_data)
        # make the data loader
        train_loader, train_image_shape = self.create_data_loader(train_data, shuffle=True, get_image_shape=True)
        val_loader = self.create_data_loader(val_data, shuffle=False)
        return train_loader, val_loader, train_image_shape

    def get_test_data(self, split:str='test', return_metadata:bool=False):
        data = self.load_data(split=split)
        if(self.transform is not None):
            data = self.transform.transform(data)
        datalodaer = self.create_data_loader(data, shuffle=False, split=split)
        if(return_metadata):
            return self.create_data_loader(data, shuffle=False, split=split)
        return self.create_data_loader(data, shuffle=False, split=split)

    def transform_predictions(self, data:np.array, split:str=None, no_time_dim:bool=False):
        if(self.transform is not None):
            return self.transform.inverse_transform(data)
        return data

class GraphDataModule(DataModule):
    def __init__(self, config:dict):
        super().__init__(config)
        self.neighbors_method = config['NEIGHBORS']

    def _create_dataset(self, data:np.array):
        return GraphPDEDataset(data, self.time_steps_in, self.time_steps_out, self.time_int, self.neighbors_method)

    def transform_predictions(self, data:np.array, split:str=None, no_time_dim:bool=False):
        data = super().transform_predictions(data, no_time_dim=no_time_dim)
        image_size = self.image_sizes[split]
        example_count = data.shape[0]
        if(no_time_dim):
            data = rearrange(data, 'b (h w) -> b h w', b=example_count, h=image_size[0], w=image_size[1])
        else:
            time_step_count = data.shape[-1]
            data = rearrange(data, 'b (h w) c -> b h w c', b=example_count, h=image_size[0], w=image_size[1], c=time_step_count)
        return data

def get_data_module(config:dict):
    if('GRAPH_DATA_LOADER' in config.keys() and config['GRAPH_DATA_LOADER'] == True):
        return GraphDataModule(config)
    return DataModule(config)