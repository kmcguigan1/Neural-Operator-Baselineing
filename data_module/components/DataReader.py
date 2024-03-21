import os
import numpy as np
import scipy
import h5py

from constants import DATA_PATH

class PDEDataReader(object):
    def __init__(self, config:dict):
        super().__init__()
        self.data_file = os.path.join(DATA_PATH, config['DATA_FILE'])
        self.time_steps_in = config['TIME_STEPS_IN']
        self.time_steps_out = config['TIME_STEPS_OUT']
        self.test_indecies = None

    def split_data(self, data:np.ndarray):
        # get the data splits
        example_count = data.shape[0]
        train_split = int(0.70 * example_count)
        val_split = train_split + int(0.15 * example_count)
        # get the data paritions
        train_data = data[:train_split, ...]
        val_data = data[train_split:val_split, ...]
        test_data = data[val_split:, ...]
        self.test_indecies = np.arange(data.shape[0])[val_split:]
        return train_data, val_data, test_data

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
        return data
    
    def get_training_data(self):
        data = self.read_data()
        train_data, val_data, self.test_data = self.split_data(data)
        return train_data, val_data

    def get_testing_data(self, split:str='test'):
        assert split == 'test'
        return self.test_data 

class CustomPDEDataReader(PDEDataReader):
    def __init__(self, config:dict):
        super().__init__(config)

    def load_data(self, split:str):
        # shape (example, time, dim, dim)
        with h5py.File(self.data_file, "r") as f:
            data = f[f'{split}_u'][:]
        data = np.transpose(data, axes=(0,2,3,1))
        data = data[..., :self.time_steps_in+self.time_steps_out]
        return data

    def get_training_data(self):
        train_data = self.load_data(split='train')
        val_data = self.load_data(split='val')
        return train_data, val_data

    def get_testing_data(self, split:str='test'):
        return self.load_data(split=split) 
