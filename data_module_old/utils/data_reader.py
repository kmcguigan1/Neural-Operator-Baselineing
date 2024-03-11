## Read data in the format (example, dim, dim, time)
import os
import numpy as np
import scipy

from constants import DATA_PATH

import gc

class BaseDataReader(object):
    def __init__(self, config:dict):
        self.data_file = os.path.join(DATA_PATH, config['DATA_FILE'])
        self.single_sample = config['SINGLE_SAMPLE_LOADER']
        self.time_steps_in = config['TIME_STEPS_IN']
        self.time_steps_out = config['TIME_STEPS_OUT']
    def cut_data(self, data):
        if(self.single_sample == True):
            data = data[..., :self.time_steps_in+self.time_steps_out]
        return data
    def load_data(self, split:str):
        raise NotImplementedError('This is the base data reader class')

class NpzDataReader(BaseDataReader):
    def __init__(self, config:dict):
        super().__init__(config)
    def load_data(self, split:str):
        with np.load(self.data_file) as file_data:
            array = file_data[f'{split}_data']
        # we want the data in (example, dim, dim, time)
        array = array.transpose(0, 2, 3, 1)
        array = self.cut_data(array)
        return array

class MatDataReader(BaseDataReader):
    def __init__(self, config:dict):
        super().__init__(config)
        self.setup_data()

    def setup_data(self):
        # a is (example, dim, dim)
        # u is (example, dim, dim, time)
        # t is (1, time)
        data = scipy.io.loadmat(self.data_file)['u']
        # cut the data
        data = self.cut_data(data)
        example_count = data.shape[0]
        self.train_split = int(0.75 * example_count)
        self.val_split = int(0.15 * example_count) + self.train_split
        # get the data splits
        data = self.cut_data(data)
        self.train_data = data[:self.train_split, ...]
        self.val_data = data[self.train_split:self.val_split, ...]
        self.test_data = data[self.val_split:, ...]
    def load_data(self, split:str):
        # load the actual data
        if(split == 'train'):
            return self.train_data
        elif(split == 'val'):
            return self.val_data
        elif(split == 'test'):
            return self.test_data
        else:
            raise Exception(f"Invalid split of {split}")

def get_data_reader(config:dict) -> BaseDataReader:
    if(config['DATA_READER'] == 'NPZ'):
        return NpzDataReader(config)
    elif(config['DATA_READER'] == 'MAT'):
        return MatDataReader(config)
    raise Exception(f"Invalid data reader type {config['DATA_READER']}")