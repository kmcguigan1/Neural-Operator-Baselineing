import os
import numpy as np 

from data_handling.data_loader import create_data_loader
from utils.constants_handler import ConstantsObject

def cut_data(config: dict, array: np.array):
    if(config['METHODOLOGY'] == 'afno'):
        x_cut = array.shape[-2] % config['PATCH_SIZE'][0]
        y_cut = array.shape[-1] % config['PATCH_SIZE'][1]
        if(x_cut > 0):
            array = array[..., :-x_cut, :]
        if(y_cut > 0):
            array = array[..., :-y_cut]
    return array

def load_dataset(config: dict, constants_object: ConstantsObject):
    data_path = os.path.join(constants_object.DATA_PATH, config['DATA_FILE'])
    with open(data_path, mode='rb') as f:
        train_data = np.load(f)
        val_data = np.load(f)
        test_data = np.load(f)
    train_data = cut_data(config, train_data)
    val_data = cut_data(config, val_data)
    test_data = cut_data(config, test_data)
    return train_data, val_data, test_data

def get_train_data_loaders(config: dict, constants_object: ConstantsObject):
    train_data, val_data, _ = load_dataset(config, constants_object)
    train_data_loader, train_example_count, train_example_shape = create_data_loader(train_data, config, shuffle=True)
    val_data_loader, val_example_count, val_example_shape = create_data_loader(val_data, config, shuffle=False)
    return train_data_loader, val_data_loader, train_example_count, train_example_shape

def get_test_data_loader(config: dict, constants_object: ConstantsObject):
    _, _, test_data = load_dataset(config, constants_object)
    test_data_loader, test_example_count, test_example_shape = create_data_loader(test_data, config, shuffle=False)
    return test_data_loader