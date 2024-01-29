import os
import numpy as np 

from data_handling.data_loader import create_data_loader
from utils.constants_handler import ConstantsObject

def load_dataset(config: dict, constants_object: ConstantsObject):
    data_path = os.path.join(constants_object.DATA_PATH, config['DATA_FILE'])
    with open(data_path, mode='rb') as f:
        train_data = np.load(f)[:4, ...]
        val_data = np.load(f)[:2, ...]
        test_data = np.load(f)[:2, ...]
    return train_data, val_data, test_data

def get_data_loaders(config: dict, constants_object: ConstantsObject):
    train_data, val_data, test_data = load_dataset(config, constants_object)
    train_data_loader, train_example_count, train_example_shape = create_data_loader(train_data, config, shuffle=True)
    val_data_loader, val_example_count, val_example_shape = create_data_loader(val_data, config, shuffle=False)
    test_data_loader, test_example_count, test_example_shape = create_data_loader(test_data, config, shuffle=False)
    return train_data_loader, val_data_loader, test_data_loader, train_example_count, train_example_shape
