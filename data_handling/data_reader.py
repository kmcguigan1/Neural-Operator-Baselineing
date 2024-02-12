import os
import numpy as np 

from utils.constants_handler import ConstantsObject
from data_handling.data_loader import create_data_loader
from data_handling.graph_data_loader import create_graph_data_loader
from data_handling.transforms import GausNorm, RangeNorm, DataTransform

def cut_data(config: dict, array: np.array):
    if(config['METHODOLOGY'] == 'afno'):
        x_cut = array.shape[-2] % config['PATCH_SIZE'][0]
        y_cut = array.shape[-1] % config['PATCH_SIZE'][1]
        if(x_cut > 0):
            array = array[..., :-x_cut, :]
        if(y_cut > 0):
            array = array[..., :-y_cut]
    return array

def apply_transforms(train_data:np.array, val_data:np.array, test_data:np.array, config:dict) -> tuple:
    transform = None
    if('NORMALIZER' in config.keys() and config['NORMALIZER'] == 'gaus'):
        transform = GausNorm()
        train_data = transform.fit_transform(train_data)
        val_data = transform.transform(val_data)
        test_data = transform.transform(test_data)
    elif('NORMALIZER' in config.keys() and config['NORMALIZER'] == 'range'):
        transform = RangeNorm()
        train_data = transform.fit_transform(train_data)
        val_data = transform.transform(val_data)
        test_data = transform.transform(test_data)
    return train_data, val_data, test_data, transform

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
    # get the transfromer that we are using on the object
    transform = None
    if('NORMALIZER' in config.keys() and config['NORMALIZER'] == 'gaus'):
        transform = GausNorm()
    elif('NORMALIZER' in config.keys() and config['NORMALIZER'] == 'range'):
        transform = RangeNorm()
    # if the transfrom is not none then we fit it and stuff
    if(transform is not None and np.logical_or('PER_INSTANCE' not in config.keys(), config['PER_INSTANCE'] == False)):
        train_data = transform.fit_transform(train_data)
        val_data = transform.transform(val_data)
    # now build the dataloaders
    if(constants_object.EXP_KIND == 'GNO'):
        train_data_loader, train_example_count, train_example_shape = create_graph_data_loader(train_data, config, shuffle=True)
        val_data_loader, val_example_count, val_example_shape = create_graph_data_loader(val_data, config, shuffle=False)
    else:
        train_data_loader, train_example_count, train_example_shape = create_data_loader(train_data, config, shuffle=True)
        val_data_loader, val_example_count, val_example_shape = create_data_loader(val_data, config, shuffle=False)
    return train_data_loader, val_data_loader, train_example_count, train_example_shape, transform

def get_test_data_loader(config: dict, constants_object: ConstantsObject, transform:DataTransform):
    _, _, test_data = load_dataset(config, constants_object)
    if(transform is not None and np.logical_or('PER_INSTANCE' not in config.keys(), config['PER_INSTANCE'] == False)):
        test_data = transform.transform(test_data)
    test_data_loader, test_example_count, test_example_shape = create_data_loader(test_data, config, shuffle=False)
    return test_data_loader