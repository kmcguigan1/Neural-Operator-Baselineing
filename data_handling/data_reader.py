import os
import numpy as np 

from data_handling.data_loader import create_data_loader
from data_handling.graph_data_loader import create_graph_data_loader

def cut_data(config: dict, array: np.array):
    if(config['EXP_KIND'] == 'AFNO' or config['EXP_KIND'] == 'VIT'):
        x_cut = array.shape[-2] % config['PATCH_SIZE']
        y_cut = array.shape[-1] % config['PATCH_SIZE']
        if(x_cut > 0):
            array = array[..., :-x_cut, :]
        if(y_cut > 0):
            array = array[..., :-y_cut]
    return array

def load_dataset(config: dict):
    data_path = os.path.join('data', config['DATA_FILE'])
    with open(data_path, mode='rb') as f:
        train_data = np.load(f)
        val_data = np.load(f)
        test_data = np.load(f)
    train_data = cut_data(config, train_data)
    val_data = cut_data(config, val_data)
    test_data = cut_data(config, test_data)
    print(train_data.shape, val_data.shape, test_data.shape)
    return train_data, val_data, test_data

def get_train_data_loaders(config: dict):
    train_data, val_data, _ = load_dataset(config)
    # if the transfrom is not none then we fit it and stuff
    dataset_statistics = {
        'mean': train_data.mean(),
        'var': train_data.std(),
        'min': train_data.min(),
        'max': train_data.max()
    }
    print(f"Train Data mean {train_data.mean():.4f} var {train_data.std():.4f} min {train_data.min():.4f} max {train_data.max():.4f}")
    print(f"Val Data mean {val_data.mean():.4f} var {val_data.std():.4f} min {val_data.min():.4f} max {val_data.max():.4f}")
    # now build the dataloaders
    if(config['EXP_KIND'] == 'GNO'):
        train_data_loader, train_img_size = create_graph_data_loader(train_data, config, dataset_statistics, shuffle=True)
        val_data_loader, val_img_size = create_graph_data_loader(val_data, config, dataset_statistics, shuffle=False)
    else:
        train_data_loader, train_img_size = create_data_loader(train_data, config, dataset_statistics, shuffle=True)
        val_data_loader, val_img_size = create_data_loader(val_data, config, dataset_statistics, shuffle=False)
    return train_data_loader, val_data_loader, train_img_size, dataset_statistics

def get_test_data_loaders(config: dict, dataset_statistics:dict=None):
    train_data, val_data, test_data = load_dataset(config)
    print(f"Test Data mean {test_data.mean():.4f} var {test_data.std():.4f} min {test_data.min():.4f} max {test_data.max():.4f}")
    if(config['EXP_KIND'] == 'GNO'):
        train_data_loader, train_img_size = create_graph_data_loader(train_data, config, dataset_statistics, shuffle=True, inference_mode=True)
        val_data_loader, val_img_size = create_graph_data_loader(val_data, config, dataset_statistics, shuffle=False, inference_mode=True)
        test_data_loader, test_img_shape = create_graph_data_loader(test_data, config, dataset_statistics, shuffle=False, inference_mode=True)
    else:
        train_data_loader, train_img_size = create_data_loader(train_data, config, dataset_statistics, shuffle=True, inference_mode=True)
        val_data_loader, val_img_size = create_data_loader(val_data, config, dataset_statistics, shuffle=False, inference_mode=True)
        test_data_loader, test_img_shape = create_data_loader(test_data, config, dataset_statistics, shuffle=False, inference_mode=True)
    return train_data_loader, val_data_loader, test_data_loader