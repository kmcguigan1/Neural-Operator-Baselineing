import os
import numpy as np
import torch
import scipy 
import h5py

from dataset.transforms import GausNorm, RangeNorm, PassNorm
from dataset.dataset import PDEDataset
from constants import DATA_PATH

def load_data(data_file:str, time_steps_in:int, time_steps_out:int):
    # shape (example, dim, dim, time)
    try:    
        data = scipy.io.loadmat(data_file)['u']
    except:
        data = h5py.File(data_file)['u']
        data = data[()]
        data = np.transpose(data, axes=range(len(data.shape) - 1, -1, -1))
    # make the data floats
    data = data.astype(np.float32)
    # get the data in the shape that we want it
    data = data[..., :time_steps_in+time_steps_out]
    return data

def get_normalizer(config):
    if(config['NORMALIZATION'] == 'gaussian'):
        return GausNorm()
    elif(config['NORMALIZATION'] == 'range'):
        return RangeNorm()
    else:
        return PassNorm()

def generate_grid(nx:int, ny:int):
    grid_x, grid_y = np.meshgrid(
        np.linspace(start=0, stop=1.0, num=nx),
        np.linspace(start=0, stop=1.0, num=ny)
    )
    grid = np.concatenate(
        (
            np.expand_dims(grid_x, axis=-1),
            np.expand_dims(grid_y, axis=-1)
        ), axis=-1
    ).astype(np.float32)
    return grid

def downsample_data(data:np.ndarray, grid:np.ndarray, ratio:int=1):
    if(ratio > 1): 
        data = data[:, ::ratio, ::ratio, :]
        grid = grid[::ratio, ::ratio, :]
    return data, grid

def cut_data(data:np.ndarray, grid:np.ndarray, patch_size:int):
    x_cut = data.shape[1] % patch_size
    y_cut = data.shape[2] % patch_size
    if(x_cut > 0):
        data = data[:, :-x_cut, :, :]
        grid = grid[:-x_cut, :, :]
    if(y_cut > 0):
        data = data[:, :, :-y_cut, :]
        grid = grid[:, :-y_cut, :]
    return data, grid

def split_data(data:np.ndarray):
    # get the data splits
    example_count = data.shape[0]
    train_split = int(0.75 * example_count)
    val_split = train_split + int(0.15 * example_count)
    # get the data paritions
    train_data = data[:train_split, ...]
    val_data = data[train_split:val_split, ...]
    test_data = data[val_split:, ...]
    return train_data, val_data, test_data

def pipeline(self, data:np.ndarray, fit:bool=False, shuffle:bool=True, downsample_ratio:int=None):
    data = self.apply_normalizer(data, split=split, fit=fit)
    grid = self.generate_grid(data.shape[1], data.shape[2])
    data, grid = self.downsample_data(data, grid, split=split, ratio=downsample_ratio)
    data, grid = self.cut_data(data, grid)
    dataset = PDEDataset(data, grid, self.time_steps_in, self.time_steps_out)
    return self.create_data_loader(dataset, shuffle=shuffle), data.shape[0], data.shape[1:-1]

def get_data_loaders(config:dict):
    # get the stuff we need from config
    data_file = os.path.join(DATA_PATH, config['DATA_FILE'])
    time_steps_in = config["TIME_STEPS_IN"]
    time_steps_out = config["TIME_STEPS_OUT"]
    batch_size = config['BATCH_SIZE']
    # read in the data
    data = load_data(data_file, time_steps_in, time_steps_out)
    train_grid = generate_grid(data.shape[1], data.shape[2])
    val_grid = train_grid.copy()
    test_grid = train_grid.copy()
    # transform the data
    transform = get_normalizer(config)
    train_data, val_data, test_data = split_data(data)
    train_data = transform.fit_transform(train_data)
    val_data = transform.transform(val_data)
    test_data = transform.transform(test_data)
    # downsample the data
    train_data, train_grid = downsample_data(train_data, train_grid)
    val_data, train_grid = downsample_data(val_data, val_grid)
    test_data, train_grid = downsample_data(test_data, test_grid)
    # cut the data if we need to
    if(False):
        train_data, train_grid = cut_data(train_data, train_grid)
        val_data, train_grid = cut_data(val_data, val_grid)
        test_data, train_grid = cut_data(test_data, test_grid)
    # create the datasets
    train_dataset = PDEDataset(train_data, train_grid, time_steps_in, time_steps_out)
    val_dataset = PDEDataset(val_data, val_grid, time_steps_in, time_steps_out)
    test_dataset = PDEDataset(test_data, test_grid, time_steps_in, time_steps_out)
    print(len(train_dataset), len(val_dataset), len(test_dataset))
    # get the dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, transform, len(train_dataset), train_dataset.image_size