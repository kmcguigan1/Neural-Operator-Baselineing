import numpy as np
import torch

class PDEDataset(torch.utils.data.Dataset):
    def __init__(self, array: np.ndarray, config: dict):
        super().__init__()
        # save the data that we need
        self.array = array.copy().astype(np.float32) # (example, time, x, y)
        # grid_x, grid_y = np.meshgrid(
        #     np.linspace(start=0, stop=1.0, num=array.shape[-2]),
        #     np.linspace(start=0, stop=1.0, num=array.shape[-1])
        # )
        # self.grid = np.concatenate(
        #     (
        #         np.expand_dims(grid_x, axis=-1),
        #         np.expand_dims(grid_y, axis=-1)
        #     ), axis=-1
        # )
        # save info from config
        self.time_steps_in = config["TIME_STEPS_IN"]
        self.time_steps_out = config["TIME_STEPS_OUT"]
        # generate the indecies for where we can take samples from
        self.indecies_map = []
        for example_idx in range(self.array.shape[0]):
            for time_idx in range(0, self.array.shape[1], config["TIME_INT"]):
                if(time_idx < self.array.shape[1] - self.time_steps_in - self.time_steps_out):
                    self.indecies_map.append((example_idx, time_idx))
    
    def __len__(self):
        return len(self.indecies_map)

    def generate_example_shape(self):
        X, _ = self.__getitem__(0)
        return X.shape

    # add the grid back in
    # then make sure that we work
    # this is a simpler way of doing the last code base so we can keep it but start thinking better practices
    # everything should be modular and selectable
    # the data reader reads a data object thing for each dataset and returns the information needed for a training
    # and other splits loader
    # then we have a module that takes all this in and creates the datasets
    # there should be a global handler that uses sub handlers for the specific datasets
    # we can then pass

    # actually these are different enough diff code bases is good, these have less complex covariates or weird things 
    # we want to do anyways
    # this is cleaner

    # we could have covariate or head or things that like all models pull from
    # for now this simple rebuild was fine, it was fast to do and good code is slow to design

    # the grid is a must for all examples, don't add timesteps, but 
    # no covariates for now
    def __getitem__(self, idx):
        (exmaple_idx, time_idx) = self.indecies_map[idx]
        # get the observations
        X = self.array[exmaple_idx, time_idx:time_idx+self.time_steps_in, ...]
        y = self.array[exmaple_idx, time_idx+self.time_steps_in:time_idx+self.time_steps_in+self.time_steps_out, ...]
        # reshape the array so that time is last
        X = X.transpose([1, 2, 0])
        y = y.transpose([1, 2, 0])
        return X, y

def create_data_loader(array: np.ndarray, config: dict, shuffle: bool = True):
    dataset = PDEDataset(array, config)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=config['BATCH_SIZE'], shuffle=shuffle, num_workers=3, persistent_workers=True, pin_memory=False)
    return data_loader, len(dataset), dataset.generate_example_shape()