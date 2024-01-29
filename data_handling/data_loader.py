import numpy as np
import torch

class PDEDataset(torch.utils.data.Dataset):
    def __init__(self, array: np.ndarray, config: dict):
        super().__init__()
        # save the data that we need
        self.array = array.copy().astype(np.float32) # (example, time, x, y)
        self.return_grid = False
        if("USE_GRID" in config.keys() and config["USE_GRID"] == True):
            self.return_grid = True
            grid_x, grid_y = np.meshgrid(
                np.linspace(start=0, stop=1.0, num=array.shape[-2]),
                np.linspace(start=0, stop=1.0, num=array.shape[-1])
            )
            self.grid = np.concatenate(
                (
                    np.expand_dims(grid_x, axis=-1),
                    np.expand_dims(grid_y, axis=-1)
                ), axis=-1
            ).astype(np.float32)
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
        X, y, grid = self.__getitem__(0)
        return (X.shape, grid.shape)

    def __getitem__(self, idx):
        (exmaple_idx, time_idx) = self.indecies_map[idx]
        # get the observations
        X = self.array[exmaple_idx, time_idx:time_idx+self.time_steps_in, ...]
        y = self.array[exmaple_idx, time_idx+self.time_steps_in:time_idx+self.time_steps_in+self.time_steps_out, ...]
        # reshape the array so that time is last
        X = X.transpose([1, 2, 0])
        y = y.transpose([1, 2, 0])
        if(self.return_grid):
            return X, y, self.grid
        return X, y

def create_data_loader(array: np.ndarray, config: dict, shuffle: bool = True):
    dataset = PDEDataset(array, config)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=config['BATCH_SIZE'], shuffle=shuffle, num_workers=3, persistent_workers=True, pin_memory=False)
    return data_loader, len(dataset), dataset.generate_example_shape()