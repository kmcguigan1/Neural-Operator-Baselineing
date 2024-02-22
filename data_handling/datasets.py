import numpy as np
import torch

class PDEDataset(torch.utils.data.Dataset):
    def __init__(self, array:np.ndarray, time_steps_in:int, time_steps_out:int, time_int:int):
        super().__init__()
        # save the data that we need
        self.array = array.copy().astype(np.float32) # (example, time, x, y)
        self.image_shape = self.array.shape[-2:]
        self.time_steps_in = time_steps_in
        self.time_steps_out = time_steps_out
        self.time_int = time_int
        # get the grid
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
        # generate the indecies for where we can take samples from
        self.indecies_map = []
        for example_idx in range(self.array.shape[0]):
            for time_idx in range(0, self.array.shape[1], self.time_int):
                if(time_idx < self.array.shape[1] - self.time_steps_in - self.time_steps_out):
                    self.indecies_map.append((example_idx, time_idx))
        # transpose the array to not do it every time we read data
        self.array = self.array.transpose([0, 2, 3, 1]) # (example, x, y, time)
    
    def __len__(self):
        return len(self.indecies_map)

    def __getitem__(self, idx):
        (exmaple_idx, time_idx) = self.indecies_map[idx]
        # get the observations
        X = self.array[exmaple_idx, ..., time_idx:time_idx+self.time_steps_in]
        y = self.array[exmaple_idx, ...,  time_idx+self.time_steps_in:time_idx+self.time_steps_in+self.time_steps_out]
        return X, y, self.grid