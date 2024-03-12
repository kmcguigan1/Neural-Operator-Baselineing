import numpy as np
import torch

class PDEDataset(torch.utils.data.Dataset):
    def __init__(self, data:np.ndarray, grid:np.ndarray, time_steps_in:int, time_steps_out:int):
        super().__init__()
        # save the data that we need
        self.time_steps_in = time_steps_in
        self.time_steps_out = time_steps_out
        self.data = data.copy().astype(np.float32)
        self.grid = grid.copy().astype(np.float32)
        self.image_size = self.data.shape[1:-1]
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # get the observations
        X = self.data[idx, ..., :self.time_steps_in]
        y = self.data[idx, ...,  self.time_steps_in:self.time_steps_in+self.time_steps_out]
        return X, y, self.grid