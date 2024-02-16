import numpy as np
import torch

class PDEDataset(torch.utils.data.Dataset):
    def __init__(self, array: np.ndarray, config: dict, dataset_statistics:dict, inference_mode:bool=False):
        super().__init__()
        # save the data that we need
        self.dataset_statistics = dataset_statistics
        self.transform = None
        if("NORMALIZER" in config.keys()):
            self.transform = config["NORMALIZER"]
        self.inference_mode = inference_mode
        self.array = array.copy().astype(np.float32) # (example, time, x, y)
        self.image_shape = self.array.shape[-2:]
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
        if(self.return_grid):
            X, y, grid = self.__getitem__(0)
            return (X.shape, grid.shape)
        X, y = self.__getitem__(0)
        return X.shape

    def apply_transform(self, arr):
        if(self.transform == 'gaus'):
            arr = (arr - self.dataset_statistics['mean']) / self.dataset_statistics['var']
        elif(self.transform == 'range'):
            arr = (arr - self.dataset_statistics['min']) / (self.dataset_statistics['max'] - self.dataset_statistics['min'])
        return arr

    def __getitem__(self, idx):
        (exmaple_idx, time_idx) = self.indecies_map[idx]
        # get the observations
        X = self.array[exmaple_idx, time_idx:time_idx+self.time_steps_in, ...]
        y = self.array[exmaple_idx, time_idx+self.time_steps_in:time_idx+self.time_steps_in+self.time_steps_out, ...]
        # apply transforms
        X = self.apply_transform(X)
        if(not self.inference_mode):
            y = self.apply_transform(y)
        # reshape the array so that time is last
        X = X.transpose([1, 2, 0])
        y = y.transpose([1, 2, 0])
        if(self.return_grid):
            return X, y, self.grid
        return X, y

def create_data_loader(array: np.ndarray, config: dict, dataset_statistics:dict, shuffle: bool = True, inference_mode:bool=False):
    dataset = PDEDataset(array, config, dataset_statistics, inference_mode=inference_mode)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=config['BATCH_SIZE'], shuffle=shuffle, num_workers=3, persistent_workers=True, pin_memory=False)
    return data_loader, len(dataset), dataset.generate_example_shape(), dataset.image_shape