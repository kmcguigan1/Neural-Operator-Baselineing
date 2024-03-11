from dataclasses import dataclass
import numpy as np
import torch

from data_module.utils.data_reader import get_data_reader
from data_module.utils.data_processor import DataProcessor, GraphDataProcessor, DataContainer, GraphDataContainer
from data_module.utils.dataset import PDEDataset, GraphPDEDataset

@dataclass
class DataloaderContainer:
    """This is the class that manages the context for the data loader objects.
    Sometimes we need extra information about the datasets and this will be saved here"""
    dataloader: torch.utils.data.DataLoader
    image_size: int
    indecies: any  
    shuffling: bool

class DataModule(object):
    def __init__(self, config:dict):
        # modules to load the data and to process it
        self.data_reader = get_data_reader(config)  
        self.data_processor = self.get_data_processor(config)
        # information we need for the data
        self.dataset_kwargs = self.get_dataset_kwargs(config)
        self.batch_size = config['BATCH_SIZE']
        self.is_graph_loader = config.get('GRAPH_LOADER', False)

    def get_data_processor(self, config:dict):
        if(config.get('GRAPH_LOADER', False) == True):
            return GraphDataProcessor(config)
        return DataProcessor(config)

    def get_dataset_kwargs(self, config):
        kwargs = {
            'time_steps_in': config['TIME_STEPS_IN'],
            'time_steps_out': config['TIME_STEPS_OUT']
        }
        if(config.get('GRAPH_LOADER', False) == True):
            kwargs['n_neighbors'] = config['N_NEIGHBORS']
        if(config.get('SINGLE_SAMPLE_LOADER', False) == False):
            kwargs['time_interval'] = config['TIME_INTERVAL']
        else:
            kwargs['time_interval'] = -1
        return kwargs
        
    def get_dataset(self, data_container:DataContainer):
        if(self.is_graph_loader == False):
            return PDEDataset(data_container, **self.dataset_kwargs)
        if(self.is_graph_loader == True):
            return GraphPDEDataset(data_container, **self.dataset_kwargs)
        raise Exception("A valid combo to select a dataset was not specified")

    def pipeline(self, split:str, shuffle:bool=True, fit:bool=False, inference:bool=False):
        data = self.data_reader.load_data(split=split)
        data_container = self.data_processor.transform(data, split=split, fit=fit)
        dataset = self.get_dataset(data_container)
        return self._create_data_loader(dataset, shuffle=shuffle)

    def _create_data_loader(self, dataset, shuffle:bool=True):
        image_size = dataset.image_shape
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=3, persistent_workers=True, pin_memory=False)
        return DataloaderContainer(dataloader, image_size, dataset.indecies_map, shuffle)

    def get_training_data(self):
        train_dataset = self.pipeline(split='train', shuffle=True, fit=True)
        val_dataset = self.pipeline(split='val', shuffle=False)
        return train_dataset, val_dataset

    def get_test_data(self, split:str='test'):
        return self.pipeline(split=split, shuffle=False, inference=True)

    def transform_predictions(self, data:np.array, split:str=None):
        return self.data_processor.inverse_predictions(data, split=split)