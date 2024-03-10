from dataclasses import dataclass
import numpy as np
import torch

from data_module.utils.data_reader import get_data_reader
from data_module.utils.data_processor import DataProcessor, SingleSampleDataProcessor, GraphDataProcessor, GraphSingleSampleDataProcessor
from data_module.utils.dataset import PDEDataset, GraphPDEDataset, SingleSamplePDEDataset, SingleSampleGraphPDEDataset

def get_data_module(config:dict):
    if(config.get('SINGLE_SAMPLE_LOADER', False) == True and config.get('GRAPH_LOADER', False) == False):
        return SingleSampleDataModule(config)
    if(config.get('SINGLE_SAMPLE_LOADER', False) == True and config.get('GRAPH_LOADER', False) == True):
        return SingleSampleGraphDataModule(config)
    if(config.get('SINGLE_SAMPLE_LOADER', False) == False and config.get('GRAPH_LOADER', False) == False):
        return DataModule(config)
    if(config.get('SINGLE_SAMPLE_LOADER', False) == False and config.get('GRAPH_LOADER', False) == True):
        return GraphDataModule(config)
    raise Exception("A valid combo to select a data module was not specified")

@dataclass
class DataloaderContainer:
    """This is the class that manages the context for the data loader objects.
    Sometimes we need extra information about the datasets and this will be saved here"""
    dataloader: torch.utils.data.DataLoader
    image_size: int  
    indecies: any

class DataModule(object):
    def __init__(self, config:dict):
        # modules to load the data and to process it
        self.data_reader = get_data_reader(config)  
        self.data_processor = self.get_data_processor(config)
        # information we need for the data
        self.time_steps_in = config['TIME_STEPS_IN']
        self.time_steps_out = config['TIME_STEPS_OUT']
        self.time_int = config['TIME_INTERVAL']
        self.batch_size = config['BATCH_SIZE']

    def get_data_processor(self, config:dict):
        return DataProcessor(config)

    def pipeline(self, split:str, shuffle:bool=True, fit:bool=False, inference:bool=False):
        data = self.data_reader.load_data(split=split)
        if(fit):
            data, grid = self.data_processor.fit(data, split=split)
        else:
            data, grid = self.data_processor.transform(data, split=split)
        dataset = PDEDataset(data, grid, self.time_steps_in, self.time_steps_out, self.time_int)
        return self._create_data_loader(dataset, shuffle=shuffle)

    def _create_data_loader(self, dataset, shuffle:bool=True):
        image_size = dataset.image_shape
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=3, persistent_workers=True, pin_memory=False)
        return DataloaderContainer(dataloader, image_size, dataset.indecies_map)

    def get_training_data(self):
        train_dataset = self.pipeline(split='train', shuffle=True, fit=True)
        val_dataset = self.pipeline(split='val', shuffle=False)
        return train_dataset, val_dataset

    def get_test_data(self, split:str='test'):
        return self.pipeline(split=split, shuffle=False, inference=True)

    def transform_predictions(self, data:np.array, split:str=None):
        return self.data_processor.inverse_predictions(data, split=split)

class GraphDataModule(DataModule):
    def __init__(self, config:dict):
        super().__init__(config)

    def get_data_processor(self, config:dict):
        return GraphDataProcessor(config)

    def pipeline(self, split:str, shuffle:bool=True, fit:bool=False, inference:bool=False):
        data = self.data_reader.load_data(split=split)
        if(fit):
            data, grid, edges, edge_feats = self.data_processor.fit(data, split=split)
        else:
            data, grid, edges, edge_feats = self.data_processor.transform(data, split=split)
        dataset = GraphPDEDataset(data, grid, edges, edge_feats, self.time_steps_in, self.time_steps_out, self.time_int, self.neighbors_method)
        return self._create_data_loader(dataset, shuffle=shuffle)

class SingleSampleDataModule(DataModule):
    def __init__(self, config:dict):
        super().__init__(config)
    
    def get_data_processor(self, config:dict):
        return SingleSampleDataProcessor(config)

    def pipeline(self, split:str, shuffle:bool=True, fit:bool=False, inference:bool=False):
        data = self.data_reader.load_data(split=split)
        if(fit):
            x, y, grid = self.data_processor.fit(data, split=split)
        else:
            x, y, grid = self.data_processor.transform(data, split=split)
        dataset = SingleSamplePDEDataset(x, y, grid)
        return self._create_data_loader(dataset, shuffle=shuffle)

class SingleSampleGraphDataModule(GraphDataModule):
    def __init__(self, config:dict):
        super().__init__(config)

    def get_data_processor(self, config:dict):
        return GraphSingleSampleDataProcessor(config)

    def pipeline(self, split:str, shuffle:bool=True, fit:bool=False, inference:bool=False):
        data = self.data_reader.load_data(split=split)
        if(fit):
            x, y, grid, edges, edge_feats = self.data_processor.fit(data, split=split)
        else:
            x, y, grid, edges, edge_feats = self.data_processor.transform(data, split=split)
        dataset = SingleSampleGraphPDEDataset(x, y, grid, edges, edge_feats)
        return self._create_data_loader(dataset, shuffle=shuffle)