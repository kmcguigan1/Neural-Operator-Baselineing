from DataModule.data_readers.DataReader import get_data_reader

class DataModule(object):
    def __init__(self, config:dict):
        self.data_reader = get_data_reader(config)