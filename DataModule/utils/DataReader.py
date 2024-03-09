class BaseDataReader(object):
    def __init__(self, config:dict):
        self.data_file = config['DATA_FILE']
    def load_data(self, split:str):
        pass

class NpzDataReader(BaseDataReader):
    def __init__(self, config:dict):
        super().__init__(config)
    def load_data(self, split:str):
        with np.load(self.data_file) as file_data:
            array = file_data[f'{split}_data']
        return array

class MatDataReader(BaseDataReader):
    def __init__(self, config:dict):
        super().__init__(config)
        self.data = {}
    def _extract_data(self):
        # a is (example, dim, dim)
        # u is (example, dim, dim, time)
        # t is (1, time)
        data = scipy.io.loadmat('data/ns_data_V1e-4_N20_T50_R256test.mat')
        # get the splits
        example_count = data['u'].shape[0]
        train_split = int(0.75 * example_count)
        val_split = int(0.15 * example_count) + train_split
        # load the actual data
        self.data['train'] = data['u'][:train_split, ...]
        self.data['val'] = data['u'][train_split:val_split, ...]
        self.data['test'] = data['u'][val_split:, ...]
    def load_data(self, split:str):
        if(split not in self.data.keys()):
            self._extract_data()
        return self.data[split]

def get_data_reader(config:dict):
    if(config['DATA_READER'] == 'NPZ'):
        return NpzDataReader(config)
    elif(config['DATA_READER'] == 'MAT'):
        return MatDataReader(config)
    raise Exception(f"Invalid data reader type {config['DATA_READER']}")