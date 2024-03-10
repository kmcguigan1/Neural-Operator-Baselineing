import unittest
from copy import copy

from data_module.data_module import DataModule
from data_module.utils.data_reader import MatDataReader, NpzDataReader

class TestSimpleDataModuleMethods(unittest.TestCase):
    def test_create_module(self):
        data_module = DataModule(SIMPLE_CONFIG)

    def test_training_data(self):
        data_module = DataModule(SIMPLE_CONFIG)
        train_loader, val_loader = data_module.get_training_data()

class TestDataReader(unittest.TestCase):
    def test_npz_reading(self):
        reader = NpzDataReader(SIMPLE_CONFIG)
        train = reader.load_data(split='train')
        print("Train data shape: ", train.shape)

    def test_mat_reading(self):
        reader = MatDataReader(MAT_CONFIG)
        train = reader.load_data(split='train')
        print("Train data shape: ", train.shape)




SIMPLE_CONFIG = {
    'TIME_STEPS_IN': 7,
    'TIME_STEPS_OUT': 28,
    'TIME_INTERVAL': 1,
    'BATCH_SIZE': 8,
    'DATA_READER': 'NPZ',
    'DATA_FILE': 'diffusion_varying_sinusoidal_init_fixed_diffusivity_non_periodic_boundaries.npz',
    'NORMALIZATION': 'none',
    'DATA_PROC_KIND': 'multi_sample',
}

MAT_CONFIG = {
    'TIME_STEPS_IN': 7,
    'TIME_STEPS_OUT': 28,
    'TIME_INTERVAL': 1,
    'BATCH_SIZE': 8,
    'DATA_READER': 'NPZ',
    'DATA_FILE': 'ns_data_V1e-4_N20_T50_R256test.mat',
    'NORMALIZATION': 'none',
    'DATA_PROC_KIND': 'multi_sample',
}

if __name__ == '__main__':
    unittest.main()