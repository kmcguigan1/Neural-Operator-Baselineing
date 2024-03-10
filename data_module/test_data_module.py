import unittest
from copy import copy
import numpy as np

from data_module.data_module import DataModule
from data_module.utils.data_reader import MatDataReader, NpzDataReader

def data_loader_to_all_samples(dataloader):
    xs, ys = [], []
    for batch in dataloader.dataloader:
        x, y = batch[0], batch[1]
        xs.append(x.detach().numpy())
        ys.append(y.detach().numpy())
    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    return xs, ys

class TestSimpleDataModuleMethods(unittest.TestCase):
    def test_create_module(self):
        data_module = DataModule(SIMPLE_CONFIG)

    def test_training_data(self):
        data_module = DataModule(SIMPLE_CONFIG)
        train_loader, val_loader = data_module.get_training_data()

    def test_training_data_distributions(self):
        none_train_loader, _ = DataModule(SIMPLE_CONFIG).get_training_data()
        xs, ys = data_loader_to_all_samples(none_train_loader)
        print("No transform x y shapes: ", xs.shape, ys.shape)
        print("No transform x y means: ", xs.mean(), xs.std(), ys.mean(), ys.std())

        gaus_conf = copy(SIMPLE_CONFIG)
        gaus_conf.update({'NORMALIZATION':'gaussian'})
        gaus_train_loader, _ = DataModule(gaus_conf).get_training_data()
        gxs, gys = data_loader_to_all_samples(gaus_train_loader)
        print("Gaus transform x y shapes: ", gxs.shape, gys.shape)
        print("Gaus transform x y means: ", gxs.mean(), gxs.std(), gys.mean(), gys.std())

        range_conf = copy(SIMPLE_CONFIG)
        range_conf.update({'NORMALIZATION':'range'})
        range_train_loader, _ = DataModule(range_conf).get_training_data()
        rxs, rys = data_loader_to_all_samples(range_train_loader)
        print("Range transform x y shapes: ", rxs.shape, rys.shape)
        print("Range transform x y means: ", rxs.mean(), rxs.std(), rys.mean(), rys.std())

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
}

MAT_CONFIG = {
    'TIME_STEPS_IN': 7,
    'TIME_STEPS_OUT': 28,
    'TIME_INTERVAL': 1,
    'BATCH_SIZE': 8,
    'DATA_READER': 'NPZ',
    'DATA_FILE': 'ns_data_V1e-4_N20_T50_R256test.mat',
    'NORMALIZATION': 'none',
}

if __name__ == '__main__':
    unittest.main()