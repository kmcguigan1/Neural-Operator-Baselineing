import os 
import numpy as np

# base = 'bad_format_data'
# for file in os.listdir(base):
file = 'burgers_64_period_init_x_non_period_y_non_period_varying_nu'
with open(f'{file}.npy', mode='rb') as f:
    train = np.load(f)
    val = np.load(f)
    test = np.load(f)

    train_meta = np.load(f)
    val_meta = np.load(f)
    test_meta = np.load(f)

np.savez(
    f'{file}.npz',
    train_data=train,
    val_data=val,
    test_data=test,
    train_meta=train_meta,
    val_meta=val_meta,
    test_meta=test_meta
)
print(file)

