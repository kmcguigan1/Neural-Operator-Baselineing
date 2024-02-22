import os 
import numpy as np

base = 'bad_format_data'
for file in os.listdir(base):
    with open(os.path.join(base, file), mode='rb') as f:
        train = np.load(f)
        val = np.load(f)
        test = np.load(f)

        train_meta = np.load(f)
        val_meta = np.load(f)
        test_meta = np.load(f)

    np.savez(
        file.replace('.npy', '.npz'),
        train_data=train,
        val_data=val,
        test_data=test,
        train_meta=train_meta,
        val_meta=val_meta,
        test_meta=test_meta
    )
    print(file)

