PARAMETERS = {
    'LATENT_DIMS': {
        'values': [16, 32, 64, 128]
    },
    'PATCH_SIZE': {
        'values': [3, 5, 7]
    },
    'NUM_BLOCKS': {
        'values': [1, 2, 4]
    },
    'DROP_RATE': {
        'values': [0.0, 0.2, 0.4]
    },
    'DROP_PATH_RATE': {
        'values': [0.0, 0.2, 0.4]
    },
    'MLP_RATIO': {
        'values': [1, 2]
    },
    'DEPTH': {
        'values': [2, 4, 6]
    },
    'NORMALIZATION': {
        'values': [None, 'gaus', 'range']
    },
}