PARAMETERS = {
    'LATENT_DIMS': {
        'values': [16, 32, 64]
    },
    'MODES1': {
        'values': [8, 12, 16]
    },
    'MODES2': {
        'values': [8, 12, 16]
    },
    'DROP_RATE': {
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


# CONV_LSTM_PARAMS = {
#     'LATENT_DIMS': [16, 32, 64, 128],
#     'PROJ_KERNEL_SIZE': [3, 5, 7],
#     'KERNEL_SIZE': [3, 5, 7],
#     'DEPTH': [2, 4, 6],
#     'NORMALIZATION': ['None', 'gaus', 'range']
# }

# FNO_PARAMS = {
#     'LATENT_DIMS': [16, 32, 64, 128],
#     'MODES1': [8, 12, 16],
#     'MODES2': [8, 12, 16],
#     'DROP_RATE': [0.0, 0.2, 0.4],
#     'MLP_RATIO': [1, 2],
#     'DEPTH': [2, 4, 6],
#     'NORMALIZATION': ['None', 'gaus', 'range']
# }

# GNO_PARAMS = {
#     'LATENT_DIMS': [16, 32, 64, 128],
#     'DROP_RATE': [0.0, 0.2, 0.4],
#     'MLP_RATIO': [1, 2],
#     'DEPTH': [2, 4, 6],
#     'NORMALIZATION': ['None', 'gaus', 'range']
# }

# VIT_PARAMS = {
#     'LATENT_DIMS': [16, 32, 64, 128],
#     'PATCH_SIZE': [3, 5, 7],
#     'DEPTH': [2, 4, 6],
#     'NHEAD': [1, 2, 4],
#     'DROPOUT': [0.0, 0.2, 0.4],
#     'NORMALIZATION': ['None', 'gaus', 'range']
# }