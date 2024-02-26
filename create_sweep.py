from copy import copy
import wandb

BASE_PARAMETERS = {
    'SEED': {'value': 1999},
    'DATA_FILE': {'value': 'burgers_64_x_period_y_period_varying_nu.npz'},
    'TIME_STEPS_IN': {'value': 7},
    'TIME_STEPS_OUT': {'value': 28},
    'BATCH_SIZE': {'value': 16},
    'TIME_INT': {'value': 1},
    'LEARNING_RATE': {'values': [0.01, 0.001]},
    'OPTIMIZER': {'value': {'KIND':'adam'}},
    'SCHEDULER': {'value': {'KIND':'reducer', 'FACTOR':0.1, 'PATIENCE':2}},
    'EPOCHS': {'value': 100},
    'LOSS': {'values': ['MSE', 'L1NORM']}
}

GNO_PARAMETERS = {
    'EXP_KIND': {
        'value': 'GNO'
    },
    'LATENT_DIMS': {
        'values': [16, 32, 64]
    },
    'GRAPH_DATA_LOADER': {
        'value': True
    },
    'NEIGHBORS': {
        'value': 'radial'
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
    'OUTPUT_MODE': {
        'values': ['none', 'add', 'concat']
    }
}

def create_sweep():
    sweep_config = {
        'method': 'random'
    }
    parameters = copy(BASE_PARAMETERS)
    parameters.update(GNO_PARAMETERS)
    sweep_config['parameters'] = parameters
    sweep_id = wandb.sweep(sweep_config, project="PDE-Operators-Baselines")
    print(f"Running Sweep {sweep_id}")

if __name__ == '__main__':
    create_sweep()