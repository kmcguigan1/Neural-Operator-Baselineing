BASE_PARAMETERS = {
    'SEED': {'value': 1999},
    'DATA_FILE': {'value': 'diffusion_varying_sinusoidal_init_fixed_diffusivity_non_periodic_boundaries.npy'},
    'TIME_STEPS_IN': {'value': 7},
    'TIME_STEPS_OUT': {'value': 28},
    'BATCH_SIZE': {'value': 16},
    'TIME_INT': {'value': 1},
    'OPTIMIZER': {'values': [{'KIND':'adam', 'LEARNING_RATE':0.01}, {'KIND':'adam', 'LEARNING_RATE':0.001}]},
    'SCHEDULER': {'value': {'KIND':'reducer', 'FACTOR':0.1, 'PATIENCE':2}},
    'EPOCHS': {'value': 100},
    'LOSS': {'values': ['MSE', 'L1NORM']}
}