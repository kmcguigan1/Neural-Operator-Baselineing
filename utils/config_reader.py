import os
import yaml
import argparse
import numpy as np 

from constants import CONFIG_PATH

def fix_config_info(config: dict):
    # fix the None or bools
    for key, val in config.items():
        if(val == 'None'):
            config[key] = None
        elif(val == 'False'):
            config[key] = False
        elif(val == 'True'):
            config[key] = True
    return config

def display_config_file(config: dict):
    print("CONFIG FILE")
    for k, v in config.items():
        print(f"{k} -> {v}")

def load_config(exp_kind:str, exp_name:str):
    # pull the base config path
    with open(os.path.join(CONFIG_PATH, 'base.yml'), 'r') as config_file:
        config = yaml.safe_load(config_file)
    with open(os.path.join(CONFIG_PATH, exp_kind, 'base.yml'), mode='r') as base_config_file:
        base_config = yaml.safe_load(base_config_file)
    config.update(base_config)
    # check if the experiment config option is valid and we add that
    if(exp_name is not None and exp_name != 'None'):
        with open(os.path.join(CONFIG_PATH, exp_kind, 'experiments', exp_name), mode='r') as experiment_config_file:
            experiment_config = yaml.safe_load(experiment_config_file)
        # update the config
        config.update(experiment_config)
    # fix some of the things in the config
    config = fix_config_info(config)
    return config

def parse_args():
    parser = argparse.ArgumentParser(
        prog='NeuralOperatorGeneralCode',
        description='This experiment runs different versions of the neural operator.'
    )
    parser.add_argument('--exp-kind', default='LATENT_FNO', choices=['LATENT_FNO','FNO','CONV_LSTM','AFNO','PERSISTANCE','GNO','VIT'], help='Kind of expierment to be run.')
    parser.add_argument('--exp-name', default=None, type=str, help='Name of the experiment config to use.')
    parser.add_argument('--run-wandb', action=argparse.BooleanOptionalAction, help='Should wandb be run.')
    parser.add_argument('--data-file', default='burgers_bc_fixed', type=str, help='Shorthand of the datafile to use.', choices=[
        'burgers_bc_fixed','burgers_bc_vary','burgers_vary',
        'diff_bc_fixed','diff_bc_vary','diff_vary'
    ])
    args = parser.parse_args()
    return args