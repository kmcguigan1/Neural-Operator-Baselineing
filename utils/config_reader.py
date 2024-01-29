import os
import yaml
import argparse
import numpy as np 

from datetime import datetime
import wandb

from utils.constants_handler import ConstantsObject

## WAND CONSTANTS
ENTITY = "kmcguigan"
PROJECT = "PDE-Operators-Baselines"

def setup_wandb(args, config, constant_object):
    # login and setup wandb
    if(args.run_wandb):
        wandb.login(key=constant_object.WANDB_KEY)
        lightning_logger = WandbLogger(log_model="all", project=PROJECT, group=config['METHODOLOGY'])
        lightning_logger.log_hyperparams(config)
        run_name = wandb.run.name
    else:
        lightning_logger = None
        run_name = datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    return lightning_logger, run_name

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

def load_config(constants_object:ConstantsObject, experiment_name: str = None):
    # pull the base config path
    with open(constants_object.BASE_CONFIG_PATH, mode='r') as config_file:
        config = yaml.safe_load(config_file)
    # check if the experiment config option is none
    if(experiment_name is None or experiment_name == 'None'):
        # fix some of the things in the config
        config = fix_config_info(config)
        return config
    # pull the experiment config
    experiment_file = os.path.join(constants_object.CONFIG_PATH, experiment_name)
    with open(experiment_file, mode='r') as experiment_config_file:
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
    parser.add_argument('--exp-kind', default='LATENT_FNO', choices=['CUSTOM_AFNO','FOURCASTNET','LATENT_FNO','SIMPLE_AFNO','VIT','CONV_LSTM'], help='Kind of expierment to be run.')
    parser.add_argument('--exp-name', default=None, type=str, help='Name of the experiment config to use.')
    parser.add_argument('--run-wandb', action=argparse.BooleanOptionalAction, help='Should wandb be run.')
    args = parser.parse_args()
    return args

def setup_wandb(args, config: dict, constant_object: ConstantsObject):
    # login and setup wandb
    if(args.run_wandb):
        wandb.login(key=constant_object.WANDB_KEY)
        lightning_logger = WandbLogger(log_model="all", project=PROJECT, group=config['METHODOLOGY'])
        lightning_logger.log_hyperparams(config)
        run_name = wandb.run.name
    else:
        lightning_logger = None
        run_name = datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    return lightning_logger, run_name