import os
import gc
from datetime import datetime
import numpy as np
import wandb

import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint, LearningRateMonitor, RichProgressBar, TQDMProgressBar, ProgressBar, ModelSummary

from utils.config_reader import parse_args, display_config_file, load_config
from data_module.PDEDataModule import PDEDataModule
from model_module.OperatorModelModule import OperatorModelModule
from model_module.ModelModule import ModuleModule
from utils.eval_predictions import run_all_metrics, save_predictions
from constants import ACCELERATOR, LOGS_PATH

import torch
torch.set_float32_matmul_precision('medium')

## WAND CONSTANTS
ENTITY = "kmcguigan"
PROJECT = "PDE-Operators-Baselines"

from constants import WANDB_KEY
wandb.login(key=WANDB_KEY, relogin=True)

def parse_model_outputs(preds:list, idx:int) -> np.array:
    predictions = np.concatenate([
        pred[idx].detach().numpy() for pred in preds
    ])
    return predictions

def evaluate_model(trainer, model, data_module, loader, split, indecies:np.ndarray=None, save_results:bool=False, data_file:str=None):
    outputs = trainer.predict(model=model, dataloaders=loader)
    predictions, actuals = parse_model_outputs(outputs, 0), parse_model_outputs(outputs, 1)
    predictions = data_module.inverse_transform(predictions)
    actuals = data_module.inverse_transform(actuals)
    key_metric = run_all_metrics(predictions, actuals, split)
    if(save_results):
        save_predictions(predictions, actuals, indecies, split, data_file)
    return key_metric

def run_experiment(config=None):
    # setup our wandb run we may choose not to track a run if we want
    # by using wandb offline or something
    with wandb.init(config=config, entity=ENTITY, project=PROJECT, dir=LOGS_PATH):
        print(wandb.run.name)
        # get the configuration
        config = wandb.config
        display_config_file(config)
        # seed the environment
        seed_everything(config['SEED'], workers=True)
        # get the data that we will need to train on
        data_module = PDEDataModule(config)
        train_loader, val_loader = data_module.get_training_data()
        # get the model that we will be fitting
        model = ModelModule(config, data_module.train_example_count, data_module.image_size)
        # get the trainer that we will use to fit the model
        lightning_logger = WandbLogger(log_model=False)
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        early_stopping = EarlyStopping('val/loss', patience=10)
        model_checkpoint_val_loss = ModelCheckpoint(dirpath=os.path.join(LOGS_PATH, 'lightning', wandb.run.name), monitor="val/loss", mode="min", filename="Ep{epoch:02d}-val{val/loss:.2f}-best", auto_insert_metric_name=False, verbose=True)
        trainer = pl.Trainer(
            accelerator=ACCELERATOR,
            logger=lightning_logger,
            max_epochs=config['EPOCHS'],
            deterministic=False,
            callbacks=[early_stopping, model_checkpoint_val_loss, lr_monitor, ModelSummary()],
            log_every_n_steps=15,
        )
        # fit the model on the training data
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        # evaluate on the data
        evaluate_model(trainer, model, data_module, val_loader, 'val')
        del val_loader
        gc.collect()
        evaluate_model(trainer, model, data_module, train_loader, 'train')
        del train_loader
        gc.collect()
        # get the testing data
        test_loader, test_indecies = data_module.get_testing_data()
        key_metric = evaluate_model(trainer, model, data_module, test_loader, 'test', indecies=test_indecies, save_results=True, data_file=config['DATA_FILE'])
        del test_loader
        gc.collect()
        # predict the model
        del data_module
        del model
        del trainer 
        gc.collect()
        return key_metric

def main():
    # get the args
    args = parse_args()
    print("=========PROGRAM ARGS===========")
    print(args)
    print("========================================")
    # load the config object
    config = load_config(args.exp_kind, args.exp_name)
    # add the experiment name to the config file
    config['EXP_NAME'] = args.exp_name
    config['EXP_KIND'] = args.exp_kind
    # get the data file
    # # config['DATA_FILE'] = 'ns_V1e-3_N5000_T50.mat'
    # # config['DATA_FILE'] = 'ns_V1e-4_N10000_T30.mat'
    # config['DATA_FILE'] = 'NavierStokes_V1e-5_N1200_T20.mat'
    # run the experiment
    run_experiment(config=config)

def run_as_sweep(sweep_id:str):
    wandb.agent(f'PDE-Operators-Baselines/{sweep_id}', run_experiment, count=15)

if __name__ == '__main__':
    main()
    # run_as_sweep('c7vp9fvs')


