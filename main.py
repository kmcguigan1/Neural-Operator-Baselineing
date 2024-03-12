import gc
from datetime import datetime
import wandb

import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint, LearningRateMonitor, RichProgressBar, TQDMProgressBar, ProgressBar, ModelSummary

from utils.config_reader import parse_args, display_config_file, load_config
from dataset.data import get_data_loaders
from model.model import Model
from constants import ACCELERATOR

import torch
torch.set_float32_matmul_precision('medium')

## WAND CONSTANTS
ENTITY = "kmcguigan"
PROJECT = "PDE-Operators-Baselines"

from constants import WANDB_KEY
wandb.login(key=WANDB_KEY, relogin=True)

def run_experiment(config=None):
    # setup our wandb run we may choose not to track a run if we want
    # by using wandb offline or something
    with wandb.init(config=config, entity=ENTITY, project=PROJECT):
        # get the configuration
        config = wandb.config
        display_config_file(config)
        # seed the environment
        seed_everything(config['SEED'], workers=True)
        # get the data that we will need to train on
        train_loader, val_loader, test_loader, transform, train_example_count, train_image_size = get_data_loaders(config)
        # get the model that we will be fitting
        model = Model(config, train_example_count, train_image_size)
        # get the trainer that we will use to fit the model
        lightning_logger = WandbLogger(log_model=False)
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        early_stopping = EarlyStopping('val/loss', patience=4)
        model_checkpoint_val_loss = ModelCheckpoint(monitor="val/loss", mode="min", filename="Ep{epoch:02d}-val{val/loss:.2f}-best", auto_insert_metric_name=False, verbose=True)
        trainer = pl.Trainer(
            accelerator=ACCELERATOR,
            logger=lightning_logger,
            max_epochs=config['EPOCHS'],
            deterministic=False,
            callbacks=[early_stopping, model_checkpoint_val_loss, lr_monitor, ModelSummary()],
            log_every_n_steps=1
        )
        # fit the model on the training data
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        trainer.predict(model=model, dataloaders=test_loader)
        # predict the model
        gc.collect()
        return

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
    config['DATA_FILE'] = 'ns_V1e-3_N5000_T50.mat'
    # run the experiment
    run_experiment(config=config)

def run_as_sweep(sweep_id:str):
    wandb.agent(f'PDE-Operators-Baselines/{sweep_id}', run_experiment, count=15)

if __name__ == '__main__':
    main()
    # run_as_sweep('c7vp9fvs')


