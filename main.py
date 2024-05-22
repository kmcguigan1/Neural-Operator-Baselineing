import os
os.environ["WANDB__SERVICE_WAIT"] = "300"
import gc
from datetime import datetime
import numpy as np
import wandb

# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint, LearningRateMonitor, RichProgressBar, TQDMProgressBar, ProgressBar, ModelSummary

from utils.config_reader import parse_args, display_config_file, load_config
# data modules
from data_module.PDEDataModule import PDEDataModule
from data_module.GraphPDEDataModule import GraphPDEDataModule
from data_module.MPGraphPDEDataModule import MPGraphPDEDataModule, MPGraphPDEDataModuleCust
from data_module.BoundaryGraphPDEDataModule import BoundaryGraphPDEDataModule
# model modules
from model_module.GraphModelModule import GraphModelModule
from model_module.OperatorModelModule import OperatorModelModule
from model_module.GraphOperatorModelModule import GraphOperatorModelModule
from model_module.MPGraphOperatorModelModule import MPGraphOperatorModelModule, MPGraphOperatorModelModuleCust
from model_module.BoundaryGraphOperatorModelModule import BoundaryGraphOperatorModelModule
from model_module.ModelModule import ModelModule
# others
from utils.eval_predictions import run_all_metrics, save_predictions
from constants import ACCELERATOR

## WAND CONSTANTS
ENTITY = "kmcguigan"
# PROJECT = "NIPS-24-GFNO"
PROJECT = "DEBUG"

from constants import WANDB_KEY
wandb.login(key=WANDB_KEY)

def parse_model_outputs(preds:list, idx:int) -> np.array:
    predictions = np.concatenate([
        pred[idx].detach().numpy() for pred in preds
    ])
    return predictions

def evaluate_model(trainer, model, data_module, loader, split, exp_kind, indecies:np.ndarray=None, save_results:bool=False, data_file:str=None):
    outputs = trainer.predict(model=model, dataloaders=loader, ckpt_path="best")
    predictions, actuals = parse_model_outputs(outputs, 0), parse_model_outputs(outputs, 1)
    del outputs
    gc.collect()
    predictions = data_module.inverse_transform(predictions)
    actuals = data_module.inverse_transform(actuals)
    key_metric = run_all_metrics(predictions, actuals, split)
    if(save_results):
        save_predictions(predictions, actuals, indecies, split, data_file, exp_kind)
    del predictions
    del actuals
    gc.collect()
    return key_metric

def run_experiment(config=None):
    # setup our wandb run we may choose not to track a run if we want
    # by using wandb offline or something
    with wandb.init(config=config, entity=ENTITY, project=PROJECT):
        print(wandb.run.name)
        # get the configuration
        config = wandb.config
        display_config_file(config)
        # seed the environment
        seed_everything(config['SEED'], workers=True)
        # get the data that we will need to train on
        if(config['EXP_KIND'] in ['GNO','GCN','GFNO','GINO','LATENT_GFNO','GFNO_EFF','GNO_EFF']):
            data_module = GraphPDEDataModule(config)
        elif(config['EXP_KIND'] in ['MPGNO',]):
            data_module = MPGraphPDEDataModuleCust(config)
        elif(config['EXP_KIND'] in ['BENO',]):
            data_module = BoundaryGraphPDEDataModule(config)
        elif(config['EXP_KIND'] in ['CONV_LSTM','FNO','CONV_FNO','FCN_LSTM']):     
            data_module = PDEDataModule(config)
        else:
            raise Exception('No data module can be found')
        train_loader, val_loader = data_module.get_training_data()
        # get the model that we will be fitting
        if(config['EXP_KIND'] in ['GNO','GKN','GCN','GFNO',"GINO",'GFNO_EFF','GNO_EFF']):
            model = GraphOperatorModelModule(config, data_module.train_example_count, data_module.image_size)
        elif(config['EXP_KIND'] in ['LATENT_GFNO',]):
            model = GraphModelModule(config, data_module.train_example_count, data_module.image_size)
        elif(config['EXP_KIND'] in ['MPGNO',]):
            model = MPGraphOperatorModelModuleCust(config, data_module.train_example_count, data_module.image_size)
        elif(config['EXP_KIND'] in ['BENO',]):
            model = BoundaryGraphOperatorModelModule(config, data_module.train_example_count, data_module.image_size)
        elif(config['EXP_KIND'] in ['FNO','CONV_FNO']):
            model = OperatorModelModule(config, data_module.train_example_count, data_module.image_size)
        elif(config['EXP_KIND'] in ['CONV_LSTM','FCN_LSTM']):
            model = ModelModule(config, data_module.train_example_count, data_module.image_size)
        else:
            raise Exception('No data module can be found')
        # get the trainer that we will use to fit the model
        lightning_logger = WandbLogger(log_model=False)
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        early_stopping = EarlyStopping('val/loss', patience=10)
        model_checkpoint_val_loss = ModelCheckpoint(monitor="val/loss", mode="min", filename="Ep{epoch:02d}-val{val/loss:.2f}-best", auto_insert_metric_name=False, verbose=True)
        trainer = pl.Trainer(
            accelerator=ACCELERATOR,
            logger=lightning_logger,
            max_epochs=config['EPOCHS'],
            deterministic=False,
            callbacks=[early_stopping, model_checkpoint_val_loss, lr_monitor],
            log_every_n_steps=10,
        )
        # fit the model on the training data
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        # evaluate on the data
        # evaluate_model(trainer, model, data_module, val_loader, 'val')
        del val_loader
        gc.collect()
        # evaluate_model(trainer, model, data_module, train_loader, 'train')
        del train_loader
        gc.collect()
        # get the testing data
        save_results = config.get("SAVE_RESULTS", False)
        if(config['EXP_KIND'] != 'CONV_LSTM'):
            print("Running Test on 1 downsample ratio")
            test_loader = data_module.get_testing_data(downsample_ratio=1)
            key_metric = evaluate_model(trainer, model, data_module, test_loader, 'test_upsampled', config['EXP_KIND'], indecies=data_module.data_reader.test_indecies, data_file=config['DATA_FILE'], save_results=save_results)
            del test_loader
            gc.collect()
        print("Running Test on 2 downsample ratio")
        test_loader = data_module.get_testing_data(downsample_ratio=2)
        key_metric = evaluate_model(trainer, model, data_module, test_loader, 'test', config['EXP_KIND'], indecies=data_module.data_reader.test_indecies, data_file=config['DATA_FILE'], save_results=save_results)
        del test_loader
        gc.collect()
        # predict the model
        del data_module
        del model
        del trainer 
        gc.collect()
        return key_metric
    
# def load_and_eval_model(config=None):
#     with wandb.init(config=config, entity=ENTITY, project=PROJECT):
#         print(wandb.run.name)
#         seed_everything(config['SEED'], workers=True)
#         data_module = GraphPDEDataModule(config)
#         train_loader, val_loader = data_module.get_training_data()
#         del train_loader
#         del val_loader
#         gc.collect()
#         model = GraphOperatorModelModule.load_from_checkpoint(r'C:\Users\Kiernan\Documents\GitHub\Neural-Operator-Baselineing\lightning_logs\umpeej83\checkpoints\Ep44-val15.98-best.ckpt', config=config, train_example_count=data_module.train_example_count, image_size=data_module.image_size)
#         lightning_logger = WandbLogger(log_model=False)
#         lr_monitor = LearningRateMonitor(logging_interval='epoch')
#         early_stopping = EarlyStopping('val/loss', patience=6)
#         model_checkpoint_val_loss = ModelCheckpoint(monitor="val/loss", mode="min", filename="Ep{epoch:02d}-val{val/loss:.2f}-best", auto_insert_metric_name=False, verbose=True)
#         trainer = pl.Trainer(
#             accelerator=ACCELERATOR,
#             logger=lightning_logger,
#             max_epochs=config['EPOCHS'],
#             deterministic=False,
#             callbacks=[early_stopping, model_checkpoint_val_loss, lr_monitor],
#             log_every_n_steps=1,
#         )
#         print("Running Test on 1 downsample ratio")
#         test_loader = data_module.get_testing_data(downsample_ratio=1)
#         key_metric = evaluate_model(trainer, model, data_module, test_loader, 'test_upsampled', indecies=data_module.data_reader.test_indecies, data_file=config['DATA_FILE'])
#         del test_loader
#         gc.collect()
#         print("Running Test on 2 downsample ratio")
#         test_loader = data_module.get_testing_data(downsample_ratio=2)
#         key_metric = evaluate_model(trainer, model, data_module, test_loader, 'test', indecies=data_module.data_reader.test_indecies, data_file=config['DATA_FILE'])
#         del test_loader
#         gc.collect()
#         # predict the model
#         del data_module
#         del model
#         del trainer 
#         gc.collect()
#         return key_metric



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
    # config['DATA_FILE'] = 'heat_equation_non_periodic.h5'
    # config['DATA_FILE'] = 'ns_V1e-3_N5000_T50.mat'
    # # config['DATA_FILE'] = 'ns_V1e-4_N10000_T30.mat'
    # config['DATA_FILE'] = 'NavierStokes_V1e-5_N1200_T20.mat'
    # run the experiment
    run_experiment(config=config)

def run_as_sweep(sweep_id:str):
    wandb.agent(f'PDE-Operators-Baselines/{sweep_id}', run_experiment, count=15)

if __name__ == '__main__':
    main()
    # run_as_sweep('c7vp9fvs')


