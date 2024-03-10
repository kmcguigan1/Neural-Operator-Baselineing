import numpy as np
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import MSELoss

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint, LearningRateMonitor, RichProgressBar, TQDMProgressBar, ProgressBar


from data_handling.data_module import DataModule
from constants import ACCELERATOR

class LightningModule(pl.LightningModule):
    def __init__(self, config:dict, image_shape:tuple):
        super().__init__()
        self.automatic_optimization = True
        self.leraning_rate = config['LEARNING_RATE']
        self.optimizer_params = config['OPTIMIZER']
        self.scheduler_params = config['SCHEDULER']
        # get the model
        if(config.get("GRAPH_LOADER", False) == True):
            self.model = GraphBaseModel(config, image_shape)
        else:
            self.model = BaseModel(config, image_shape)
        # get the loss and metrics
        if(config['LOSS'] == 'LPLOSS'):
            self._loss_fn = LpLoss()
        elif(config['LOSS'] == 'MSE'):
            self._loss_fn = MSELoss()
        elif(config['LOSS'] == 'HSLOSS'):
            raise Exception('HS LOSS not implemented')
        else:
            raise Exception(f'Invalid loss {config["LOSS"]}')
        self._train_acc = CustomMAE()
        self._val_acc = CustomMAE()

    def _run_model(self, batch, inference:bool = False):
        if(inference):
            x, y, preds = self.model(batch, inference=True)
            return x[..., -1], y, preds
        else:
            y, preds = self.model(batch)
            return preds, y

    def training_step(self, batch, batch_idx):
        preds, y = self._run_model(batch)
        loss = self._loss_fn(preds, y)
        self._train_acc(preds, y)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/mae', self._train_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        preds, y = self._run_model(batch)
        loss = self._loss_fn(preds, y)
        self._val_acc(preds, y)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/mae', self._val_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
    def predict_step(self, batch, batch_idx):
        last_observation, y, preds = self._run_model(batch, inference=True)
        return preds, y, last_observation

    def configure_optimizers(self):
        # get the optimizer
        if(self.optimizer_params['KIND'] == 'adam'):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.leraning_rate)#, weight_decay=self.optimizer_params['WEIGHT_DECAY'])
        else:
            raise Exception(f"Invalid optimizer specified of {self.optimizer_params['KIND']}")
        # get the 
        if(self.scheduler_params['KIND'] == 'cosine'):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self._train_example_count*self._config['EPOCHS'], eta_min=self.scheduler_params['MIN_LR'])
            return [optimizer], [{"scheduler":scheduler,"interval":"step"}]
        elif(self.scheduler_params['KIND'] == 'reducer'):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.scheduler_params['PATIENCE'], factor=self.scheduler_params['FACTOR'], verbose=True)
            return [optimizer], [{"scheduler":scheduler,"interval":"epoch","monitor":"val/loss"}]
        else:
            raise Exception(f"Invalid scheduler specified of {self.scheduler_params['KIND']}")

class ModelModule(object):
    def __init__(self, config:dict):
        self.lightning_module = None
        self.trainer = None
        self.time = None

    def _setup_trainer(self, config:dict):
        lightning_logger = WandbLogger(log_model=False)
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        early_stopping = EarlyStopping('val/loss', patience=4)
        model_checkpoint_val_loss = ModelCheckpoint(monitor="val/loss", mode="min", filename="Ep{epoch:02d}-val{val/loss:.2f}-best", auto_insert_metric_name=False, verbose=True)
        timer_callback = TimingCallback()
        trainer = pl.Trainer(
            accelerator=ACCELERATOR,
            logger=lightning_logger,
            max_epochs=config['EPOCHS'],
            deterministic=False,
            callbacks=[early_stopping, model_checkpoint_val_loss, lr_monitor, timer_callback],
            log_every_n_steps=10
        )
        return trainer, timer_callback

    def _create_lightning_module(self, config:dict, image_shape:tuple):
        assert self.lightning_module is None, "attempting to reset lightning module in model module."
        self.lightning_module = LightningModule(config, image_shape)
        self.trainer, self.timer = self._setup_trainer(config)

    def fit(self, data_module:DataModule, config:dict):
        train_loader, val_loader, train_image_shape = data_module.get_training_data()
        self._create_lightning_module(config, train_image_shape)
        if(config['EXP_KIND'] == 'PERSISTANCE'):
            return
        self.trainer.fit(model=self.lightning_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
        average_time_per_epoch = self.timer._get_average_time_per_epoch()
        total_epochs = self.timer.epoch_count
        wandb.log({'average_time_per_epoch':average_time_per_epoch, 'total_epochs':total_epochs})
            
    def parse_model_outputs(self, preds:list, idx:int) -> np.array:
        predictions = np.concatenate([
            pred[idx].detach().numpy() for pred in preds
        ])
        return predictions

    def predict(self, data_module:DataModule, split:str, return_metadata:bool=False):
        if(return_metadata):
            data_loader, indecies, metadata = data_module.get_test_data(split=split, return_metadata=True)
        else:
            data_loader = data_module.get_test_data(split=split)
        preds = self.trainer.predict(self.lightning_module, data_loader)
        # get the forecasts
        forecasts = self.parse_model_outputs(preds, 0)
        actuals = self.parse_model_outputs(preds, 1)
        last_input = self.parse_model_outputs(preds, 2)
        # inverse the model predictions
        forecasts = data_module.transform_predictions(forecasts, split=split)
        actuals = data_module.transform_predictions(actuals, split=split)
        last_input = data_module.transform_predictions(last_input, split=split, no_time_dim=True)
        if(return_metadata):
            return forecasts, actuals, last_input, metadata, indecies
        return forecasts, actuals, last_input


        




