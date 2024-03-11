import numpy as np
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import MSELoss

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint, LearningRateMonitor, RichProgressBar, TQDMProgressBar, ProgressBar

from data_module.data_module import DataModule
from constants import ACCELERATOR

from trainer.losses_and_metrics import CustomMAE, LpLoss, TimingCallback
from trainer.losses_and_metrics import GausInstNorm, RangeInstNorm, PassInstNorm

from models.fno import FNO2d
from models.basic_fno import FNO2d as BasicFNO2d
from models.conv_lstm import ConvLSTMModel
from models.afno import AFNO
from models.gno import GNO
from models.persistance import PersistanceModel
from models.vit import VIT

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
        if(config.get("GRAPH_DATA_LOADER", False) == True):
            self.lightning_module = GraphLightningModule(config, image_shape)
        else:
            self.lightning_module = LightningModule(config, image_shape)
        self.trainer, self.timer = self._setup_trainer(config)

    def fit(self, data_module:DataModule, config:dict):
        train_loader, val_loader = data_module.get_training_data()
        self._create_lightning_module(config, train_loader.image_size)
        if(config['EXP_KIND'] == 'PERSISTANCE'):
            return
        self.trainer.fit(model=self.lightning_module, train_dataloaders=train_loader.dataloader, val_dataloaders=val_loader.dataloader)
        average_time_per_epoch = self.timer._get_average_time_per_epoch()
        total_epochs = self.timer.epoch_count
        wandb.log({'average_time_per_epoch':average_time_per_epoch, 'total_epochs':total_epochs})
            
    def parse_model_outputs(self, preds:list, idx:int) -> np.array:
        predictions = np.concatenate([
            pred[idx].detach().numpy() for pred in preds
        ])
        return predictions

    def predict(self, data_module:DataModule, split:str):
        data_loader = data_module.get_test_data(split=split)
        preds = self.trainer.predict(self.lightning_module, data_loader.dataloader)
        # get the forecasts
        forecasts = self.parse_model_outputs(preds, 0)
        actuals = self.parse_model_outputs(preds, 1)
        inputs = self.parse_model_outputs(preds, 2)
        # inverse the model predictions
        forecasts = data_module.transform_predictions(forecasts, split=split)
        actuals = data_module.transform_predictions(actuals, split=split)
        inputs = data_module.transform_predictions(inputs, split=split)
        return forecasts, actuals, inputs, data_loader.indecies

class LightningModule(pl.LightningModule):
    def __init__(self, config:dict, image_shape:tuple):
        super().__init__()
        self.automatic_optimization = True
        self.learning_rate = config['LEARNING_RATE']
        self.optimizer_params = config['OPTIMIZER']
        self.scheduler_params = config['SCHEDULER']
        # get the model that we are using
        self.model = self._get_model(config, image_shape)
        # get the normalization layer
        self.norm_layer = self._setup_norm_layer(config)
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

    def _get_model(self, config:dict, image_size:tuple):
        if(config['EXP_KIND'] == 'LATENT_FNO'):
            return FNO2d(config)
        if(config['EXP_KIND'] == 'FNO'):
            return BasicFNO2d(config)
        if(config['EXP_KIND'] == 'CONV_LSTM'):
            return ConvLSTMModel(config, image_shape)
        if(config['EXP_KIND'] == 'AFNO'):
            return AFNO(config, image_shape)
        if(config['EXP_KIND'] == 'PERSISTANCE'):
            return PersistanceModel(config)
        if(config['EXP_KIND'] == 'VIT'):
            return VIT(config)
        if(config['EXP_KIND'] == 'GNO'):
            return GNO(config)
        raise Exception(f"Invalid EXP_KIND of {config['EXP_KIND']}")

    def _setup_norm_layer(self, config:dict, dims:tuple=(1,2,3)):
        if(config['NORMALIZATION'] == 'pointwise_gaussian'):
            return GausInstNorm(dims=dims)
        if(config['NORMALIZATION'] == 'pointwise_range'):
            return RangeInstNorm(dims=dims)
        return PassInstNorm()

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
        x, y, preds = self._run_model(batch, inference=True)
        return preds, y, x

    def configure_optimizers(self):
        # get the optimizer
        if(self.optimizer_params['KIND'] == 'adam'):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)#, weight_decay=self.optimizer_params['WEIGHT_DECAY'])
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

    def _run_model(self, batch, inference:bool=False):
        # pull the data
        x, y, grid = batch
        B, H, W, C = x.shape
        # normalize the data
        x, info = self.norm_layer.forward(x)
        # run the model with the other info
        preds = self.model(x, grid)
        # undo the transform
        preds = self.norm_layer.inverse(preds, info)
        # if we are not inferencing just return
        if(not inference):
            return y, preds
        # undo the x transform and return that info as well
        x = self.norm_layer.inverse(x, info)
        return x, y, preds 

class GraphLightningModule(LightningModule):
    def __init__(self, config:dict, image_shape:tuple):
        super().__init__(config, image_shape)
    def _run_model(self, batch, inference:bool=False):
        # pull the data
        x, y, grid, edge_index, edge_attrs = batch
        B, H, W, C = x.shape
        # normalize the data
        x, info = self.norm_layer.forward(x)
        # save the predictions
        preds = torch.zeros((B, H, W, y.shape[-1]), dtype=x.dtype, device=x.device)
        # iterate over the batch
        for batch_idx in range(x.shape[0]):
            preds[batch_idx, ...] = self.model(x[batch_idx, ...], grid[batch_idx, ...], edge_index[batch_idx, ...], edge_features[batch_idx, ...])
        # undo the transform
        preds = self.norm_layer.inverse(preds, info)
        # undo the transform
        preds = self.norm_layer.inverse(preds, info)
        # if we are not inferencing just return
        if(not inference):
            return y, preds
        # undo the x transform and return that info as well
        x = self.norm_layer.inverse(x, info)
        return x, y, preds

# class OperatorLightningModule(LightningModule):
#     def __init__(self, config:dict, image_shape:tuple):
#         super().__init__(config, image_shape)
#     def _run_model(self, batch, inference:bool=False):
#         # pull the data
#         x, y, grid = batch
#         B, H, W, C = x.shape
#         # normalize the data
#         x, info = self.norm_layer.forward(x)
#         # iterate over the output steps to get predictions form the operator
#         time_steps_out = y.shape[-1]
#         xx = x
#         preds = torch.zeros((B, H, W, time_steps_out), dtype=x.dtype, device=x.device)
#         loss = 0
#         for idx in range(time_steps_out):
#             # add the new predictions to the x data
#             if(idx > 0):
#                 xx = torch.cat((xx[..., 1:], predictions[..., idx-1:idx]), dim=-1)
#             # get the prediction
#             pred = self.model(xx, grid)
#             # update the loss function
#             loss += self._loss_fn(pred.reshape(B, -1), y[..., idx].reshape(B, -1))
#             # save the prediction
#             preds[..., idx] = pred
#         # undo the transform
#         preds = self.norm_layer.inverse(preds, info)
#         # undo the transform
#         preds = self.norm_layer.inverse(preds, info)
#         # if we are not inferencing just return
#         if(not inference):
#             return y, preds
#         # undo the x transform and return that info as well
#         x = self.norm_layer.inverse(x, info)
#         return x, y, preds

# class GraphOperatorLightningModule(LightningModule):
#     def __init__(self, config:dict, image_shape:tuple):
#         super().__init__(config, image_shape)
#     def _run_model(self, batch, inference:bool=False):
#         # pull the data
#         x, y, grid, edge_index, edge_attrs = batch
#         B, H, W, C = x.shape
#         # normalize the data
#         x, info = self.norm_layer.forward(x)
#         # save the predictions
#         time_steps_out = y.shape[-1]
#         # iterate over the
#         # iterate over the output steps to get predictions form the operator
#         xx = x
#         preds = torch.zeros((B, H, W, time_steps_out), dtype=x.dtype, device=x.device)
#         for idx in range(time_steps_out):
#             # add the new predictions to the x data
#             if(idx > 0):
#                 xx = torch.cat((xx[..., 1:], predictions[..., idx-1:idx]), dim=-1)
#             # get the prediction
#             pred = self.model(xx, grid)
#             # save the prediction
#             preds[..., idx] = pred
#         # undo the transform
#         preds = self.norm_layer.inverse(preds, info)
#         # undo the transform
#         preds = self.norm_layer.inverse(preds, info)
#         # if we are not inferencing just return
#         if(not inference):
#             return y, preds
#         # undo the x transform and return that info as well
#         x = self.norm_layer.inverse(x, info)
#         return x, y, preds