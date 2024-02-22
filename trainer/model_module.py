import time
import numpy as np
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchmetrics import Metric
from torch.nn import MSELoss

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint, LearningRateMonitor, RichProgressBar, TQDMProgressBar, ProgressBar

from data_handling.data_module import DataModule
from constants import ACCELERATOR

from models.fno import FNO2d
from models.basic_fno import FNO2d as BasicFNO2d
from models.conv_lstm import ConvLSTMModel
from models.afno import AFNO
from models.gno import GNO
from models.persistance import PersistanceModel
from models.vit import VIT

EPSILON = 1e-6

# class GausNorm(nn.Module):
#     def __init__(self):
#         self.means = torch.mean(x, dim=(1,2,3), keepdim=True)
#         self.stds = torch.std(x, dim=(1,2,3), keepdim=True) + EPSILON
#     def forward(self, x, y, inference:bool=False):
#         x = (x - means) / stds
#         if(not inference):
#             y = (y - means) / stds
#         return x, y
#     def inverse(self, x, preds):
#         pass

# class RangeNorm(nn.Module):
#     def __init__(self):
#         self.mins = None 
#         self.maxs = None
#     def forward(self, x, y):
#         x = (x - means) / stds
#         if(not inference):
#             y = (y - means) / stds
#         return x, y, (means, stds)
#     def inverse(self, x, preds):
#         pass

# class PassNorm(nn.Module):
#     def __init__(self):
#         pass
#     def forward(self, x, y):
#         return x, y
#     def inverse(self, x, preds):
#         return x, preds

class BaseModel(nn.Module):
    def __init__(self, config:dict):
        super().__init__()
        # get the normalization kind
        self.instance_normalization = None
        if(config['NORMALIZATION'] is not None and 'inst' in config['NORMALIZATION']):
            self.instance_normalization = config['NORMALIZATION']
        # get the model
        if(config['EXP_KIND'] == 'LATENT_FNO'):
            self.model = FNO2d(config)
        elif(config['EXP_KIND'] == 'FNO'):
            self.model = BasicFNO2d(config)
        elif(config['EXP_KIND'] == 'CONV_LSTM'):
            self.model = ConvLSTMModel(config)
        elif(config['EXP_KIND'] == 'AFNO'):
            self.model = AFNO(config)
        elif(config['EXP_KIND'] == 'GNO'):
            self.model = GNO(config)
        elif(config['EXP_KIND'] == 'PERSISTANCE'):
            self.model = PersistanceModel(config)
        elif(config['EXP_KIND'] == 'VIT'):
            self.model = VIT(config)
        else:
            raise Exception(f"{config['EXP_KIND']} is not implemented please implement this.")

    def gaussian_norm(self, x, y, inference:bool=False):
        means = torch.mean(x, dim=(1,2,3), keepdim=True)
        stds = torch.std(x, dim=(1,2,3), keepdim=True) + EPSILON
        x = (x - means) / stds
        if(not inference):
            y = (y - means) / stds
        return x, y, (means, stds)

    def inv_gaussian_norm(self, x, preds, means, stds):
        x = x * stds + means
        preds = preds * stds + means
        return x, preds

    def range_norm(self, x, y, inference:bool=False):
        mins = torch.minimum(x, dim=(1,2,3), keepdim=True)
        maxs = torch.maximum(x, dim=(1,2,3), keepdim=True)
        x = (x - mins) / (maxs - mins)
        if(not inference):
            y = (y - mins) / (maxs - mins)
        return x, y, (mins, maxs)

    def inv_range_norm(self, x, preds, mins, maxs):
        x = x * (maxs - mins) + mins
        preds = preds * (maxs - mins) + mins
        return x, preds

    def forward(self, batch, inference:bool=False):
        x, y, grid = batch
        B, H, W, C = x.shape
        if(self.instance_normalization == 'gaus_inst'):
            means = torch.mean(x, dim=(1,2,3), keepdim=True)
            stds = torch.std(x, dim=(1,2,3), keepdim=True) + EPSILON
            x = (x - means) / stds
            if(not inference):
                y = (y - means) / stds
        elif(self.instance_normalization == 'range_inst'):
            mins = torch.minimum(x, dim=(1,2,3), keepdim=True)
            maxs = torch.maximum(x, dim=(1,2,3), keepdim=True)
            x = (x - mins) / (maxs - mins)
            if(not inference):
                y = (y - mins) / (maxs - mins)

        preds = self.model(x, grid)

        if(inference):
            if(self.instance_normalization == 'gaus_inst'):
                x = x * stds + means
                preds = preds * stds + means
            elif(self.instance_normalization == 'range_inst'):
                x = x * (maxs - mins) + mins
                preds = preds * (maxs - mins) + mins
            
            return x, y, preds
        return y, preds
# class BaseModel(nn.Module):
#     def __init__(self, config:dict):
#         super().__init__()
#         # get the normalization kind
#         if('inst' in config['NORMALIZATION']):
#             self.instance_normalization = config['NORMALIZATION']
#         # get the model
#         if(config['EXP_KIND'] == 'LATENT_FNO'):
#             self.model = FNO2d(config)
#         elif(config['EXP_KIND'] == 'FNO'):
#             self.model = BasicFNO2d(config)
#         elif(config['EXP_KIND'] == 'CONV_LSTM'):
#             self.model = ConvLSTMModel(config)
#         elif(config['EXP_KIND'] == 'AFNO'):
#             self.model = AFNO(config)
#         elif(config['EXP_KIND'] == 'GNO'):
#             self.model = GNO(config)
#         elif(config['EXP_KIND'] == 'PERSISTANCE'):
#             self.model = PersistanceModel(config)
#         elif(config['EXP_KIND'] == 'VIT'):
#             self.model = VIT(config)
#         else:
#             raise Exception(f"{config['EXP_KIND']} is not implemented please implement this.")

#     def forward(self, batch, inference:bool=False):
#         x, y, grid = batch
#         preds = self.model(x, grid)
#         if(inference):
#             return x, y, preds
#         return y, preds

class CustomMAE(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        assert pred.shape == target.shape
        self.error += torch.sum(torch.abs(torch.subtract(pred, target)))
        self.total += target.numel()
    def compute(self):
        return self.error.float() / self.total

class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=False, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


class LightningModule(pl.LightningModule):
    def __init__(self, config:dict):
        super().__init__()
        self.automatic_optimization = True
        self.optimizer_params = config['OPTIMIZER']
        self.scheduler_params = config['SCHEDULER']
        # get the model
        self.model = BaseModel(config)
        # get the loss and metrics
        if(config['LOSS'] == 'L1NORM'):
            self._loss_fn = LpLoss()
        else:
            self._loss_fn = MSELoss()
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
            optimizer = torch.optim.Adam(self.parameters(), lr=self.optimizer_params['LEARNING_RATE'])#, weight_decay=self.optimizer_params['WEIGHT_DECAY'])
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

class TimingCallback(Callback):
    def __init__(self):
        super().__init__()
        self.epoch_start_time = 0
        self.epoch_total = 0
        self.epoch_count = 0

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - self.epoch_start_time
        self.epoch_total += epoch_duration
        self.epoch_count += 1

    def _get_average_time_per_epoch(self):
        return self.epoch_total / self.epoch_count


class ModelModule(object):
    def __init__(self, config:dict):
        self.lightning_module = LightningModule(config)
        self.trainer, self.timer = self._setup_trainer(config)

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

    def fit(self, data_module:DataModule):
        train_loader, val_loader = data_module.get_training_data()
        self.trainer.fit(model=self.lightning_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
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
        preds = self.trainer.predict(self.lightning_module, data_loader)
        # get the forecasts
        forecasts = self.parse_model_outputs(preds, 0)
        actuals = self.parse_model_outputs(preds, 1)
        last_input = self.parse_model_outputs(preds, 2)
        # inverse the model predictions
        forecasts = data_module.transform_predictions(forecasts)
        actuals = data_module.transform_predictions(actuals)
        last_input = data_module.transform_predictions(last_input)
        return forecasts, actuals, last_input


        




