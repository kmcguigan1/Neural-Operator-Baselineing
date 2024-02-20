import time
import numpy as np
from prettytable import PrettyTable

import torch
# torch.autograd.set_detect_anomaly(True)
from torch.nn import MSELoss

from einops import rearrange

from torchmetrics import Metric
from torchsummary import summary

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint, LearningRateMonitor, RichProgressBar, TQDMProgressBar, ProgressBar

# from models.afno_boundary_conditions import AFNO
from models.fno import FNO2d
from models.basic_fno import FNO2d as BasicFNO2d
from models.conv_lstm import ConvLSTMModel
from models.afno import AFNO
from models.gno import GNO
from models.persistance import PersistanceModel
from models.vit import VIT
# from models.afno_simple import SimpleAFNO
# from models.vit import VIT
# from models.custom_afno import CustomAFNO
# from models.custom_afno_compl import CustomAFNO as CustomComplAFNO
# from models.conv_lstm import ConvLSTMModel

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

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

class LightningModel(pl.LightningModule):
    def __init__(self, config:dict, image_shape:tuple) -> None:
        # lightning information
        super().__init__()
        self.automatic_optimization = True
        # save variables
        self._config = config
        self._image_shape = image_shape
        # determine things about our loader
        self.graph_style_loader = False
        if("GRAPH_DATA_LOADER" in config.keys() and config["GRAPH_DATA_LOADER"] == True):
            self.graph_style_loader = True
        # define loss and metrics
        if('LOSS' in self._config.keys() and self._config['LOSS'] == 'L1NORM'):
            self._loss_fn = LpLoss()
        else:
            self._loss_fn = MSELoss()
        self._train_acc = CustomMAE()
        self._val_acc = CustomMAE()
        # define the model
        if(config['EXP_KIND'] == 'LATENT_FNO'):
            self.model = FNO2d(config, self._image_shape)
        elif(config['EXP_KIND'] == 'FNO'):
            self.model = BasicFNO2d(config, self._image_shape)
        elif(config['EXP_KIND'] == 'CONV_LSTM'):
            self.model = ConvLSTMModel(config, self._image_shape)
        elif(config['EXP_KIND'] == 'AFNO'):
            self.model = AFNO(config, self._image_shape)
        elif(config['EXP_KIND'] == 'GNO'):
            self.model = GNO(config, self._image_shape)
        elif(config['EXP_KIND'] == 'PERSISTANCE'):
            self.model = PersistanceModel(config, self._image_shape)
        elif(config['EXP_KIND'] == 'VIT'):
            self.model = VIT(config, self._image_shape)
        else:
            raise Exception(f"{config['EXP_KIND']} is not implemented please implement this.")

    def _print_summary(self):
        count_parameters(self.model)

    def _process_graph_batch(self, batch):
        x, y, grid, edge_index, edge_features = batch[:5]
        preds = torch.zeros_like(y, device=x.device)
        for batch_idx in range(x.shape[0]):
            preds[batch_idx, ...] = self.model(x[batch_idx, ...], grid[batch_idx, ...], edge_index[batch_idx, ...], edge_features[batch_idx, ...])
        return x, y, preds

    def _get_preds(self, batch):
        if(self.graph_style_loader):
            x, y, preds = self._process_graph_batch(batch)
        else: 
            x, y, grid = batch[:3]
            preds = self.model(x, grid)
        batch_size = x.shape[0]
        return x, y, preds, batch_size

    def training_step(self, batch, batch_idx):
        x, y, preds, batch_size = self._get_preds(batch)
        loss = self._loss_fn(preds.reshape(batch_size,-1), y.reshape(batch_size,-1))
        self._train_acc(preds.reshape(batch_size,-1), y.reshape(batch_size,-1))
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/mae', self._train_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, preds, batch_size = self._get_preds(batch)
        loss = self._loss_fn(preds.reshape(batch_size,-1), y.reshape(batch_size,-1))
        self._val_acc(preds.reshape(batch_size,-1), y.reshape(batch_size,-1))
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/mae', self._val_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
    def predict_step(self, batch, batch_idx):
        x, y, preds, batch_size = self._get_preds(batch)
        # if we have a graph style loader reshape the images
        if(self.graph_style_loader):
            x = rearrange(x, 'b (h w) f -> b h w f', b=batch_size, h=self._image_shape[0], w=self._image_shape[1], f=self._config['TIME_STEPS_IN'])
            y = rearrange(y, 'b (h w) f -> b h w f', b=batch_size, h=self._image_shape[0], w=self._image_shape[1], f=self._config['TIME_STEPS_OUT'])
            preds = rearrange(preds, 'b (h w) f -> b h w f', b=batch_size, h=self._image_shape[0], w=self._image_shape[1], f=self._config['TIME_STEPS_OUT'])
        last_observation = x[..., -1]
        return preds, y, last_observation

    def configure_optimizers(self):
        # get the optimizer
        if(self._config['OPTIMIZER']['KIND'] == 'adam'):
            optimizer = torch.optim.Adam(self.parameters(), lr=self._config['OPTIMIZER']['LEARNING_RATE'])#, weight_decay=self._config['OPTIMIZER']['WEIGHT_DECAY'])
        else:
            raise Exception(f"Invalid optimizer specified of {self._config['OPTIMIZER']['KIND']}")
        # get the 
        if(self._config['SCHEDULER']['KIND'] == 'cosine'):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self._train_example_count*self._config['EPOCHS'], eta_min=self._config['SCHEDULER']['MIN_LR'])
            return [optimizer], [{"scheduler":scheduler,"interval":"step"}]
        elif(self._config['SCHEDULER']['KIND'] == 'reducer'):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self._config['SCHEDULER']['PATIENCE'], factor=self._config['SCHEDULER']['FACTOR'], verbose=True)
            return [optimizer], [{"scheduler":scheduler,"interval":"epoch","monitor":"val/loss"}]
        else:
            raise Exception(f"Invalid scheduler specified of {self._config['SCHEDULER']['KIND']}")

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
    
class MyEarlyStopping(EarlyStopping):
    def on_validation_end(self, trainer, pl_module):
        # override this to disable early stopping at the end of val loop
        pass

def get_lightning_trainer(config: dict, dataset_statistics:dict, img_size:tuple, accelerator: str = 'auto'):
    # define the callbacks we want to use
    lightning_logger = WandbLogger(log_model=False)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    early_stopping = EarlyStopping('val/loss', patience=4)
    variance = dataset_statistics['var']
    if(config['NORMALIZATION'] == 'gaus'):
        variance = 1
    elif(config['NORMALIZATION'] == 'range'):
        variance = 0.75
    threshold = img_size[0] * img_size[1] * variance * 6
    print("Stopping MAE threshold: ", threshold)
    early_stopping_upper_bound = EarlyStopping('val/mae', patience=20, stopping_threshold=threshold)
    model_checkpoint_val_loss = ModelCheckpoint(monitor="val/loss", mode="min", filename="Ep{epoch:02d}-val{val/loss:.2f}-best", auto_insert_metric_name=False, verbose=True)
    timer_callback = TimingCallback()
    trainer = pl.Trainer(
        accelerator=accelerator,
        logger=lightning_logger,
        max_epochs=config['EPOCHS'],
        deterministic=False,
        callbacks=[early_stopping, model_checkpoint_val_loss, lr_monitor, timer_callback, early_stopping_upper_bound],
        log_every_n_steps=10
    )
    return trainer, timer_callback

