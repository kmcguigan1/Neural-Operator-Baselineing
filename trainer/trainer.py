from torchmetrics import Metric
import torch
# torch.autograd.set_detect_anomaly(True)
from torch.nn import MSELoss

import time

from torchsummary import summary
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, RichProgressBar, TQDMProgressBar, ProgressBar, Callback

# import the models that we have
from utils.constants_handler import ConstantsObject

# from models.afno_boundary_conditions import AFNO
from models.fno import FNO2d
from models.basic_fno import FNO2d as BasicFNO2d
from models.conv_lstm import ConvLSTMModel
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

class LightningModel(pl.LightningModule):
    def __init__(self, config:dict, constants_object:ConstantsObject, train_example_count:int, image_shape:tuple) -> None:
        # lightning information
        super().__init__()
        self.automatic_optimization = True
        # save variables
        self._config = config
        self._train_example_count = train_example_count
        self._image_shape = image_shape
        # define loss and metrics
        self._loss_fn = MSELoss()
        self._train_acc = CustomMAE()
        self._val_acc = CustomMAE()
        # define the model
        if(constants_object.EXP_KIND == 'LATENT_FNO'):
            self.model = FNO2d(config, self._image_shape)
        elif(constants_object.EXP_KIND == 'FNO'):
            self.model = BasicFNO2d(config, self._image_shape)
        elif(constants_object.EXP_KIND == 'CONV_LSTM'):
            self.model = ConvLSTMModel(config, self._image_shape)
        else:
            raise Exception(f"{constants_object.EXP_KIND} is not implemented please implement this.")

    def _print_summary(self, sample_shapes):
        summary(self.model, sample_shapes)

    def _get_preds(self, batch):
        x = batch[0]
        grid = batch[2]
        preds = self.model(x, grid)
        return preds, x.shape[0]

    def training_step(self, batch, batch_idx):
        preds, batch_size = self._get_preds(batch)
        y = batch[1]
        loss = self._loss_fn(preds.reshape(batch_size,-1), y.reshape(batch_size,-1))
        self._train_acc(preds.reshape(batch_size,-1), y.reshape(batch_size,-1))
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/mae', self._train_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        preds, batch_size = self._get_preds(batch)
        y = batch[1]
        loss = self._loss_fn(preds.reshape(batch_size,-1), y.reshape(batch_size,-1))
        self._val_acc(preds.reshape(batch_size,-1), y.reshape(batch_size,-1))
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/mae', self._val_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
    def predict_step(self, batch, batch_idx):
        preds, batch_size = self._get_preds(batch)
        plain_actuals = batch[1]
        last_observation = batch[0][..., -1]
        return preds, plain_actuals, last_observation

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
        print("in start")
        self.epoch_start_time = time.time()

    def on_train_start(self):
        print('starting')

    def on_train_epoch_end(self, trainer, pl_module):
        print("In on epoch end")
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - self.epoch_start_time
        self.epoch_total += epoch_duration
        self.epoch_count += 1

    def _get_average_time_per_epoch(self):
        return self.epoch_total / self.epoch_count

def get_lightning_trainer(config: dict, lightning_logger: WandbLogger = None, accelerator: str = 'auto'):
    # define the callbacks we want to use
    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stopping = EarlyStopping('val/loss', patience=4)
    model_checkpoint_val_loss = ModelCheckpoint(monitor="val/loss", mode="min", filename="Ep{epoch:02d}-val{val/loss:.2f}-best", auto_insert_metric_name=False, verbose=True)
    timer_callback = TimingCallback()
    trainer = pl.Trainer(
        accelerator=accelerator,
        logger=lightning_logger,
        max_epochs=config['EPOCHS'],
        deterministic=False,
        callbacks=[early_stopping, model_checkpoint_val_loss, lr_monitor, timer_callback],#, RichProgressBar(leave=True)],
        log_every_n_steps=10
    )
    return trainer, timer_callback

