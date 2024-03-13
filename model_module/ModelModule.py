import time

import torch
from torch.nn import MSELoss
import lightning.pytorch as pl

from models.basic.conv_lstm import ConvLSTMModel

class ModuleModule(pl.LightningModule):
    def __init__(self, config:dict, train_example_count:int, image_size:tuple):
        super().__init__()
        self.automatic_optimization = True
        # save the info we need from config
        self.learning_rate = config['LEARNING_RATE']
        self.optimizer_params = config['OPTIMIZER']
        self.scheduler_params = config['SCHEDULER']
        self.train_example_count = train_example_count
        self.image_size = image_size
        self.iterations = config['EPOCHS'] * (train_example_count//config['BATCH_SIZE'])
        self.gaussian_norm = True if config['NORMALIZATION'] == 'pointwise_gaussian' else False
        # get the loss and metrics
        self.loss_fn = self.get_loss(config)
        # get the model
        self.model = self.get_model(config, image_size)
        # print the model summary
        # summary(self.model, [(*image_size, config['TIME_STEPS_IN']), (*image_size, 2)])

    def run_batch(self, batch):
        x, y, grid = batch
        preds = self.model(x, grid)
        loss = self.loss_fn(preds, y)
        return loss, preds, y
        
    def get_model(self, config:dict, image_size:tuple):
        if(config['EXP_KIND'] == 'CONV_LSTM'):
            return FNO2d(config)
        raise Exception(f"Invalid model kind specified of {config['EXP_KIND']}")
    
    def get_loss(self, config:dict):
        if(config['LOSS'] == 'MSE'):
            return MSELoss()
        raise Exception(f'Invalid loss {config["LOSS"]}')

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.run_batch(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, _ = self.run_batch(batch)
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
    def predict_step(self, batch, batch_idx):
        _, pred, actual = self.run_batch(batch)
        return pred, actual

    def configure_optimizers(self):
        # get the optimizer
        if(self.optimizer_params['KIND'] == 'adam'):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        else:
            raise Exception(f"Invalid optimizer specified of {self.optimizer_params['KIND']}")
        # get the scheduler
        if(self.scheduler_params['KIND'] == 'cosine'):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.iterations)
            return [optimizer], [{"scheduler":scheduler,"interval":"step"}]
        elif(self.scheduler_params['KIND'] == 'reducer'):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.scheduler_params['PATIENCE'], factor=self.scheduler_params['FACTOR'], verbose=True)
            return [optimizer], [{"scheduler":scheduler,"interval":"epoch","monitor":"val/loss"}]
        else:
            raise Exception(f"Invalid scheduler specified of {self.scheduler_params['KIND']}")