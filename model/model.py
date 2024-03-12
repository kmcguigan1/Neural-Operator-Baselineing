from abc import ABC, abstractmethod
import time

import torch
from torch.nn import MSELoss
import lightning.pytorch as pl

from torchsummary import summary

from model.operators.basic_fno import FNO2d as BasicFNO2d

EPSILON = 1e-5

class Model(pl.LightningModule):
    def __init__(self, config:dict, train_example_count:int, image_size:tuple):
        super().__init__()
        self.automatic_optimization = True
        # save the info we need from config
        self.learning_rate = config['LEARNING_RATE']
        self.optimizer_params = config['OPTIMIZER']
        self.scheduler_params = config['SCHEDULER']
        self.iterations = config['EPOCHS'] * (train_example_count//config['BATCH_SIZE'])
        # get the loss and metrics
        self.loss_fn = self.get_loss(config)
        # get the model
        if(config['EXP_KIND'] == 'FNO'):
            self.model = BasicFNO2d(config)
        else:
            raise Exception(f"Invalid model kind specified of {config['EXP_KIND']}")
        # print the model summary
        # summary(self.model, [(*image_size, config['TIME_STEPS_IN']), (*image_size, 2)])

    def run_batch(self, batch):
        xx, yy, grid = batch
        batch_size = xx.shape[0]
        # run the model
        loss = 0
        for t in range(yy.shape[-1]):
            # get the prediction at this stage
            im = self.model(xx, grid)
            # append the predictions
            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)
            # update the current observed values
            xx = torch.cat((xx[..., 1:], im), dim=-1)
            # calculate the loss function
            y = yy[..., t:t + 1]
            loss += self.loss_fn(im.reshape(batch_size, -1), y.reshape(batch_size, -1))
        return loss, pred
    
    def get_loss(self, config:dict):
        if(config['LOSS'] == 'LPLOSS'):
            return LpLoss()
        elif(config['LOSS'] == 'MSE'):
            return MSELoss()
        elif(config['LOSS'] == 'HSLOSS'):
            raise Exception('HS LOSS not implemented')
        raise Exception(f'Invalid loss {config["LOSS"]}')

    def training_step(self, batch, batch_idx):
        loss, _ = self.run_batch(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _ = self.run_batch(batch)
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
    def predict_step(self, batch, batch_idx):
        _, pred = self.run_batch(batch)
        y = batch[1]
        return pred, y

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
