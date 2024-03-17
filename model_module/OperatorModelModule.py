import torch
from torch.nn import MSELoss

from models.operators.basic_fno import FNO2d
from model_module.ModelModule import ModelModule

class OperatorModelModule(ModelModule):
    def __init__(self, config:dict, train_example_count:int, image_size:tuple):
        super().__init__(config, train_example_count, image_size)

    def get_loss(self, config:dict):
        if(config['LOSS'] == 'LPLOSS'):
            return LpLoss()
        elif(config['LOSS'] == 'MSE'):
            return MSELoss()
        elif(config['LOSS'] == 'HSLOSS'):
            raise Exception('HS LOSS not implemented')
        raise Exception(f'Invalid loss {config["LOSS"]}')

    def get_model(self, config:dict, image_size:tuple):
        if(config['EXP_KIND'] == 'FNO'):
            return FNO2d(config)
        raise Exception(f"Invalid model kind specified of {config['EXP_KIND']}")

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
        return loss, pred, yy

"""
Losses for the models to use
"""
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
