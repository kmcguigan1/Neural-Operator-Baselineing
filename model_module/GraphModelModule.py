import time

import torch
from torch.nn import MSELoss

from model_module.ModelModule import ModelModule

class GraphModelModule(ModelModule):
    def __init__(self, config:dict, train_example_count:int, image_size:tuple):
        super().__init__(config, train_example_count, image_size)
        self.edge_drop_rate = config['EDGE_DROPOUT']

    def get_model(self, config:dict, image_size:tuple):
        if(config['EXP_KIND'] == 'LATENT_BNO'):
            return GNOModel(config)
        raise Exception(f"Invalid model kind specified of {config['EXP_KIND']}")
    
    def get_loss(self, config:dict):
        return None

    def run_batch(self, batch):
        xx, yy, grid, edge_index, edge_attr, image_size, batch, ptr = (
            batch.x, batch.y, batch.grid, batch.edge_index, batch.edge_attr, batch.image_size, batch.batch, batch.ptr
        )
        edge_index, edge_mask = dropout_edge(edge_index, self.edge_drop_rate)
        edge_mask = torch.where(edge_mask)[0]
        edge_attr = edge_attr[edge_mask, ...]
        # run the model
        loss = 0
        for t in range(yy.shape[-1]):
            # get the prediction at this stage
            im = self.model(xx, grid, edge_index, edge_attr, batch, ptr)
            # append the predictions
            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)
            # update the current observed values
            xx = torch.cat((xx[..., 1:], im), dim=-1)
            # calculate the loss function
            y = yy[..., t:t + 1]
            loss += torch.norm(im.view(-1) - y.view(-1),1)
        return loss, pred, yy, image_size