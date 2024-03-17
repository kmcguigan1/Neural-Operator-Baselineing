import torch

from models.operators.gcn import GCN_Net as GCNModel
from models.operators.gkn import KernelNN as GKNModel
from models.operators.gno import KernelNN as GNOModel
from models.operators.mgkn import MKGN as MKGNModel
from model_module.OperatorModelModule import OperatorModelModule

class GraphOperatorModelModule(OperatorModelModule):
    def __init__(self, config:dict, train_example_count:int, image_size:tuple):
        super().__init__(config, train_example_count, image_size)

    def get_model(self, config:dict, image_size:tuple):
        if(config['EXP_KIND'] == 'GNO'):
            return GNOModel(config)
        raise Exception(f"Invalid model kind specified of {config['EXP_KIND']}")

    def get_loss(self, config:dict):
        return None

    def run_batch(self, batch):
        xx, yy, grid, edge_index, edge_attr = (
            batch.x, batch.y, batch.grid, batch.edge_index, batch.edge_attr
        )
        # run the model
        loss = 0
        for t in range(yy.shape[-1]):
            # get the prediction at this stage
            im = self.model(xx, grid, edge_index, edge_attr)
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

        return loss, pred, yy