import torch
from torch_geometric.utils import dropout_edge

from einops import rearrange

from models.operators.gcn import GCN_Net as GCNModel
from models.operators.gkn import KernelNN as GKNModel
from models.operators.gno import KernelNN as GNOModel
from models.operators.mgkn import MKGN as MKGNModel
from models.operators.my_operator import CustomOPP
from model_module.OperatorModelModule import OperatorModelModule

class GraphOperatorModelModule(OperatorModelModule):
    def __init__(self, config:dict, train_example_count:int, image_size:tuple):
        super().__init__(config, train_example_count, image_size)
        self.edge_drop_rate = config['EDGE_DROPOUT']

    def get_model(self, config:dict, image_size:tuple):
        if(config['EXP_KIND'] == 'GNO'):
            return GNOModel(config)
        elif(config['EXP_KIND'] == 'BNO'):
            return CustomOPP(config, image_size)
        elif(config['EXP_KIND'] == 'GCN'):
            return GCNModel(config)
        elif(config['EXP_KIND'] == 'GKN'):
            return GKNModel(config)
        elif(config['EXP_KIND'] == 'MGKN'):
            return MKGNModel(config)
        raise Exception(f"Invalid model kind specified of {config['EXP_KIND']}")
    
    def get_loss(self, config:dict):
        return None
    
    def predict_step(self, batch, batch_idx):
        _, pred, actual, image_size = self.run_batch(batch)
        batchsize = batch.ptr.size(0) - 1
        dims = pred.size(-1)
        pred = rearrange(pred, "(b h w) c -> b h w c", b=batchsize, h=image_size[0], w=image_size[1], c=dims)
        actual = rearrange(actual, "(b h w) c -> b h w c", b=batchsize, h=image_size[0], w=image_size[1], c=dims)
        return pred, actual

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