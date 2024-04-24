import torch
from torch_geometric.utils import dropout_edge

from einops import rearrange

from models.one_step.gfno import GFNO
from models.one_step.gino import GINO
from models.one_step.gno import GNO
from models.one_step.gcn import GCN
from models.one_step.gfno_efficient import GFNOEfficient
from model_module.OperatorModelModule import OperatorModelModule, LpLoss, NormError

class GraphOperatorModelModule(OperatorModelModule):
    def __init__(self, config:dict, train_example_count:int, image_size:tuple):
        super().__init__(config, train_example_count, image_size)
        self.edge_drop_rate = config.get('EDGE_DROPOUT', None)

    def get_model(self, config:dict, image_size:tuple):
        if(config['EXP_KIND'] == 'GFNO'):
            return GFNO(config)
        elif(config['EXP_KIND'] == 'GINO'):
            return GINO(config)
        elif(config['EXP_KIND'] == 'GNO'):
            return GNO(config)
        elif(config['EXP_KIND'] == 'GCN'):
            return GCN(config)
        elif(config['EXP_KIND'] == 'GFNO_EFF'):
            return GFNOEfficient(config)
        raise Exception(f"Invalid model kind specified of {config['EXP_KIND']}")
    
    def get_loss(self, config:dict):
        if(config['LOSS'] == 'LPLOSS'):
            return LpLoss()
        return NormError()
    
    def predict_step(self, batch, batch_idx):
        pred, actual, image_size = self.run_inference(batch)
        batchsize = batch.ptr.size(0) - 1
        dims = pred.size(-1)
        pred = rearrange(pred, "(b h w) c -> b h w c", b=batchsize, h=image_size[0], w=image_size[1], c=dims)
        actual = rearrange(actual, "(b h w) c -> b h w c", b=batchsize, h=image_size[0], w=image_size[1], c=dims)
        return pred, actual
    
    def run_inference(self, batch):
        xx, yy, grid, edge_index, edge_attr, image_size, batch, ptr = (
            batch.x, batch.y, batch.grid, batch.edge_index, batch.edge_attr, batch.image_size, batch.batch, batch.ptr
        )
        base_image_size = image_size[:2]
        image_size = [base_image_size[0].item(), base_image_size[1].item()]
        batch_size = len(ptr) - 1
        # run the model
        for t in range(yy.shape[-1]):
            # get the prediction at this stage
            im = self.model(xx, grid, edge_index, edge_attr, batch_size, image_size)
            # append the predictions
            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)
            # update the current observed values
            xx = torch.cat((xx[..., 1:], im), dim=-1)
        return pred, yy, base_image_size
    
    def run_batch(self, batch):
        xx, yy, grid, edge_index, edge_attr, image_size, batch, ptr = (
            batch.x, batch.y, batch.grid, batch.edge_index, batch.edge_attr, batch.image_size, batch.batch, batch.ptr
        )
        image_size = [image_size[0].item(), image_size[1].item()]
        batch_size = len(ptr) - 1
        edge_index, edge_mask = dropout_edge(edge_index, self.edge_drop_rate)
        edge_mask = torch.where(edge_mask)[0]
        edge_attr = edge_attr[edge_mask, ...]
        # run the model
        loss = 0
        for t in range(yy.shape[-1]):
            # get the prediction at this stage
            im = self.model(xx, grid, edge_index, edge_attr, batch_size, image_size)
            # update the current observed values
            xx = torch.cat((xx[..., 1:], im), dim=-1)
            # calculate the loss function
            y = yy[..., t:t + 1]
            # rearrange the things for our predictions
            im = rearrange(im, "(b h w) c -> b (h w c)", b=batch_size, h=image_size[0], w=image_size[1], c=1)
            y = rearrange(y, "(b h w) c -> b (h w c)", b=batch_size, h=image_size[0], w=image_size[1], c=1)
            loss += self.loss_fn(im, y)
        return loss