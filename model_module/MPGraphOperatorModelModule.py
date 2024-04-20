import torch
from torch_geometric.utils import dropout_edge

from einops import rearrange

from models.one_step.mpgno import MPGNO
from model_module.OperatorModelModule import LpLoss, NormError
from model_module.GraphOperatorModelModule import GraphOperatorModelModule

class MPGraphOperatorModelModule(GraphOperatorModelModule):
    def __init__(self, config:dict, train_example_count:int, image_size:tuple):
        super().__init__(config, train_example_count, image_size)

    def get_model(self, config:dict, image_size:tuple):
        if(config['EXP_KIND'] == 'MPGNO'):
            return MPGNO(config)
        raise Exception(f"Invalid model kind specified of {config['EXP_KIND']}")
    
    def get_loss(self, config:dict):
        if(config['LOSS'] == 'LPLOSS'):
            return LpLoss()
        return NormError()
    
    def run_inference(self, batch):
        xx, yy, grid, image_size, batch, ptr = (
            batch.x, batch.y, batch.grid, batch.image_size, batch.batch, batch.ptr
        )
        edge_index_1, edge_index_2, edge_index_3 = (
            batch.edge_index_1, batch.edge_index_2, batch.edge_index_3 
        )
        edge_attr_1, edge_attr_2, edge_attr_3 = (
            batch.edge_attr_1, batch.edge_attr_2, batch.edge_attr_3 
        )
        image_size = image_size[:2]
        batch_size = len(ptr) - 1
        # run the model
        for t in range(yy.shape[-1]):
            # get the prediction at this stage
            im = self.model(xx, grid, edge_index_1, edge_index_2, edge_index_3, edge_attr_1, edge_attr_2, edge_attr_3, batch_size, image_size)
            # append the predictions
            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)
            # update the current observed values
            xx = torch.cat((xx[..., 1:], im), dim=-1)
        return pred, yy, image_size
    
    def run_batch(self, batch):
        xx, yy, grid, image_size, batch, ptr = (
            batch.x, batch.y, batch.grid, batch.image_size, batch.batch, batch.ptr
        )
        edge_index_1, edge_index_2, edge_index_3 = (
            batch.edge_index_1, batch.edge_index_2, batch.edge_index_3 
        )
        edge_attr_1, edge_attr_2, edge_attr_3 = (
            batch.edge_attr_1, batch.edge_attr_2, batch.edge_attr_3 
        )
        image_size = image_size[:2]
        batch_size = len(ptr) - 1
        # run dropout
        edge_index_1, edge_mask_1 = dropout_edge(edge_index_1, 0.4)
        edge_mask_1 = torch.where(edge_mask_1)[0]
        edge_attr_1 = edge_attr_1[edge_mask_1, ...]

        edge_index_2, edge_mask_2 = dropout_edge(edge_index_2, 0.25)
        edge_mask_2 = torch.where(edge_mask_2)[0]
        edge_attr_2 = edge_attr_2[edge_mask_2, ...]

        edge_index_3, edge_mask_3 = dropout_edge(edge_index_3, 0.1)
        edge_mask_3 = torch.where(edge_mask_3)[0]
        edge_attr_3 = edge_attr_3[edge_mask_3, ...]
        # run the model
        loss = 0
        for t in range(yy.shape[-1]):
            # get the prediction at this stage
            im = self.model(xx, grid, edge_index_1, edge_index_2, edge_index_3, edge_attr_1, edge_attr_2, edge_attr_3, batch_size, image_size)
            # update the current observed values
            xx = torch.cat((xx[..., 1:], im), dim=-1)
            # calculate the loss function
            y = yy[..., t:t + 1]
            # rearrange the things for our predictions
            im = rearrange(im, "(b h w) c -> b (h w c)", b=batch_size, h=image_size[0], w=image_size[1], c=1)
            y = rearrange(y, "(b h w) c -> b (h w c)", b=batch_size, h=image_size[0], w=image_size[1], c=1)
            loss += self.loss_fn(im, y)
        return loss