import torch
from torch_geometric.utils import dropout_edge

from einops import rearrange

from models.one_step.beno import BENO
from model_module.OperatorModelModule import LpLoss, NormError
from model_module.GraphOperatorModelModule import GraphOperatorModelModule

class BoundaryGraphOperatorModelModule(GraphOperatorModelModule):
    def __init__(self, config:dict, train_example_count:int, image_size:tuple):
        super().__init__(config, train_example_count, image_size)

    def get_model(self, config:dict, image_size:tuple):
        if(config['EXP_KIND'] == 'BENO'):
            return BENO(config)
        raise Exception(f"Invalid model kind specified of {config['EXP_KIND']}")
    
    def get_loss(self, config:dict):
        if(config['LOSS'] == 'LPLOSS'):
            return LpLoss()
        return NormError()
    
    def run_inference(self, batch):
        xx, yy, grid, image_size, ptr = (
            batch.x, batch.y, batch.grid, batch.image_size, batch.ptr
        )
        edge_index, boundary_edge_index, boundary_node_index, boundary_node_mask = (
            batch.edge_index, batch.boundary_edge_index, batch.boundary_node_index, batch.boundary_node_mask
        )
        edge_attr, boundary_edge_attr = (
            batch.edge_attr, batch.boundary_edge_attr
        )
        base_image_size = image_size[:2]
        image_size = [base_image_size[0].item(), base_image_size[1].item()]
        batch_size = len(ptr) - 1
        # run the model
        for t in range(yy.shape[-1]):
            # get the prediction at this stage
            im = self.model(xx, grid, edge_index, edge_attr, boundary_edge_index, boundary_edge_attr, boundary_node_index, boundary_node_mask, batch_size, image_size)
            # append the predictions
            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)
            # update the current observed values
            xx = torch.cat((xx[..., 1:], im), dim=-1)
        return pred, yy, base_image_size
    
    def run_batch(self, batch):
        xx, yy, grid, image_size, ptr = (
            batch.x, batch.y, batch.grid, batch.image_size, batch.ptr
        )
        edge_index, boundary_edge_index, boundary_node_index, boundary_node_mask = (
            batch.edge_index, batch.boundary_edge_index, batch.boundary_node_index, batch.boundary_node_mask
        )
        edge_attr, boundary_edge_attr = (
            batch.edge_attr, batch.boundary_edge_attr
        )
        image_size = [image_size[0].item(), image_size[1].item()]
        batch_size = len(ptr) - 1
        # run dropout
        edge_index, edge_mask = dropout_edge(edge_index, 0.3)
        edge_mask = torch.where(edge_mask)[0]
        edge_attr = edge_attr[edge_mask, ...]
        # run model
        loss = 0
        for t in range(yy.shape[-1]):
            # get the prediction at this stage
            im = self.model(xx, grid, edge_index, edge_attr, boundary_edge_index, boundary_edge_attr, boundary_node_index, boundary_node_mask, batch_size, image_size)
            # update the current observed values
            xx = torch.cat((xx[..., 1:], im), dim=-1)
            # calculate the loss function
            y = yy[..., t:t + 1]
            # rearrange the things for our predictions
            im = rearrange(im, "(b h w) c -> b (h w c)", b=batch_size, h=image_size[0], w=image_size[1], c=1)
            y = rearrange(y, "(b h w) c -> b (h w c)", b=batch_size, h=image_size[0], w=image_size[1], c=1)
            loss += self.loss_fn(im, y)
        return loss