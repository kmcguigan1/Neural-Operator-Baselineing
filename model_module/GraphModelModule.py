import torch
from einops import rearrange

from torch_geometric.utils import dropout_edge
from model_module.ModelModule import ModelModule
from model_module.OperatorModelModule import LpLoss, NormError
from models.multi_step.latent_gfno import LatentGFNO

class GraphModelModule(ModelModule):
    def __init__(self, config:dict, train_example_count:int, image_size:tuple):
        super().__init__(config, train_example_count, image_size)
        self.edge_drop_rate = config['EDGE_DROPOUT']
        self.steps_out = config['TIME_STEPS_OUT']

    def get_model(self, config:dict, image_size:tuple):
        if(config['EXP_KIND'] == 'LATENT_GFNO'):
            return LatentGFNO(config)
        raise Exception(f"Invalid model kind specified of {config['EXP_KIND']}")
    
    def get_loss(self, config:dict):
        if(config['LOSS'] == 'LPLOSS'):
            return LpLoss()
        return NormError()
    
    def run_inference(self, batch):
        xx, yy, grid, edge_index, edge_attr, image_size, batch, ptr = (
            batch.x, batch.y, batch.grid, batch.edge_index, batch.edge_attr, batch.image_size, batch.batch, batch.ptr
        )
        base_image_size = image_size[:2]
        image_size = [base_image_size[0].item(), base_image_size[1].item()]
        batch_size = len(ptr) - 1
        edge_index, edge_mask = dropout_edge(edge_index, self.edge_drop_rate)
        edge_mask = torch.where(edge_mask)[0]
        edge_attr = edge_attr[edge_mask, ...]
        # run the model
        preds = self.model(xx, grid, edge_index, edge_attr, batch_size, image_size)
        preds = rearrange(preds, "(b h w) c -> b h w c", b=batch_size, h=image_size[0], w=image_size[1], c=self.steps_out)
        yy = rearrange(yy, "(b h w) c -> b h w c", b=batch_size, h=image_size[0], w=image_size[1], c=self.steps_out)
        return preds, yy, base_image_size

    def run_batch(self, batch):
        xx, yy, grid, edge_index, edge_attr, image_size, batch, ptr = (
            batch.x, batch.y, batch.grid, batch.edge_index, batch.edge_attr, batch.image_size, batch.batch, batch.ptr
        )
        base_image_size = image_size[:2]
        image_size = [base_image_size[0].item(), base_image_size[1].item()]
        batch_size = len(ptr) - 1
        edge_index, edge_mask = dropout_edge(edge_index, self.edge_drop_rate)
        edge_mask = torch.where(edge_mask)[0]
        edge_attr = edge_attr[edge_mask, ...]
        # run the model
        preds = self.model(xx, grid, edge_index, edge_attr, batch_size, image_size)
        preds = rearrange(preds, "(b h w) c -> b (h w c)", b=batch_size, h=image_size[0], w=image_size[1], c=self.steps_out)
        yy = rearrange(yy, "(b h w) c -> b (h w c)", b=batch_size, h=image_size[0], w=image_size[1], c=self.steps_out)
        loss = self.loss_fn(preds, yy)
        return loss