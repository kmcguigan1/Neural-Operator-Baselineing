import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fno import FNO2d
from models.basic_fno import FNO2d as BasicFNO2d
from models.conv_lstm import ConvLSTMModel
from models.afno import AFNO
from models.gno import GNO
from models.persistance import PersistanceModel
from models.vit import VIT

from trainer.losses_and_metrics import GausInstNorm, RangeInstNorm, PassInstNorm

class BaseModel(nn.Module):
    def __init__(self, config:dict, image_shape:tuple):
        super().__init__()
        # get the norm layer
        self.norm_layer = self._setup_norm_layer(config)
        # get the model
        self.model = self._get_model(config, image_shape)

    def _setup_norm_layer(self, config:dict, dims:tuple=(1,2,3)):
        # if we have a single sample loader then the instance wise stuff is 
        # never gonna be applied here
        if(config.get('SINGLE_SAMPLE_LOADER', False) == True):
            return PassInstNorm()
        # if we have a mutli sample model we should then grab the kind
        # of normalization we are using
        if(config['NORMALIZATION'] == 'pointwise_gaussian'):
            return GausInstNorm(dims=dims)
        if(config['NORMALIZATION'] == 'pointwise_range'):
            return RangeInstNorm(dims=dims)
        return PassInstNorm()

    def _get_model(self, config:dict, image_shape:tuple):
        if(config.get("OPERATOR_MODEL", False) == True):
            return NeuralOperatorBase(config, image_shape)
        if(config['EXP_KIND'] == 'LATENT_FNO'):
            return FNO2d(config)
        if(config['EXP_KIND'] == 'FNO'):
            return BasicFNO2d(config)
        if(config['EXP_KIND'] == 'CONV_LSTM'):
            return ConvLSTMModel(config, image_shape)
        if(config['EXP_KIND'] == 'AFNO'):
            return AFNO(config, image_shape)
        if(config['EXP_KIND'] == 'PERSISTANCE'):
            return PersistanceModel(config)
        if(config['EXP_KIND'] == 'VIT'):
            return VIT(config)
        raise Exception(f"Invalid model for base model of {config['EXP_KIND']}")

    def forward(self, batch, inference:bool=False):
        # load in the data
        x, y, grid = batch
        # apply the instance norm layer
        x, info = self.norm_layer.forward(x)
        # run the model with the other info
        preds = self.model(x, grid)
        # undo the transform
        preds = self.norm_layer.inverse(preds, info)
        if(inference):
            x = self.norm_layer.inverse(x, info)
            return x, y, preds
        return y, preds

class GraphBaseModel(BaseModel):
    def __init__(self, config:dict, image_shape:tuple):
        super().__init__(config, image_shape)

    def _get_model(self, config:dict, image_shape:tuple):
        if(config['EXP_KIND'] == 'GNO'):
            return GNO(config)
        raise Exception(f"Invalid model for base model of {config['EXP_KIND']}")

    def forward(self, batch, inference:bool=False):
        # load in the data
        x, y, grid, edges, edge_attrs = batch
        # we get x in the shape
        B, H, W, C = x.shape
        # apply the instance norm layer
        x, info = self.norm_layer.forward(x)
        # save the predictions
        predictions = torch.zeros((B, H, W, self.time_steps_out), dtype=xx.dtype, device=xx.device)
        # iterate over the batch
        for batch_idx in range(x.shape[0]):
            predictions[batch_idx, ...] = self.model(x[batch_idx, ...], grid[batch_idx, ...], edge_index[batch_idx, ...], edge_features[batch_idx, ...])
        # undo the transform
        predictions = self.norm_layer.inverse(preds, info)
        if(inference):
            x = self.norm_layer.inverse(x, info)
            return x, y, preds
        return y, preds