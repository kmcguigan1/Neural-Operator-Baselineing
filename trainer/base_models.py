from models.fno import FNO2d
from models.basic_fno import FNO2d as BasicFNO2d
from models.conv_lstm import ConvLSTMModel
from models.afno import AFNO
from models.gno import GNO
from models.persistance import PersistanceModel
from models.vit import VIT

from trainer.losses_and_metrics import GausInstNorm, RangeInstNorm, PassInstNorm, CustomMAE, LpLoss, TimingCallback

class BaseModel(nn.Module):
    def __init__(self, config:dict, image_shape:tuple):
        super().__init__()
        if(config.get("OPERATOR_MODEL", False) == True):
            self.model = NeuralOperatorBase(config, image_shape)
        elif():
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

class GraphBase(nn.Module):
    def __init__(self, config:dict, image_shape:tuple):
        super().__init__()
        if(config.get("OPERATOR_MODEL", False) == True):
            self.model = GraphNeuralOperatorBase(config, image_shape)
        else:
            raise Exception(f"Operator not implemented yet {config['EXP_KIND']}")
    def forward(self, x, grid, edges, edge_attrs):
        # we get x in the shape
        B, H, W, C = x.shape
        # save the predictions
        predictions = torch.zeros((B, H, W, self.time_steps_out), dtype=xx.dtype, device=xx.device)
        # iterate over the batch
        for batch_idx in range(x.shape[0]):
            predictions[batch_idx, ...] = self.model(x[batch_idx, ...], grid[batch_idx, ...], edge_index[batch_idx, ...], edge_features[batch_idx, ...])
        # undo the transform
        preds = self.norm_layer.inverse(preds, info)
        if(inference):
            x = self.norm_layer.inverse(x, info)
            return x, y, preds
        return y, preds

class NeuralOperatorBase(nn.Module):
    def __init__(self, config:dict, image_shape:tuple):
        super().__init__()
        self.time_steps_out = config['TIME_STEPS_OUT']
        if(config['EXP_KIND'] == 'FNO'):
            self.operator = BasicFNO2d(config)
        else:
            raise Exception(f"Operator not implemented yet {config['EXP_KIND']}")
    def forward(self, xx, grid):
        # we get x in the shape
        B, H, W, C = xx.shape
        # save the predictions
        predictions = torch.zeros((B, H, W, self.time_steps_out), dtype=xx.dtype, device=xx.device)
        # iterate over the steps we have to forecast ahead
        for idx in range(self.time_steps_out):
            # add the new predictions to the x data
            if(idx > 0):
                xx = torch.cat((xx[..., 1:], predictions[..., idx-1:idx]), dim=-1)
            # apply the operator
            x = self.operator(xx, grid)
            # save the prediction
            predictions[..., idx] = x
        return predictions

class GraphNeuralOperatorBase(nn.Module):
    def __init__(self, config:dict, image_shape:tuple):
        super().__init__()
        self.time_steps_out = config['TIME_STEPS_OUT']
        if(config['EXP_KIND'] == 'GNO'):
            self.operator = GNO(config)
        else:
            raise Exception(f"Operator not implemented yet {config['EXP_KIND']}")
    def forward(self, xx, grid, edges, edge_attrs):
        # we get x in the shape
        B, H, W, C = xx.shape
        # save the predictions
        predictions = torch.zeros((B, H, W, self.time_steps_out), dtype=xx.dtype, device=xx.device)
        # iterate over the steps we have to forecast ahead
        for idx in range(self.time_steps_out):
            # add the new predictions to the x data
            if(idx > 0):
                xx = torch.cat((xx[..., 1:], predictions[..., idx-1:idx]), dim=-1)
            # apply the operator
            x = self.operator(xx, grid, edges, edge_attrs)
            # save the prediction
            predictions[..., idx] = x
        return predictions






class BaseModel(nn.Module):
    def __init__(self, config:dict, image_shape:tuple):
        super().__init__()
        # get the norm layer
        self.norm_layer = self._setup_norm_layer(config)
        # get the model
        if(config['EXP_KIND'] == 'LATENT_FNO'):
            self.model = FNO2d(config)
        elif(config['EXP_KIND'] == 'FNO'):
            self.model = BasicFNO2d(config)
        elif(config['EXP_KIND'] == 'CONV_LSTM'):
            self.model = ConvLSTMModel(config, image_shape)
        elif(config['EXP_KIND'] == 'AFNO'):
            self.model = AFNO(config, image_shape)
        elif(config['EXP_KIND'] == 'GNO'):
            self.model = GNO(config)
        elif(config['EXP_KIND'] == 'PERSISTANCE'):
            self.model = PersistanceModel(config)
        elif(config['EXP_KIND'] == 'VIT'):
            self.model = VIT(config)
        else:
            raise Exception(f"{config['EXP_KIND']} is not implemented please implement this.")

    def _setup_norm_layer(self, config:dict):
        return self._get_norm_layer_with_dims(config)

    def _get_norm_layer_with_dims(self, config:dict, dims:tuple=(1,2,3)):
        # get the normalization kind
        if(config['NORMALIZATION'] == 'gaus_inst'):
            norm_layer = GausInstNorm(dims=dims)
        elif(config['NORMALIZATION'] == 'range_inst'):
            norm_layer = RangeInstNorm(dims=dims)
        else:
            norm_layer = PassInstNorm()
        return norm_layer

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

    def _setup_norm_layer(self, config:dict):
        return self._get_norm_layer_with_dims(config, dims=(1,2))

    def forward(self, batch, inference:bool=False):
        # load in the data
        x, y, grid, edge_index, edge_features = batch
        # apply the instance norm layer
        x, info = self.norm_layer.forward(x)
        # run the model with the other info
        preds = torch.zeros_like(y, device=x.device)
        for batch_idx in range(x.shape[0]):
            preds[batch_idx, ...] = self.model(x[batch_idx, ...], grid[batch_idx, ...], edge_index[batch_idx, ...], edge_features[batch_idx, ...])
        # undo the transform
        preds = self.norm_layer.inverse(preds, info)
        if(inference):
            x = self.norm_layer.inverse(x, info)
            return x, y, preds
        return y, preds