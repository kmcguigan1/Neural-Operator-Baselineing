from trainer.losses_and_metrics import GausInstNorm, RangeInstNorm, PassInstNorm

class LightningModule(pl.LightningModule):
    def __init__(self, config:dict, image_shape:tuple):
        super().__init__()
        self.automatic_optimization = True
        self.leraning_rate = config['LEARNING_RATE']
        self.optimizer_params = config['OPTIMIZER']
        self.scheduler_params = config['SCHEDULER']
        # get the loss and metrics
        if(config['LOSS'] == 'LPLOSS'):
            self._loss_fn = LpLoss()
        elif(config['LOSS'] == 'MSE'):
            self._loss_fn = MSELoss()
        elif(config['LOSS'] == 'HSLOSS'):
            raise Exception('HS LOSS not implemented')
        else:
            raise Exception(f'Invalid loss {config["LOSS"]}')
        self._train_acc = CustomMAE()
        self._val_acc = CustomMAE()
        # get the model that we will be using
        self.model = self._get_model(config, image_shape)
        # get the normalization layer
        self.norm_layer = self._setup_norm_layer(config)

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
        if(config['EXP_KIND'] == 'LATENT_FNO'):
            return FNO2d(config)
        if(config['EXP_KIND'] == 'CONV_LSTM'):
            return ConvLSTMModel(config, image_shape)
        if(config['EXP_KIND'] == 'AFNO'):
            return AFNO(config, image_shape)
        if(config['EXP_KIND'] == 'PERSISTANCE'):
            return PersistanceModel(config)
        if(config['EXP_KIND'] == 'VIT'):
            return VIT(config)
        raise Exception(f"Invalid model for base model of {config['EXP_KIND']}")

    def _run_model(self, batch, inference:bool=False):
        # pull the data
        x, y, grid = batch
        # normalize the data
        x, info = self.norm_layer.forward(x)
        # run the model with the other info
        preds = self.model(x, grid)
        # undo the transform
        preds = self.norm_layer.inverse(preds, info)
        if(inference):
            x = self.norm_layer.inverse(x, info)
            return x, y, preds
        return y, preds 

    def training_step(self, batch, batch_idx):
        preds, y = self._run_model(batch)
        loss = self._loss_fn(preds, y)
        self._train_acc(preds, y)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/mae', self._train_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        preds, y = self._run_model(batch)
        loss = self._loss_fn(preds, y)
        self._val_acc(preds, y)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/mae', self._val_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
    def predict_step(self, batch, batch_idx):
        x, y, preds = self._run_model(batch, inference=True)
        return preds, y, x

    def configure_optimizers(self):
        # get the optimizer
        if(self.optimizer_params['KIND'] == 'adam'):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.leraning_rate)#, weight_decay=self.optimizer_params['WEIGHT_DECAY'])
        else:
            raise Exception(f"Invalid optimizer specified of {self.optimizer_params['KIND']}")
        # get the 
        if(self.scheduler_params['KIND'] == 'cosine'):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self._train_example_count*self._config['EPOCHS'], eta_min=self.scheduler_params['MIN_LR'])
            return [optimizer], [{"scheduler":scheduler,"interval":"step"}]
        elif(self.scheduler_params['KIND'] == 'reducer'):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.scheduler_params['PATIENCE'], factor=self.scheduler_params['FACTOR'], verbose=True)
            return [optimizer], [{"scheduler":scheduler,"interval":"epoch","monitor":"val/loss"}]
        else:
            raise Exception(f"Invalid scheduler specified of {self.scheduler_params['KIND']}")

class GraphLightningModule(LightningModule):
    def __init__(self, config:dict, image_shape:tuple):
        super().__init__(config, image_shape)
        self.time_steps_out = config["TIME_STEPS_OUT"]

    def _get_model(self, config:dict, image_shape:tuple):
        if(config['EXP_KIND'] == 'LATENT_GNO'):
            raise NotImplementedError('Implement Latent GNO')
            #return LatentGNO(config)
        raise Exception(f"Invalid model for base model of {config['EXP_KIND']}")

    def _run_model(self, batch, inference:bool=False):
        # pull the data
        x, y, grid, edge_index, edge_attrs = batch
        # normalize the data
        x, info = self.norm_layer.forward(x)
        # save the predictions
        preds = torch.zeros((B, H, W, self.time_steps_out), dtype=x.dtype, device=x.device)
        # iterate over the batch
        for batch_idx in range(x.shape[0]):
            preds[batch_idx, ...] = self.model(x[batch_idx, ...], grid[batch_idx, ...], edge_index[batch_idx, ...], edge_features[batch_idx, ...])
        # undo the transform
        preds = self.norm_layer.inverse(preds, info)
        # undo the transform
        preds = self.norm_layer.inverse(preds, info)
        if(inference):
            x = self.norm_layer.inverse(x, info)
            return x, y, preds
        return y, preds

class OperatorLightningModule(LightningModule):
    def __init__(self, config:dict, image_shape:tuple):
        super().__init__(config, image_shape)
        self.time_steps_out = config["TIME_STEPS_OUT"]

    def _get_model(self, config:dict, image_shape:tuple):
        if(config['EXP_KIND'] == 'FNO'):
            return BasicFNO2d(config)
        raise Exception(f"Invalid model for base model of {config['EXP_KIND']}")

    def _run_model(self, batch, inference:bool=False):
        # pull the data
        x, y, grid, edge_index, edge_attrs = batch
        # normalize the data
        x, info = self.norm_layer.forward(x)
        # save the predictions
        preds = torch.zeros((B, H, W, self.time_steps_out), dtype=xx.dtype, device=xx.device)
        # iterate over the batch
        for batch_idx in range(x.shape[0]):
            preds[batch_idx, ...] = self.model(x[batch_idx, ...], grid[batch_idx, ...], edge_index[batch_idx, ...], edge_features[batch_idx, ...])
        # undo the transform
        preds = self.norm_layer.inverse(preds, info)
        # undo the transform
        preds = self.norm_layer.inverse(preds, info)
        if(inference):
            x = self.norm_layer.inverse(x, info)
            return x, y, preds
        return y, preds

class GraphMultiSampleLightningModule(LightningModule):
    def __init__(self, config:dict, image_shape:tuple):
        super().__init__(config, image_shape)
        self.norm_layer = self._setup_norm_layer(config)
        self.time_steps_out = config["TIME_STEPS_OUT"]

    def _setup_norm_layer(self, config:dict, dims:tuple=(1,2,3)):
        if(config['NORMALIZATION'] == 'pointwise_gaussian'):
            return GausInstNorm(dims=dims)
        if(config['NORMALIZATION'] == 'pointwise_range'):
            return RangeInstNorm(dims=dims)
        return PassInstNorm()

    def _get_model(self, config:dict, image_shape:tuple):
        raise Exception(f"Invalid model for base model of {config['EXP_KIND']}")

    def _run_model(self, batch, inference:bool=False):
        # pull the data
        x, y, grid, edge_index, edge_attrs = batch
        # normalize the data
        x, info = self.norm_layer.forward(x)
        # save the predictions
        preds = torch.zeros((B, H, W, self.time_steps_out), dtype=xx.dtype, device=xx.device)
        # iterate over the batch
        for batch_idx in range(x.shape[0]):
            preds[batch_idx, ...] = self.model(x[batch_idx, ...], grid[batch_idx, ...], edge_index[batch_idx, ...], edge_features[batch_idx, ...])
        # undo the transform
        preds = self.norm_layer.inverse(preds, info)
        # undo the transform
        preds = self.norm_layer.inverse(preds, info)
        if(inference):
            x = self.norm_layer.inverse(x, info)
            return x, y, preds
        return y, preds

class SampleLightningModule(LightningModule):
    def __init__(self, config:dict, image_shape:tuple):
        super().__init__(config, image_shape)
        self.norm_layer = self._setup_norm_layer(config)

    def _setup_norm_layer(self, config:dict, dims:tuple=(1,2,3)):
        if(config['NORMALIZATION'] == 'pointwise_gaussian'):
            return GausInstNorm(dims=dims)
        if(config['NORMALIZATION'] == 'pointwise_range'):
            return RangeInstNorm(dims=dims)
        return PassInstNorm()

    def _get_model(self, config:dict, image_shape:tuple):
        if(config['EXP_KIND'] == 'LATENT_FNO'):
            return FNO2d(config)
        if(config['EXP_KIND'] == 'CONV_LSTM'):
            return ConvLSTMModel(config, image_shape)
        if(config['EXP_KIND'] == 'AFNO'):
            return AFNO(config, image_shape)
        if(config['EXP_KIND'] == 'PERSISTANCE'):
            return PersistanceModel(config)
        if(config['EXP_KIND'] == 'VIT'):
            return VIT(config)
        raise Exception(f"Invalid model for base model of {config['EXP_KIND']}")

    def _run_model(self, batch, inference:bool=False):
        # pull the data
        x, y, grid = batch
        # normalize the data
        x, info = self.norm_layer.forward(x)
        # run the model with the other info
        preds = self.model(x, grid)
        # undo the transform
        preds = self.norm_layer.inverse(preds, info)
        if(inference):
            x = self.norm_layer.inverse(x, info)
            return x, y, preds
        return y, preds