import os
from datetime import datetime
import numpy as np

from torch.utils.data import DataLoader
import lightning.pytorch as pl

import wandb

from utils.constants_handler import ConstantsObject

def calculate_mean_absolute_error(forecasts:np.array, actuals:np.array, metric_name:str, split_name:str, weighted_delta:np.array=None, use_wandb:bool=False) -> None:
    abs_error = np.abs(np.subtract(forecasts, actuals))
    if(weighted_delta is not None):
        abs_error = abs_error * weighted_delta
    mean_abs_error = np.mean(abs_error)
    mean_abs_step_error = np.array([np.mean(abs_error[...,idx]) for idx in range(abs_error.shape[-1])])
    log_single_metric(split_name, metric_name, mean_abs_error, use_wandb)
    log_step_metric(split_name, f'{metric_name}_by_step', mean_abs_step_error, use_wandb)

def calculate_mean_squared_error(forecasts:np.array, actuals:np.array, metric_name:str, split_name:str, use_wandb:bool=False) -> None:
    abs_error = np.square(np.subtract(forecasts, actuals))
    mean_abs_error = np.mean(abs_error)
    mean_abs_step_error = np.array([np.mean(abs_error[...,idx]) for idx in range(abs_error.shape[-1])])
    log_single_metric(split_name, metric_name, mean_abs_error, use_wandb)
    log_step_metric(split_name, f'{metric_name}_by_step', mean_abs_step_error, use_wandb)

def calculate_relative_loss(forecasts:np.array, actuals:np.array, metric_name:str, split_name:str, order=1, use_wandb:bool=False) -> None:
    relative_forecasts = forecasts / (np.linalg.norm(forecasts, ord=order, axis=1, keepdims=True) + 1e-8)
    relative_actuals = actuals / (np.linalg.norm(actuals, ord=order, axis=1, keepdims=True) + 1e-8)
    abs_error = np.abs(np.subtract(relative_forecasts, relative_actuals))
    mean_abs_error = np.mean(abs_error)
    mean_abs_step_error = np.array([np.mean(abs_error[...,idx]) for idx in range(abs_error.shape[-1])])
    log_single_metric(split_name, metric_name, mean_abs_error, use_wandb)
    log_step_metric(split_name, f'{metric_name}_by_step', mean_abs_step_error, use_wandb)

def calculate_total_mean_absolute_error(forecasts:np.array, actuals:np.array, metric_name:str, split_name:str, use_wandb:bool=False) -> None:
    forecasts_sum = np.sum(forecasts, axis=1, keepdims=True)
    actuals_sum = np.sum(actuals, axis=1, keepdims=True)
    abs_error = np.abs(np.subtract(forecasts_sum, actuals_sum))
    mean_abs_error = np.mean(abs_error)
    mean_abs_step_error = np.array([np.mean(abs_error[...,idx]) for idx in range(abs_error.shape[-1])])
    log_single_metric(split_name, metric_name, mean_abs_error, use_wandb)
    log_step_metric(split_name, f'{metric_name}_by_step', mean_abs_step_error, use_wandb)

def get_weighted_step_changes(delta_forecasts:np.array, delta_actuals:np.array):
    delta_forecasts_weights = np.abs(delta_forecasts) / np.sum(np.abs(delta_forecasts), axis=1, keepdims=True)
    delta_actuals_weights = np.abs(delta_actuals) / np.sum(np.abs(delta_actuals), axis=1, keepdims=True)
    return delta_forecasts_weights, delta_actuals_weights

def get_step_changes(forecasts:np.array, actuals:np.array, last_input:np.array):
    print(forecasts.shape, np.concatenate((np.expand_dims(last_input, axis=-1), forecasts[..., :-1]), axis=-1).shape)
    delta_forecasts = forecasts - np.concatenate((np.expand_dims(last_input, axis=-1), forecasts[..., :-1]), axis=-1)
    delta_actuals = actuals - np.concatenate((np.expand_dims(last_input, axis=-1), actuals[..., :-1]), axis=-1)
    return delta_forecasts, delta_actuals

def flatten_outputs(forecasts:np.array, actuals:np.array, last_input:np.array):
    # the shape should be (sample, lat, lon, time)
    assert len(forecasts.shape) == 4
    assert len(actuals.shape) == 4
    assert len(last_input.shape) == 3
    # flatten what we need
    forecasts = forecasts.reshape(forecasts.shape[0], forecasts.shape[1]*forecasts.shape[2], forecasts.shape[3])
    actuals = actuals.reshape(actuals.shape[0], actuals.shape[1]*actuals.shape[2], actuals.shape[3])
    last_input = last_input.reshape(last_input.shape[0], last_input.shape[1]*last_input.shape[2])
    # now in the shape (sample, image, time)
    return forecasts, actuals, last_input

def parse_model_outputs(preds:list, idx:int) -> np.array:
    predictions = np.concatenate([
        pred[idx].detach().numpy() for pred in preds
    ])
    return predictions

def extract_model_outputs(trainer, model, data_loader, transforms=[]):
    # get the prediction outputs
    preds = trainer.predict(model, data_loader)
    # get the forecasts
    forecasts = parse_model_outputs(preds, 0)
    actuals = parse_model_outputs(preds, 1)
    last_input = parse_model_outputs(preds, 2)
    # we are going to need to undo any transformations applied to the forecasts
    for transform in transforms:
        forecasts = transform.inverse(forecasts, last_input)
    # flatten the values so that we don't have 2 spatial domains
    forecasts, actuals, last_input = flatten_outputs(forecasts, actuals, last_input)
    # return the now real data
    return forecasts, actuals, last_input

def log_single_metric(split_name, metric_name, metric_value, use_wandb):
    if(use_wandb):
        wandb.log({f'{split_name}/final/{metric_name}': metric_value})
    else:
        print(f"{split_name} {metric_name}: {metric_value}")
    return

def log_step_metric(split_name, metric_name, metric_values, use_wandb):
    if(use_wandb):
        for step in range(len(metric_values)):
            wandb.log({f'{split_name}/final/{metric_name}': metric_values[step], 'index': step})
    else:
        print(f"{split_name} {metric_name}: {metric_values}")
    return

def inverse_transform(config, dataset_statistics, forecasts, last_input):
    if("NORMALIZER" not in config.keys()):
        return forecasts, last_input
    if(config["NORMALIZER"] == 'gaus'):
        forecasts = forecasts * dataset_statistics['var'] + dataset_statistics['mean']
        last_input = last_input * dataset_statistics['var'] + dataset_statistics['mean']
    elif(config["NORMALIZER"] == 'range'):
        forecasts = forecasts * (dataset_statistics['max'] - dataset_statistics['min']) + dataset_statistics['min']
        last_input = last_input * (dataset_statistics['max'] - dataset_statistics['min']) + dataset_statistics['min']
    else:
        raise Exception(f"normalization inverse has not been defined {config['NORMALIZER']}")
    return forecasts, last_input

def evaluate_model(config, trainer, model, data_loader, split_name, dataset_statistics, use_wandb=False):
    forecasts, actuals, last_input = extract_model_outputs(trainer, model, data_loader)
    forecasts, last_input = inverse_transform(config, dataset_statistics, forecasts, last_input)
    print(f"forecasts: {forecasts.shape} actuals {actuals.shape} last input: {last_input.shape}")
    print(f"Forecasts Data mean {forecasts.mean():.4f} var {forecasts.std():.4f} min {forecasts.min():.4f} max {forecasts.max():.4f}")
    print(f"Actuals Data mean {actuals.mean():.4f} var {actuals.std():.4f} min {actuals.min():.4f} max {actuals.max():.4f}")
    # do all the simple metrics on the model predictions
    calculate_mean_absolute_error(forecasts, actuals, 'mean_absolute_error', split_name, use_wandb=use_wandb)
    calculate_mean_squared_error(forecasts, actuals, 'mean_squared_error', split_name, use_wandb=use_wandb)
    calculate_relative_loss(forecasts, actuals, 'relative_l1_norm_error', split_name, order=1, use_wandb=use_wandb)
    calculate_relative_loss(forecasts, actuals, 'relative_l2_norm_error', split_name, order=2, use_wandb=use_wandb)
    calculate_total_mean_absolute_error(forecasts, actuals, 'summed_mean_absolute_error', split_name, use_wandb=use_wandb)
    # get more complex information we need 
    delta_forecasts, delta_actuals = get_step_changes(forecasts, actuals, last_input)
    weighted_delta_forecasts, weighted_delta_actuals = get_weighted_step_changes(delta_forecasts, delta_actuals)
    # calculate the more complex metrics
    calculate_mean_absolute_error(delta_forecasts, delta_actuals, 'delta_mean_absolute_error', split_name, use_wandb=use_wandb)
    calculate_relative_loss(delta_forecasts, delta_actuals, 'delta_relative_l1_norm_error', split_name, order=1, use_wandb=use_wandb)
    calculate_mean_absolute_error(forecasts, actuals, 'delta_forecast_weighted_mean_absolute_error', split_name, weighted_delta=weighted_delta_forecasts, use_wandb=use_wandb)
    calculate_mean_absolute_error(forecasts, actuals, 'delta_actuals_weighted_mean_absolute_error', split_name, weighted_delta=weighted_delta_actuals, use_wandb=use_wandb)
    calculate_mean_absolute_error(delta_forecasts, delta_actuals, 'delta_forecast_weighted_delta_mean_absolute_error', split_name, weighted_delta=weighted_delta_forecasts, use_wandb=use_wandb)
    calculate_mean_absolute_error(delta_forecasts, delta_actuals, 'delta_actuals_weighted_delta_mean_absolute_error', split_name, weighted_delta=weighted_delta_actuals, use_wandb=use_wandb)


# def save_model_predictions()