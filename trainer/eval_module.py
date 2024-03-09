import os
from datetime import datetime
import numpy as np

import wandb

from data_handling.data_module import DataModule
from trainer.model_module import ModelModule


def calculate_mean_absolute_error(forecasts:np.array, actuals:np.array, metric_name:str, split_name:str, weighted_delta:np.array=None) -> None:
    abs_error = np.abs(np.subtract(forecasts, actuals))
    if(weighted_delta is not None):
        abs_error = abs_error * weighted_delta
    mean_abs_error = np.mean(abs_error)
    mean_abs_step_error = np.array([np.mean(abs_error[...,idx]) for idx in range(abs_error.shape[-1])])
    log_single_metric(split_name, metric_name, mean_abs_error)
    log_step_metric(split_name, f'{metric_name}_by_step', mean_abs_step_error)
    return mean_abs_error

def calculate_mean_squared_error(forecasts:np.array, actuals:np.array, metric_name:str, split_name:str) -> None:
    abs_error = np.square(np.subtract(forecasts, actuals))
    mean_abs_error = np.mean(abs_error)
    mean_abs_step_error = np.array([np.mean(abs_error[...,idx]) for idx in range(abs_error.shape[-1])])
    log_single_metric(split_name, metric_name, mean_abs_error)
    log_step_metric(split_name, f'{metric_name}_by_step', mean_abs_step_error)

def calculate_relative_loss(forecasts:np.array, actuals:np.array, metric_name:str, split_name:str, order=1) -> None:
    relative_forecasts = forecasts / (np.linalg.norm(forecasts, ord=order, axis=1, keepdims=True) + 1e-8)
    relative_actuals = actuals / (np.linalg.norm(actuals, ord=order, axis=1, keepdims=True) + 1e-8)
    abs_error = np.abs(np.subtract(relative_forecasts, relative_actuals))
    mean_abs_error = np.mean(abs_error)
    mean_abs_step_error = np.array([np.mean(abs_error[...,idx]) for idx in range(abs_error.shape[-1])])
    log_single_metric(split_name, metric_name, mean_abs_error)
    log_step_metric(split_name, f'{metric_name}_by_step', mean_abs_step_error)

def calculate_total_mean_absolute_error(forecasts:np.array, actuals:np.array, metric_name:str, split_name:str) -> None:
    forecasts_sum = np.sum(forecasts, axis=1, keepdims=True)
    actuals_sum = np.sum(actuals, axis=1, keepdims=True)
    abs_error = np.abs(np.subtract(forecasts_sum, actuals_sum))
    mean_abs_error = np.mean(abs_error)
    mean_abs_step_error = np.array([np.mean(abs_error[...,idx]) for idx in range(abs_error.shape[-1])])
    log_single_metric(split_name, metric_name, mean_abs_error)
    log_step_metric(split_name, f'{metric_name}_by_step', mean_abs_step_error)

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

def log_single_metric(split_name, metric_name, metric_value):
    wandb.log({f'{split_name}/final/{metric_name}': metric_value})

def log_step_metric(split_name, metric_name, metric_values):
    for step in range(len(metric_values)):
        wandb.log({f'{split_name}/final/{metric_name}': metric_values[step], 'index': step})

def run_all_metrics(forecasts:np.array, actuals:np.array, last_input:np.array, split_name:str):
    # flatten the outputs
    forecasts, actuals, last_input = flatten_outputs(forecasts, actuals, last_input)
    # print the shapes
    print(f"forecasts: {forecasts.shape} actuals {actuals.shape} last input: {last_input.shape}")
    print(f"Forecasts Data mean {forecasts.mean():.4f} var {forecasts.std():.4f} min {forecasts.min():.4f} max {forecasts.max():.4f}")
    print(f"Actuals Data mean {actuals.mean():.4f} var {actuals.std():.4f} min {actuals.min():.4f} max {actuals.max():.4f}")
    # do all the simple metrics on the model predictions
    mean_abs_error = calculate_mean_absolute_error(forecasts, actuals, 'mean_absolute_error', split_name)
    calculate_mean_squared_error(forecasts, actuals, 'mean_squared_error', split_name)
    calculate_relative_loss(forecasts, actuals, 'relative_l1_norm_error', split_name, order=1)
    calculate_relative_loss(forecasts, actuals, 'relative_l2_norm_error', split_name, order=2)
    calculate_total_mean_absolute_error(forecasts, actuals, 'summed_mean_absolute_error', split_name)
    # get more complex information we need 
    delta_forecasts, delta_actuals = get_step_changes(forecasts, actuals, last_input)
    weighted_delta_forecasts, weighted_delta_actuals = get_weighted_step_changes(delta_forecasts, delta_actuals)
    # calculate the more complex metrics
    calculate_mean_absolute_error(delta_forecasts, delta_actuals, 'delta_mean_absolute_error', split_name)
    calculate_relative_loss(delta_forecasts, delta_actuals, 'delta_relative_l1_norm_error', split_name, order=1)
    calculate_mean_absolute_error(forecasts, actuals, 'delta_forecast_weighted_mean_absolute_error', split_name, weighted_delta=weighted_delta_forecasts)
    calculate_mean_absolute_error(forecasts, actuals, 'delta_actuals_weighted_mean_absolute_error', split_name, weighted_delta=weighted_delta_actuals)
    calculate_mean_absolute_error(delta_forecasts, delta_actuals, 'delta_forecast_weighted_delta_mean_absolute_error', split_name, weighted_delta=weighted_delta_forecasts)
    calculate_mean_absolute_error(delta_forecasts, delta_actuals, 'delta_actuals_weighted_delta_mean_absolute_error', split_name, weighted_delta=weighted_delta_actuals)
    return mean_abs_error

class EvalModule(object):
    def evaluate_dataset(self, data_module:DataModule, model:ModelModule, split:str='test'):
        forecasts, actuals, last_input = model.predict(data_module, split)
        mean_absolute_error = run_all_metrics(forecasts, actuals, last_input, split)
        return mean_absolute_error

    def _create_metadata_per_example(self, metadata:np.array, indecies:list, data_file:str):
        info_to_save = []
        for idx, (example_idx, time_idx) in enumerate(indecies):
            param = metadata[example_idx]
            info_to_save.append((param, time_idx))
        return np.array(info_to_save)

    def save_results(self, data_file:str, data_module:DataModule, model:ModelModule, split:str='test'):
        forecasts, actuals, last_input, metadata, indecies = model.predict(data_module, split, return_metadata=True)
        metadata = self._create_metadata_per_example(metadata, indecies, data_file)
        os.makedirs('results', exist_ok=True)
        save_name = f'{wandb.run.name}-{split}-results.npz'
        np.savez(
            os.path.join('results',save_name),
            forecasts=forecasts,
            actuals=actuals,
            last_input=last_input,
            metadata=metadata,
        )

    def evaluate(self, data_module:DataModule, model:ModelModule):
        for split in ['train', 'val', 'test']:
            mean_absolute_error = self.evaluate_dataset(data_module, model, split=split)
        return mean_absolute_error
