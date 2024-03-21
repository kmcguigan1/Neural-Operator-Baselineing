import os
from datetime import datetime
import numpy as np
import pandas as pd
import wandb

from constants import RESULTS_PATH
EPSILON = 1e-5

def calculate_mean_absolute_error(forecasts:np.array, actuals:np.array, metric_name:str, split_name:str) -> None:
    abs_error = np.abs(np.subtract(forecasts, actuals))
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

def calculate_normalized_error(forecasts:np.array, actuals:np.array, metric_name:str, split_name:str, order:int=2) -> None:
    normalized_forecasts = forecasts / (np.linalg.norm(forecasts, ord=order, axis=1, keepdims=True) + EPSILON)
    normalized_actuals = actuals / (np.linalg.norm(actuals, ord=order, axis=1, keepdims=True) + EPSILON)
    abs_error = np.abs(np.subtract(normalized_forecasts, normalized_actuals))
    mean_abs_error = np.mean(abs_error)
    mean_abs_step_error = np.array([np.mean(abs_error[...,idx]) for idx in range(abs_error.shape[-1])])
    log_single_metric(split_name, metric_name, mean_abs_error)
    log_step_metric(split_name, f'{metric_name}_by_step', mean_abs_step_error)

def calculate_relative_error(forecasts:np.array, actuals:np.array, metric_name:str, split_name:str, order:int=2):
    # more scale agnostic
    # we get the root summed squared error
    # we divide this by the root summed squared actuals
    # we could do this with l1 norm if we want to have less outlier influence
    # but basically this takes the relative error which is helpful
    # we underweight error when the values are larger, this makes sense since the error
    # is a smaller relative proportion
    
    # we could do this stepwise by taking norm over the spatial axes then like dividing and stuff
    # this would give the proper result
    x = forecasts.reshape(forecasts.shape[0], -1)
    y = actuals.reshape(actuals.shape[0], -1)
    diff_norms = np.linalg.norm(x - y, ord=order, axis=1)
    y_norms = np.linalg.norm(y, ord=order, axis=1)
    relative_error = np.mean(diff_norms / y_norms)
    log_single_metric(split_name, metric_name, relative_error)

def flatten_outputs(forecasts:np.array, actuals:np.array):
    # the shape should be (sample, lat, lon, time)
    assert len(forecasts.shape) == 4
    assert len(actuals.shape) == 4
    # flatten what we need
    forecasts = forecasts.reshape(forecasts.shape[0], forecasts.shape[1]*forecasts.shape[2], forecasts.shape[3])
    actuals = actuals.reshape(actuals.shape[0], actuals.shape[1]*actuals.shape[2], actuals.shape[3])
    # now in the shape (sample, image, time)
    return forecasts, actuals

def generate_boundary_distance_mask(nx:int, ny:int):
    # define the boundaries grid mask
    x_bounds = np.linspace(start=0, stop=1, num=nx).reshape(-1, 1).repeat(ny, axis=1)
    x_bounds = np.stack((x_bounds, 1.0 - x_bounds), axis=-1).min(axis=-1)
    y_bounds = np.linspace(start=0, stop=1, num=ny).reshape(1, -1).repeat(nx, axis=0)
    y_bounds = np.stack((y_bounds, 1.0 - y_bounds), axis=-1).min(axis=-1)
    bounds = np.stack((x_bounds, y_bounds), axis=-1).min(axis=-1)
    return bounds

def calculate_boundary_distance_error(forecasts:np.ndarray, actuals:np.ndarray, bounds:np.ndarray, split_name:str):
    # (example, dim, dim, time)
    total_point_energy = np.sqrt(np.sum(np.square(actuals), axis=-1, keepdims=False))
    total_error_energy = np.sqrt(np.sum(np.square(forecasts - actuals), axis=-1, keepdims=False))
    print(total_point_energy.shape)
    print(total_error_energy.shape)

    normalized_error = total_error_energy / total_point_energy
    print(normalized_error.shape)
    normalized_error = normalized_error.mean(axis=0)
    print(normalized_error.shape)

    unnormalized_error = total_error_energy.mean(axis=0)
    print(unnormalized_error.shape)

    df = pd.DataFrame()
    df['dist_to_boundary'] = bounds.flatten()
    df['normalized_error'] = normalized_error.flatten()
    df['error'] = unnormalized_error.flatten()
    df = df.sort_values(by='dist_to_boundary', ascending=True)

    for _, row in df.iterrows():
        wandb.log({
            f'{split_name}/final/relative_boundary_error': row['normalized_error'],
            f'{split_name}/final/boundary_error': row['error'],
            'index': row['dist_to_boundary'] 
        })
    
    return 

    # N = 20
    # increment = df.shape[0] // N
    # lower = 0
    # for idx in range(N):
    #     upper = lower + increment
    #     wandb.log({
    #         f'{split_name}/final/relative_boundary_error': df.iloc[lower:upper]['normalized_error'].mean(),
    #         'index': idx
    #     })
    #     wandb.log({
    #         f'{split_name}/final/relative_boundary_error': df.iloc[lower:upper]['error'].mean(),
    #         'index': idx
    #     })
    #     lower = upper


    # bound_groups = np.linspace(start=0, stop=bounds.max(), num=15)
    # for idx in range(1, bound_groups.shape[0]):
    #     upper_bound = bound_groups[idx]
    #     lower_bound = bound_groups[idx - 1]
    #     if(idx == 1):
    #         lower_bound = -1
    #     indecies = np.where(np.logical_and(bounds <= upper_bound, bounds > lower_bound))
    #     print(indecies[0].shape)
    #     wandb.log({f'{split_name}/final/boundary_dist_count': indecies[0].shape, 'index': upper_bound})
    #     error = normalized_error[indecies[0], indecies[1]].mean()
    #     wandb.log({f'{split_name}/final/relative_boundary_error': error, 'index': upper_bound})
    #     error = unnormalized_error[indecies[0], indecies[1]].mean()
    #     wandb.log({f'{split_name}/final/boundary_error': error, 'index': upper_bound})

def log_single_metric(split_name, metric_name, metric_value):
    wandb.log({f'{split_name}/final/{metric_name}': metric_value})

def log_step_metric(split_name, metric_name, metric_values):
    for step in range(len(metric_values)):
        wandb.log({f'{split_name}/final/{metric_name}': metric_values[step], 'index': step})

def run_all_metrics(forecasts:np.array, actuals:np.array, split_name:str):
    # before flattening run shape dependent metrics
    # get the error as a factor of distance from the edges to see
    # the impacts of boundary conditions
    bounds = generate_boundary_distance_mask(forecasts.shape[1], forecasts.shape[2])
    calculate_boundary_distance_error(forecasts, actuals, bounds, split_name)
    # flatten the outputs
    forecasts, actuals = flatten_outputs(forecasts, actuals)
    # print the shapes
    print(f"forecasts: {forecasts.shape} actuals {actuals.shape}")
    print(f"Forecasts Data mean {forecasts.mean():.4f} var {forecasts.std():.4f} min {forecasts.min():.4f} max {forecasts.max():.4f}")
    print(f"Actuals Data mean {actuals.mean():.4f} var {actuals.std():.4f} min {actuals.min():.4f} max {actuals.max():.4f}")
    # do all the simple metrics on the model predictions
    mean_abs_error = calculate_mean_absolute_error(forecasts, actuals, 'mean_absolute_error', split_name)
    calculate_mean_squared_error(forecasts, actuals, 'mean_squared_error', split_name)
    calculate_normalized_error(forecasts, actuals, 'l1_norm_error', split_name, order=1)
    calculate_relative_error(forecasts, actuals, 'l2_norm_relative_error', split_name, order=2)
    calculate_relative_error(forecasts, actuals, 'l1_norm_relative_error', split_name, order=1)
    return mean_abs_error

def save_predictions(predictions:np.ndarray, actuals:np.ndarray, indecies:np.ndarray, split:str, data_file:str):
    os.makedirs(RESULTS_PATH, exist_ok=True)
    run_name = wandb.run.name
    if(len(run_name) < 2):
        run_name = datetime.now().strftime("%d-%m-%Y--%H-%M-%S")

    if(indecies is None):
        indecies = np.array([-1])
    np.savez(
        os.path.join(RESULTS_PATH, f'{run_name}-{split}.npz'),
        predictions=predictions,
        actuals=actuals,
        split=split,
        data_file=data_file,
        indecies=indecies,
    )