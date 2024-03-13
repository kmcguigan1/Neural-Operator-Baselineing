import numpy as np
import wandb
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

def log_single_metric(split_name, metric_name, metric_value):
    wandb.log({f'{split_name}/final/{metric_name}': metric_value})

def log_step_metric(split_name, metric_name, metric_values):
    for step in range(len(metric_values)):
        wandb.log({f'{split_name}/final/{metric_name}': metric_values[step], 'index': step})

def run_all_metrics(forecasts:np.array, actuals:np.array, split_name:str):
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