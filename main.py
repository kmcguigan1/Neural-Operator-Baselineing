import gc
from datetime import datetime
import wandb

from lightning.pytorch import seed_everything

from utils.config_reader import parse_args, display_config_file, load_config
from data_handling.data_module import get_data_module, DataModule
from trainer.model_module import ModelModule
from trainer.eval_module import EvalModule

## WAND CONSTANTS
ENTITY = "kmcguigan"
PROJECT = "PDE-Operators-Baselines"

def run_experiment(config=None):
    # setup our wandb run we may choose not to track a run if we want
    # by using wandb offline or something
    with wandb.init(config=config, entity=ENTITY, project=PROJECT):
        # get the configuration
        config = wandb.config
        display_config_file(config)
        # seed the environment
        seed_everything(config['SEED'], workers=True)
        # get the data module object that holds everything to do with the 
        # data for this experiment
        data_module = get_data_module(config)
        # get the model module
        # this module handles the training and things of the model
        # it should build the model and manage its train and predict calls
        model_module = ModelModule(config)
        # now we need to fit the model
        # this is something that should be handled by the model module
        model_module.fit(data_module, config)
        # after having fit the data module we should evaluate it now
        # the model doesn't need to know how to evaluate itself
        # therefore we should use an evaluator module to do this
        evaluator_module = EvalModule()
        final_metric = evaluator_module.evaluate(data_module, model_module)
        if('SAVE_PREDS' in config.keys() and config['SAVE_PREDS'] == True):
            evaluator_module.save_results(data_module, model_module, split='test')
        # the experiment is complete and everything should be logged
        # we can now teardown our experiment in order
        # del evaluator_module
        # del model_module
        # del data_module
        gc.collect()
        return final_metric

def short_to_file_name(shorthand:str):
    shorthands = {
        'burgers_bc_fixed':'burgers_64_x_non_period_y_non_period_fixed_nu.npz',
        'burgers_bc_vary':'burgers_64_x_non_period_y_non_period_varying_nu.npz',
        'burgers_vary':'burgers_64_x_period_y_period_varying_nu.npz',
        'diff_bc_fixed':'diffusion_varying_sinusoidal_init_fixed_diffusivity_non_periodic_boundaries.npz',
        'diff_bc_vary':'diffusion_varying_sinusoidal_init_varying_diffusivity_non_periodic_boundaries.npz',
        'diff_vary':'diffusion_varying_sinusoidal_init_varying_diffusivity_periodic_boundaries.npz'
    }
    return shorthands[shorthand]

def main():
    # get the args
    args = parse_args()
    print("=========PROGRAM ARGS===========")
    print(args)
    print("========================================")
    # load the config object
    config = load_config(args.exp_kind, args.exp_name)
    # add the experiment name to the config file
    config['EXP_NAME'] = args.exp_name
    config['EXP_KIND'] = args.exp_kind
    # get the data file
    # config['DATA_FILE'] = short_to_file_name(args.data_file)
    # run the experiment
    run_experiment(config=config)

def run_as_sweep(sweep_id:str):
    wandb.agent(f'PDE-Operators-Baselines/{sweep_id}', run_experiment, count=15)

if __name__ == '__main__':
    # main()
    run_as_sweep('c7vp9fvs')


