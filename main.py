import gc
from copy import copy

import wandb
from constants import WANDB_KEY
wandb.login(key=WANDB_KEY)

from lightning.pytorch import seed_everything

from data_handling.data_reader import get_train_data_loaders, get_test_data_loaders
from trainer.trainer import LightningModel, get_lightning_trainer
from trainer.evaluator import evaluate_model


def validate_hyper_params(config: dict) -> bool:
    if(config['EXP_KIND'] in ['FNO', 'LATENT_FNO']):
        if(config['LATENT_DIMS'] // 2 + 1 < config['MODES1']):
            return False
        if(config['MODES1'] > config['LATENT_DIMS'] or config['MODES2'] > config['LATENT_DIMS']):
            return False
    if(config['LATENT_DIMS'] > 64 and config['MLP_RATIO'] > 1):
        return False
    if(config['LATENT_DIMS'] > 64 and config['DEPTH'] > 4):
        return False
    return True

def run_experiment(config: dict = None):
    # setup the wandb run as a context window
    with wandb.init(config=config):
        # get the config from the sweep controller
        config = wandb.config
        if(not validate_hyper_params(config)):
            return
        # seed the experiment
        seed_everything(config['SEED'], workers=True)
        # get the dataloaders to fit the model
        (
            train_data_loader,
            val_data_loader, 
            img_size,
            dataset_statistics
        ) = get_train_data_loaders(config)
        wandb.log({"image_size": img_size})
        # get the lightning model
        model = LightningModel(config, img_size)
        # get the trainer and fit it on the initial datasets
        trainer, timer_callback = get_lightning_trainer(config, dataset_statistics, img_size)
        trainer.fit(model=model, train_dataloaders=train_data_loader, val_dataloaders=val_data_loader)
        average_time_per_epoch = timer_callback._get_average_time_per_epoch()
        total_epochs = timer_callback.epoch_count
        wandb.log({'average_time_per_epoch':average_time_per_epoch, 'total_epochs':total_epochs})
        # delete the train and val datasets and then load the evaluation forms of them
        # the eval forms don't edit the targets so this guarentees that measured results are accurate
        del train_data_loader
        del val_data_loader
        gc.collect()
        # get the eval datasets and evaluate on them
        (
            train_set_eval_data_loader,
            val_set_eval_data_loader,
            test_set_eval_data_loader
        ) = get_test_data_loaders(config, dataset_statistics)
        evaluate_model(config, trainer, model, train_set_eval_data_loader, dataset_statistics, 'train')
        evaluate_model(config, trainer, model, val_set_eval_data_loader, dataset_statistics, 'val')
        evaluate_model(config, trainer, model, test_set_eval_data_loader, dataset_statistics, 'test')
        # cleanup the experiment before finishing this run
        del trainer
        del model
        del train_set_eval_data_loader
        del val_set_eval_data_loader
        del test_set_eval_data_loader
        gc.collect()

from utils.base import BASE_PARAMETERS

METHOD = 'FNO'
from utils.fno_config import PARAMETERS

sweep_config = {
    'method': 'random'
}

def run_sweep():
    # parameters = copy(BASE_PARAMETERS)
    # parameters.update(PARAMETERS)
    # parameters['EXP_KIND'] = {'value': METHOD}
    # sweep_config['parameters'] = parameters
    # sweep_id = wandb.sweep(sweep_config, project="PDE-Operators-Baselines")
    # print(f"Running Sweep {sweep_id}")
    # wandb.agent(sweep_id, run_experiment, count=25)
    wandb.agent('PDE-Operators-Baselines/vg50ny6d', run_experiment, count=10)

if __name__ == '__main__':
    run_sweep()
    # run_experiment(config={
    #     'EXP_KIND': 'LATENT_FNO',
    #     'SEED': 1999,
    #     'DATA_FILE': 'diffusion_varying_sinusoidal_init_fixed_diffusivity_non_periodic_boundaries.npy',
    #     'TIME_STEPS_IN': 7,
    #     'TIME_STEPS_OUT': 2,
    #     'BATCH_SIZE': 16,
    #     'TIME_INT': 1,
    #     'OPTIMIZER': {'KIND':'adam', 'LEARNING_RATE':0.001},
    #     'SCHEDULER': {'KIND':'reducer', 'FACTOR':0.1, 'PATIENCE':2},
    #     'EPOCHS': 100,
    #     'LATENT_DIMS': 32,
    #     'MODES1': 8,
    #     'MODES2': 8,
    #     'DROP_RATE': 0.0,
    #     'MLP_RATIO': 1,
    #     'DEPTH': 2,
    #     'NORMALIZATION': None,
    # })


