from datetime import datetime

import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import seed_everything

from data_handling.data_reader import get_data_loaders
from trainer.trainer import LightningModel, get_lightning_trainer
from trainer.evaluator import evaluate_model
from utils.constants_handler import ConstantsObject
from utils.config_reader import parse_args, display_config_file, load_config, setup_wandb

def main():
    # get the args
    args = parse_args()
    print("=========PROGRAM ARGS===========")
    print(args)
    print("========================================")
    # get the constants to use for this run
    constants_object = ConstantsObject(args.exp_kind)
    config = load_config(constants_object, experiment_name=args.exp_name)
    # add the experiment name to the config file
    config['EXP_NAME'] = args.exp_name
    config['EXP_KIND'] = args.exp_kind
    # seed the env
    seed_everything(config['SEED'], workers=True)
    # get the wandb stuff
    lightning_logger, run_name = setup_wandb(args, config, constants_object)
    config['RUN_NAME'] = run_name
    display_config_file(config)
    # get the dataloaders
    (
        train_data_loader,
        val_data_loader, 
        test_data_loader, 
        train_example_count, 
        train_example_shape
    ) = get_data_loaders(config, constants_object)
    # get the model
    img_size = (train_example_shape[-2],train_example_shape[-1])
    model = LightningModel(config, constants_object, train_example_count, img_size)
    model._print_summary(train_example_shape)
    # get and fit the trainer
    trainer = get_lightning_trainer(config, lightning_logger=lightning_logger, accelerator=constants_object.ACCELERATOR)
    trainer.fit(model=model, train_dataloaders=train_data_loader, val_dataloaders=val_data_loader)
    # evaluate the model on all datasets
    evaluate_model(trainer, model, train_data_loader, 'train', use_wandb=args.run_wandb)
    evaluate_model(trainer, model, val_data_loader, 'val', use_wandb=args.run_wandb)
    evaluate_model(trainer, model, test_data_loader, 'test', use_wandb=args.run_wandb)

if __name__ == '__main__':
    main()


