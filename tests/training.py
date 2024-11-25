import pytorch_lightning as pl
import os
import torch
import torch.nn as nn
import wandb
import yaml
import socket
import json
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from configs.config import get_config, save_config_to_json
from models.swin2sr import Swin2SRModule
from utils.utils import Augmentations 
from utils.utils import set_global_seed
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from data_loader.biosr_dataloader import SplitDataset
from utils.directory_setup_utils import get_workdir
import git

def add_git_info(config):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    repo = git.Repo(dir_path, search_parent_directories=True)
    config['git'] = {}
    config['git']['changedFiles'] = [item.a_path for item in repo.index.diff(None)]
    config['git']['branch'] = repo.active_branch.name
    config['git']['untracked_files'] = repo.untracked_files
    config['git']['latest_commit'] = repo.head.object.hexsha

set_global_seed(42)

def create_dataset(config, transform = True, patch_size = 256):
    if transform:
        torch.manual_seed(42)
        transform = Augmentations()
    print("NOISY:", config['data']['noisy'])
    print("POISSON",config['data']['poisson_factor'])
    print("GAUSSIAN:", config['data']['gaussian_factor'])
    train_dataset = SplitDataset(
                              transform=transform,
                              data_type= config['data']['data_type'],
                              noisy_data=config['data']['noisy'],
                              poisson_factor=config['data']['poisson_factor'],
                              gaus_factor = config['data']['gaussian_factor'],
                              patch_size=patch_size,
                              mode = 'Train')
    val_dataset = SplitDataset(
                              transform=transform,
                              data_type= config['data']['data_type'],
                              noisy_data=config['data']['noisy'],
                              poisson_factor=config['data']['poisson_factor'],
                              gaus_factor = config['data']['gaussian_factor'],
                              patch_size=patch_size,
                              mode = 'Val')
    test_dataset =  SplitDataset(
                              transform=transform,
                              data_type= config['data']['data_type'],
                              noisy_data=config['data']['noisy'],
                              poisson_factor=config['data']['poisson_factor'],
                              gaus_factor = config['data']['gaussian_factor'],
                              patch_size=patch_size,
                              mode = 'Test')

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=4)
    return train_loader, val_loader , test_loader


def create_model_and_train(config, train_loader, val_loader):
    
    root_dir = "/group/jug/Michele/training/"
    #configs =  {'data_type': 'biosr', 'learning_rate': 0.0014315884438198256, 'upscale': 1, 'in_chans': 1, 'img_size': (256, 256), 'window_size': 8, 'img_range': 1.0, 'depths': [4, 3], 'embed_dim': 60, 'num_heads': [3, 4], 'mlp_ratio': 3.5, 'upsampler': 'pixelshuffledirect', 'data': {'noisy_data': True, 'poisson_factor': 1000, 'gaussian_factor': 3400}}

    experiment_directory, rel_path= get_workdir(config, root_dir)
    add_git_info(config)
    save_config_to_json(config, experiment_directory)
        
    print('')
    print('------------------------------------')
    print('New Training Started... -> see:', experiment_directory)
    print('------------------------------------')
    config_str = f"{rel_path}"
     
    config_str = f"{config.data.data_type}, {rel_path}"
    wandb_logger = WandbLogger(save_dir=experiment_directory, project="SwinTransformer", name=config_str)
    wandb_logger.experiment.config.update(config, allow_val_change=True)
    model = Swin2SRModule(config)
    model_filename = f'swin2sr'

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=100,
        mode='min')

    # Define ModelCheckpoint callback to save the best model
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        save_top_k=1,
        save_last=True,
        mode='min',
        dirpath=experiment_directory,
        filename=model_filename + '_best',
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = model_filename + "_last"
    callbacks=[early_stopping, checkpoint_callback]

    # Define the Trainer
    trainer = pl.Trainer(
        max_epochs=1000,
        logger=wandb_logger,
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        precision=16,
        enable_progress_bar= True,
        callbacks = callbacks
    )
    # Train the model
    trainer.fit(model, train_loader, val_loader)


    model_filename = f'swin2sr'

    # Save model
    saving_dir = experiment_directory
    model_filepath = os.path.join(saving_dir, model_filename)
    torch.save(model.state_dict(), model_filepath)
    wandb.finish()

if __name__ == '__main__': 
    wandb.login()
    config = get_config()
    train_loader, val_loader, test_loader= create_dataset(
        config=config,
        transform= True,
        patch_size = 256,
    )
    create_model_and_train(config=config,
                           train_loader=train_loader,
                           val_loader=val_loader)
