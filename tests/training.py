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
from models.network_swin2sr import Swin2SR
from torch.utils.data import random_split
from data_loader.biosr_dataset import BioSRDataLoader
from configs.biosr_config import get_config
#from configs.hagen_config import get_config
from models.swin2sr import Swin2SRModule
from utils.utils import Augmentations
from utils.utils import set_global_seed
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from data_loader.biosr_dataloader import SplitDataset
from data_loader.biosr_no_patching import NoPatchingSplitDataset
from utils.directory_setup_utils import get_workdir


set_global_seed(42)


def create_dataset(config, transform = True, noisy_data = False, noisy_factor = 0, gaus_factor = 6800, patch_size = 256):

    if transform:
        torch.manual_seed(42)
        transform = Augmentations()

    train_dataset = SplitDataset(
                              transform=transform,
                              data_type= config.data.data_type,
                              noisy_data=noisy_data,
                              noise_factor=noisy_factor,
                              gaus_factor=gaus_factor,
                              patch_size=patch_size,
                              mode = 'Train')
    val_dataset = SplitDataset(
                              transform=transform,
                              data_type= config.data.data_type,
                              noisy_data=noisy_data,
                              noise_factor=noisy_factor,
                              gaus_factor=gaus_factor,
                              patch_size=patch_size,
                              mode = 'Val')
    test_dataset =  SplitDataset(
                              transform=transform,
                              data_type= config.data.data_type,
                              noisy_data=noisy_data,
                              noise_factor=noisy_factor,
                              gaus_factor=gaus_factor,
                              patch_size=patch_size,
                              mode = 'Test')

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)
    return train_loader, val_loader , test_loader


def create_model_and_train(config, train_loader, val_loader):
    
    root_dir = "/group/jug/Michele/training/"
    configs = {
            'data_type': 'biosr', 
            'learning_rate': 0.00359,
            'upscale': 1,
            'in_chans': 1,
            'img_size': (256, 256),
            'window_size': 16,  # search space for window size
            'img_range': 1.0,
            'depths': [6,4],   # number of transformer blocks at stage 2
            'embed_dim':   60 ,  # embedding dimensions
            'num_heads': [8,8], # number of heads for stage 1   # number of heads for stage 2,
            'mlp_ratio':   3,  # MLP expansion ratio
            'upsampler': 'pixelshuffledirect',
            'data':{
                'noisy_data': True,
                'poisson_factor': 0,
                'gaussian_factor': 3400
            }
    }
    
    
    
    experiment_directory, rel_path= get_workdir(configs,root_dir)
    # save the dictionary to file
    with open(os.path.join(experiment_directory,'config.json'), 'w') as f:
        json.dump(configs, f)
        
    print('')
    print('------------------------------------')
    print('New Training Started... -> see:', experiment_directory)
    print('------------------------------------')
    config_str = f"{rel_path}"
     
    config_str = f"{config.data.data_type}, {rel_path}"
    wandb_logger = WandbLogger(save_dir=experiment_directory, project="SwinTransformer", name=config_str)
    wandb_logger.experiment.config.update(config, allow_val_change=True)
    model = Swin2SRModule(config)
    print("Model parameter", model.get_parameter)
    run_id = wandb_logger.experiment.id
    model_filename = f'{run_id}swin2sr'

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=100,    # How long to wait after last improvement
        #restore_best_weights=True,  # Automatically handled by PL's checkpoint system
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


    model_filename = f'{run_id}swin2sr'

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
        noisy_data= True,
        noisy_factor= 3400,
        gaus_factor= 0,
        patch_size = 256,
    )
    create_model_and_train(config=config,
                           train_loader=train_loader,
                           val_loader=val_loader)
