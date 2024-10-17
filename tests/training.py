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
from models.swin2sr import Swin2SRModule
from utils.utils import Augmentations
from utils.utils import set_global_seed
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from data_loader.biosr_dataloader import SplitDataset
from data_loader.biosr_no_patching import NoPatchingSplitDataset

set_global_seed(42)


def create_dataset(config, transform = True, noisy_data = False, noisy_factor = 0, gaus_factor = 3400, patch_size = 256):

    if transform:
        torch.manual_seed(42)
        transform = Augmentations()

    train_dataset = NoPatchingSplitDataset(
                              transform=transform,
                              data_type= config.data.data_type,
                              noisy_data=noisy_data,
                              noise_factor=noisy_factor,
                              gaus_factor=gaus_factor,
                              patch_size=patch_size,
                              mode = 'Train')
    val_dataset = NoPatchingSplitDataset(
                              transform=transform,
                              data_type= config.data.data_type,
                              noisy_data=noisy_data,
                              noise_factor=noisy_factor,
                              gaus_factor=gaus_factor,
                              patch_size=patch_size,
                              mode = 'Val')
    test_dataset = NoPatchingSplitDataset(
                              transform=transform,
                              data_type= config.data.data_type,
                              noisy_data=noisy_data,
                              noise_factor=noisy_factor,
                              gaus_factor=gaus_factor,
                              patch_size=patch_size,
                              mode = 'Test')

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    return train_loader, val_loader , test_loader


def create_model_and_train(config, logger, train_loader, val_loader, logdir):
    args = {
        "learning_rate": config.training.lr,
        "architecture": config.model.model_type,
        "dataset": "BioSRDataset",
        "epochs": config.training.num_epochs,
        "size": config.data.image_size
    }
    config_str = f"LR: {args['learning_rate']},HAGEN DATA, Noisy_data:False"


    node_name = os.environ.get('SLURMD_NODENAME', socket.gethostname())

    # Initialize WandbLogger with a custom run name
    wandb_logger = WandbLogger(save_dir=logdir, project="SwinTransformer", name=f"{node_name}" + config_str)

    wandb_logger.experiment.config.update(config.to_dict())
    model = Swin2SRModule(config)

    run_id = wandb_logger.experiment.id

    # Define two callback functions for early stopping and learning rate reduction
    model_filename = f'{run_id}swin2sr'

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=6,    # How long to wait after last improvement
        #restore_best_weights=True,  # Automatically handled by PL's checkpoint system
        mode='min')

        # Define ModelCheckpoint callback to save the best model
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        save_top_k=1,
        save_last=True,
        mode='min',
        dirpath='/home/michele.prencipe/tesi/transformer/swin2sr/logdir',
        filename=model_filename + '_best',
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = model_filename + "_last"
    callbacks=[early_stopping, checkpoint_callback]

    # Define the Trainer
    trainer = pl.Trainer(
        max_epochs=10,
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
    saving_dir = "/home/michele.prencipe/tesi/transformer/swin2sr/logdir"
    model_filepath = os.path.join(saving_dir, model_filename)
    torch.save(model.state_dict(), model_filepath)
    wandb.finish()

if __name__ == '__main__':
    logdir = 'tesi/transformer/swin2sr/logdir'
    wandb.login()
    config = get_config()
    train_loader, val_loader, test_loader= create_dataset(
        config=config,
        transform=True,
        noisy_data= False,
        noisy_factor=0,
        gaus_factor=0,
        patch_size = 256,
    )
    create_model_and_train(config=config,
                           logger=wandb,
                           train_loader=train_loader,
                           val_loader=val_loader,
                           logdir=logdir)
