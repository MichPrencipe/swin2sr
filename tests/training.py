import pytorch_lightning as pl
import os
import torch
import torch.nn as nn
import wandb

import time

from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from models.network_swin2sr import Swin2SR
from torch.utils.data import DataLoader, random_split


import wandb
import torch.nn as nn
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from data_loader.biosr_dataset import BioSRDataLoader

from configs.biosr_config import get_config
from models.swin2sr import Swin2SRModule
from utils.utils import Augmentations


def create_dataset(config,
                   datadir,
                   kwargs_dict=None):
    
    if kwargs_dict is None:
        kwargs_dict = {}
    
    torch.manual_seed(42)
    
    resize_to_shape = (1004, 1004)
    # Define your working directory and data directory
    work_dir = "."
    tensorboard_log_dir = os.path.join(work_dir, "tensorboard_logs")
    os.makedirs(tensorboard_log_dir, exist_ok=True) 
    augmentations = Augmentations() 
    dataset = BioSRDataLoader(root_dir=datadir, resize_to_shape = resize_to_shape, transform=None)
    
    # Define the split ratios
    train_ratio = 0.8
    val_ratio = 0.1
        
    # Calculate the number of samples for each set
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    # Perform the split
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])     
    

    # Create DataLoader for each split
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=15)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=15)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=15)       
    
    return dataset, train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def create_model_and_train(config, logger, train_loader, val_loader, logdir): 
    args = {
        "learning_rate": config.training.lr,
        "architecture": config.model.model_type,
        "dataset": "BioSRDataset",
        "epochs": config.training.num_epochs
    }
    
    # Initialize WandbLogger
    wandb_logger = WandbLogger(save_dir=logdir, project="SwinFormer")
    wandb_logger.experiment.config.update(config.to_dict())
    
    wandb.init(project="SwinFormer", config=args)
    model = Swin2SRModule(config)
    
    # Define the Trainer
    trainer = pl.Trainer(
        max_epochs=config.training.num_epochs,
        logger=wandb_logger,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        precision=16
    )
    
    # Train the model
    trainer.fit(model, train_loader, val_loader)
    
    # Save the trained model
    saving_dir = "/home/michele.prencipe/tesi/transformer/swin2sr/logdir"
    model_filename = f'swin2sr_epoch{trainer.current_epoch}_valloss{trainer.callback_metrics["val_loss"]:.4f}_{int(time.time())}.pth'
    model_filepath = os.path.join(saving_dir, model_filename)
    torch.save(model.state_dict(), model_filepath)
    
if __name__ == '__main__':
    logger = wandb.login()
    logdir = 'tesi/transformer/swin2sr/logdir'
    
    config = get_config()
    dataset, train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = create_dataset(config, datadir='/group/jug/ashesh/data/BioSR/')
    create_model_and_train(config=config, logger=logger, train_loader = train_loader, val_loader = val_loader, logdir = logdir)
