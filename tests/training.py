import pytorch_lightning as pl
import os
import torch
import torch.nn as nn
import wandb
import yaml
import socket

from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from models.network_swin2sr import Swin2SR
from torch.utils.data import random_split
from data_loader.biosr_dataset import BioSRDataLoader
from configs.biosr_config import get_config
from models.swin2sr import Swin2SRModule
from utils.utils import Augmentations


def create_dataset(config, datadir, kwargs_dict=None, noisy_data = False, noisy_factor = 0.1):
    if kwargs_dict is None:
        kwargs_dict = {}
        
    resize_to_shape = (768, 768)
    
    augmentations = Augmentations() 
    dataset = BioSRDataLoader(root_dir=datadir, resize_to_shape=resize_to_shape, transform=augmentations, noisy_data=noisy_data, noise_factor=noisy_factor)
    
    train_ratio, val_ratio = 0.8, 0.1
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    torch.manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=15)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=15)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=15)
    
    return dataset, train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def create_model_and_train(config, logger, train_loader, val_loader, logdir): 
    args = {
        "learning_rate": config.training.lr,
        "architecture": config.model.model_type,
        "dataset": "BioSRDataset",
        "epochs": config.training.num_epochs
    }
    config_str = f"LR: {args['learning_rate']}, Epochs: {args['epochs']}, Augmentations: True, Noisy_data: true" 
    
    
    print(f"Learning rate: {args['learning_rate']}")
    
    # Get node or hostname for custom run name, #TODO changethe name
    node_name = os.environ.get('SLURMD_NODENAME', socket.gethostname())  
    
    # Initialize WandbLogger with a custom run name
    wandb_logger = WandbLogger(save_dir=logdir, project="SwinTransformer", name=f"{node_name}" + config_str)
    
    wandb_logger.experiment.config.update(config.to_dict())   
    model = Swin2SRModule(config)
    
    # Define the Trainer
    trainer = pl.Trainer(
        max_epochs=config.training.num_epochs,
        logger=wandb_logger,
        log_every_n_steps=22,
        check_val_every_n_epoch=1,
        precision=16,
        enable_progress_bar=True
    )
    
    # Train the model
    trainer.fit(model, train_loader, val_loader)
    
    # Log metrics and model filename
    psnr1 = trainer.callback_metrics.get("val_psnr channel 1", None)
    psnr2 = trainer.callback_metrics.get("val_psnr channel 2", None)
    val_loss = trainer.callback_metrics.get("val_loss", None)
    
    run_id = wandb_logger.experiment.id
    
    wandb_logger.experiment.summary["Performance_Section"] = f"""
    ### Run ID: {run_id}
    **Configuration:** {config_str}

    #### Performance Metrics:
    - **PSNR channel 1**: {psnr1}
    - **PSNR channel 2**: {psnr2}
    """
    
    model_filename = f'swin2sr_epoch{trainer.current_epoch}_valloss{val_loss:.4f}.pth' if val_loss is not None else 'model.pth'
    wandb_logger.experiment.summary['model_weights_filename'] = model_filename
    
    # Save model
    saving_dir = "/home/michele.prencipe/tesi/transformer/swin2sr/logdir"
    model_filepath = os.path.join(saving_dir, model_filename)
    torch.save(model.state_dict(), model_filepath)
    
    # Save wandb summary to a file
    summary_dict = dict(wandb_logger.experiment.summary)
    with open(os.path.join(logdir, "wandb_summary.yaml"), "w") as f:
        yaml.dump(summary_dict, f)
    
    wandb.finish()


if __name__ == '__main__':
    logdir = 'tesi/transformer/swin2sr/logdir'
    wandb.login()
    config = get_config()
    
    dataset, train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = create_dataset(
        config=config, datadir='/group/jug/ashesh/data/BioSR/'
    )
    create_model_and_train(config=config, logger=wandb, train_loader=train_loader, val_loader=val_loader, logdir=logdir)
