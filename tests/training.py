import os
import torch
import time
from torch.utils.data import DataLoader, random_split


import pytorch_lightning as pl
import wandb
import torch.nn as nn
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from data_loader.biosr_dataset import BioSRDataLoader

from loss.charbonnier_loss import CharbonnierLoss
from models.model_utils import create_model
from configs.biosr_config import get_config

from models.network_swin2sr import Swin2SR

def create_dataset(config,
                   datadir,
                   kwargs_dict=None):
    
    if kwargs_dict is None:
        kwargs_dict = {}
    
    resize_to_shape = (256,256)
    # Define your working directory and data directory
    work_dir = "."
    tensorboard_log_dir = os.path.join(work_dir, "tensorboard_logs")
    os.makedirs(tensorboard_log_dir, exist_ok=True)  
    dataset = BioSRDataLoader(root_dir=datadir, resize_to_shape=resize_to_shape)
    
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
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader

def create_model_and_train(config, logger, train_loader, val_loader, logdir): 
    
    args={
      "learning_rate": config.training.lr,
      "architecture": config.model.model_type,
      "dataset": "BioSRDataset",
      "epochs": config.training.num_epochs
      }    
    
    logger = WandbLogger(save_dir=logdir,
                         project="SwinFormer")
    logger.experiment.config.update(config.to_dict())    
    
    wandb.init(project = "SwinFormer", config=args)    
    
    criterion = nn.MSELoss()
    
    num_epochs = 200

    depths = [3, 3]
    num_heads = [3, 3]

    model = Swin2SR(upscale=1, in_chans = 1, img_size=(256, 256),
                   window_size=4, img_range=1., depths=depths,
                   embed_dim=60, num_heads=num_heads, mlp_ratio=2, upsampler='pixelshuffledirect')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.cuda()   
        
    # Magic
    wandb.watch(model, log_freq=100)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # Training loop
        for inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()

            # Ensure inputs and targets have the correct dimensions
            if len(inputs.shape) == 3:  # If inputs are [batch_size, height, width]
                inputs = inputs.unsqueeze(1)  # Add channel dimension to make it [batch_size, 1, height, width]
            if len(targets.shape) == 3:  # If targets are [batch_size, height, width]
                targets = targets.unsqueeze(1)  # Add channel dimension to make it [batch_size, 1, height, width]
            
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs)  # Forward pass
            
            # Ensure outputs and targets have the same dimensions for loss computation
            if outputs.shape != targets.shape:
                print(f"Output shape: {outputs.shape}, Target shape: {targets.shape}")
                raise ValueError("Output and target shapes do not match!")
            
            loss = criterion(outputs, targets)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            torch.cuda.synchronize()
            running_loss += loss.item()  # Accumulate loss
            
        # Compute average loss for the epoch
        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
        wandb.log({"epoch":epoch+1, "loss": epoch_loss})
        
        # Validation loop (optional but recommended)
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.cuda(), targets.cuda()

                # Ensure inputs and targets have the correct dimensions
                if len(inputs.shape) == 3:  # If inputs are [batch_size, height, width]
                    inputs = inputs.unsqueeze(1)  # Add channel dimension to make it [batch_size, 1, height, width]
                if len(targets.shape) == 3:  # If targets are [batch_size, height, width]
                    targets = targets.unsqueeze(1)  # Add channel dimension to make it [batch_size, 1, height, width]

                outputs = model(inputs)  # Forward pass
                
                # Ensure outputs and targets have the same dimensions for loss computation
                if outputs.shape != targets.shape:
                    print(f"Output shape: {outputs.shape}, Target shape: {targets.shape}")
                    raise ValueError("Output and target shapes do not match!")
                
                loss = criterion(outputs, targets)  # Compute loss
                val_loss += loss.item()  # Accumulate loss
                
        # Compute average validation loss for the epoch
        val_loss = val_loss / len(val_loader)
        print(f'Validation Loss: {val_loss:.4f}')
        wandb.log({"val_loss": val_loss})
    
    saving_dir= "/home/michele.prencipe/tesi/transformer/swin2sr/logdir"
    # Save the trained model
    # Save the model with a unique filename
    model_filename = f'swin2sr_epoch{epoch+1}_loss{epoch_loss:.4f}_valloss{val_loss:.4f}_{int(time.time())}.pth'
    model_filepath = os.path.join(saving_dir, model_filename)
    torch.save(model.state_dict(), model_filepath)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    
    from configs.biosr_config import get_config
    
    logger = wandb.login()
    logdir = 'tesi/transformer/swin2sr/logdir'

    config = get_config()
    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = create_dataset(config, datadir='/group/jug/ashesh/data/BioSR/')    
    create_model_and_train(config = config, logger = logger, train_loader = train_loader, val_loader= val_loader, logdir = logdir)