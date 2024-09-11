import glob
import os
import pickle
import numpy as np

import pytorch_lightning as pl
import torch
import torch.nn as nn

from models.network_swin2sr import Swin2SR
from core.model_type import ModelType
import torchmetrics
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

class Swin2SRModule(pl.LightningModule):
    def __init__(self, config):
        super(Swin2SRModule, self).__init__()
        self.model = Swin2SR(
            upscale=1, in_chans=1, img_size=(256, 256),
            window_size=4, img_range=1., depths=[3, 3],
            embed_dim=60, num_heads=[3, 3], mlp_ratio=2,
            upsampler='pixelshuffledirect'
        )
        self.criterion = nn.MSELoss()
        self.learning_rate = config.training.lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        if len(inputs.shape) == 3:
            inputs = inputs.unsqueeze(1)
        if len(targets.shape) == 3:
            targets = targets.unsqueeze(1)

        outputs = self(inputs)   
        loss = self.criterion(outputs, targets)
        
        outputs = outputs.cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()
        
        # Calculate PSNR and SSIM
        psnr_value1 = psnr(outputs[...,0], targets[...,0], data_range=1.0)
        psnr_value2 = psnr(outputs[...,1], targets[...,1], data_range=1.0)  
        
        self.log("loss", loss, on_step=False, on_epoch=True)
        self.log('train_psnr channel 1', psnr_value1, prog_bar=True, logger=True)
        self.log('train_psnr channel 2', psnr_value2, prog_bar=True, logger=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        if len(inputs.shape) == 3:
            inputs = inputs.unsqueeze(1)
        if len(targets.shape) == 3:
            targets = targets.unsqueeze(1)

        outputs = self(inputs)
        val_loss = self.criterion(outputs, targets)       
        
        outputs = outputs.cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()
        
        # Calculate PSNR and SSIM
        psnr_value1 = psnr(outputs[...,0], targets[...,0], data_range=1.0)
        psnr_value2 = psnr(outputs[...,1], targets[...,1], data_range=1.0)   
        
        self.log("val_loss", val_loss, on_step=False, on_epoch=True)
        self.log('val_psnr channel 1', psnr_value1, prog_bar=True, logger=True)
        self.log('val_psnr channel 2', psnr_value2, prog_bar=True, logger=True)
        
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    

