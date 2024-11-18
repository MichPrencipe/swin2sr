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
from core.psnr import PSNR
from utils.utils import set_global_seed
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
set_global_seed(42)
class Swin2SRModule(pl.LightningModule):
    def __init__(self, config):
        super(Swin2SRModule, self).__init__()        
        # self.model = Swin2SR(
        #     upscale=1, in_chans=1, img_size=(256, 256),
        #     window_size=16, img_range=1., depths=[3, 3],
        #     embed_dim=60, num_heads=[3, 3], mlp_ratio=2,
        #     upsampler='pixelshuffledirect'
        # )
        
        self.model = Swin2SR(
            upscale=config['model']['upscale'], 
            in_chans=config['model']['in_chans'],
            img_size=config['model']['img_size'],
            window_size=config['model']['window_size'], 
            img_range=config['model']['img_range'], 
            depths=config['model']['depths'],
            embed_dim=config['model']['embed_dim'], 
            num_heads=config['model']['num_heads'],
            mlp_ratio=config['model']['mlp_ratio'],
            upsampler=config['model']['upsampler'],
            patch_size=config['model']['patch_size']

        )
        
        print(config)
        
        # config_dict = {'upscale': 1,
        #                'in_chans': 1,
        #                'img_size': (256, 256),
        #                'window_size': 16, 
        #                'img_range': 1.0, 
        #                'depths': [6, 4],
        #                'embed_dim': 96, 
        #                'num_heads': [8, 8],
        #                'mlp_ratio': 4.0,
        #                'upsampler': 'pixelshuffledirect'}        
        
        
        # self.model = Swin2SR(**config_dict)
        self.criterion = nn.MSELoss()
        self.learning_rate = config['training']['lr']

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        if len(inputs.shape) == 3:
            inputs = inputs.unsqueeze(1)
        if len(targets.shape) == 3:
            targets = targets.unsqueeze(1)        
        
        outputs = self.forward(inputs)   
        loss = self.criterion(outputs, targets)
        outputs = outputs.cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()     
        psnr_arr = { 0: [] , 1: []}
        for ch_idx in range(outputs.shape[1]):
            if ch_idx == 0:
                data_range = targets[:, ch_idx].max() -  targets[:,ch_idx].min() 
            else:
                data_range = targets[:, ch_idx].max() -  targets[:, ch_idx].min()
            psnr_arr[ch_idx].append(PSNR(targets[:, ch_idx], outputs[:, ch_idx], range_= data_range))  
        
        self.log("loss", loss, on_step=True, on_epoch=True)
        self.log('train_psnr channel 1', np.mean(psnr_arr[0]), prog_bar=False, logger=True)
        self.log('train_psnr channel 2', np.mean(psnr_arr[1]), prog_bar=False, logger=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        if len(inputs.shape) == 3:
            inputs = inputs.unsqueeze(1)
        if len(targets.shape) == 3:
            targets = targets.unsqueeze(1)

        outputs = self.forward(inputs)
        val_loss = self.criterion(outputs, targets)
        outputs = outputs.cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()
        
        psnr_arr = { 0: [] , 1: []}
        for ch_idx in range(outputs.shape[1]):
            if ch_idx == 0:
                data_range = targets[:, ch_idx].max() -  targets[:,ch_idx].min() 
            else:
                data_range = targets[:, ch_idx].max() -  targets[:, ch_idx].min()
            psnr_arr[ch_idx].append(PSNR(targets[:, ch_idx], outputs[:, ch_idx], range_= data_range))        
        
                
        self.log("val_loss", val_loss, on_step=True, on_epoch=True)
        self.log('val_psnr channel 1', np.mean(psnr_arr[0]), prog_bar=False, logger=True)
        self.log('val_psnr channel 2', np.mean(psnr_arr[1]), prog_bar=False, logger=True)
        
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adamax(self.parameters(), lr=self.learning_rate)
         
        scheduler = {
                        'scheduler': StepLR(optimizer, step_size=50, gamma=0.1),
                        'interval': 'epoch',
                        'frequency': 1
                    }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


    


