import glob
import os
import pickle

import pytorch_lightning as pl
import torch
import torch.nn as nn

from models.network_swin2sr import Swin2SR
from core.model_type import ModelType

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
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        if len(inputs.shape) == 3:
            inputs = inputs.unsqueeze(1)
        if len(targets.shape) == 3:
            targets = targets.unsqueeze(1)

        outputs = self(inputs)
        val_loss = self.criterion(outputs, targets)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    

