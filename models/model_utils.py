import glob
import os
import pickle

import pytorch_lightning as pl
import torch
import torch.nn as nn

from models.network_swin2sr import Swin2SR
from core.model_type import ModelType

def create_model(config):
    if config.model.modeltype == ModelType.Swin2SR:
        depths = [3, 3]
        num_heads = [3, 3]

        model = Swin2SR(upscale=1, in_chans = 1, img_size=(256, 256),
                   window_size=4, img_range=1., depths=depths,
                   embed_dim=60, num_heads=num_heads, mlp_ratio=2, upsampler='pixelshuffledirect')
        print("Model Upload correctly")
        print(model)
    else:
        raise Exception('Invalid model type:', config.model.model_type)
    return model
