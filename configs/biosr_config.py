from configs.default_config import get_default_config
from core.data_type import DataType
from core.loss_type import LossType
from core.model_type import ModelType
from core.sampler_type import SamplerType


def get_config():
    config = get_default_config()
    data = config.data
    data.image_size = 1004
    data.data_type = DataType.BioSR_MRC
    # data.channel_1 = 0
    # data.channel_2 = 1
    data.ch1_fname = 'ER/GT_all.mrc'
    data.ch2_fname = 'CCPs/GT_all.mrc'
    data.num_channels = 2
   

    loss = config.loss
    loss.loss_type = LossType.CharbonnierLoss    

    model = config.model
    model.model_type = ModelType.Swin2SR
    model.img_shape = (1004,1004)
    model.upscale = 1
    model.in_chans = 1
    

    training = config.training
    training.lr = 0.001
    training.num_epochs = 400
    # training.precision = 16
    return config