from swin2sr.configs.default_config import get_default_config
from swin2sr.core.data_type import DataType
from swin2sr.core.loss_type import LossType
from swin2sr.core.model_type import ModelType
from swin2sr.core.sampler_type import SamplerType


def get_config():
    config = get_default_config()
    data = config.data
    data.image_size = 256
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
    model.img_shape = (256,256)
    model.upscale = 1
    model.in_chans = 1
    

    training = config.training
    training.lr = 0.001
    training.num_epochs = 400
    # training.precision = 16
    return config