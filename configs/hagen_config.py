from configs.default_config import get_default_config
from core.data_type import DataType
from core.loss_type import LossType
from core.model_type import ModelType
from core.sampler_type import SamplerType


def get_config():
    config = get_default_config()
    data = config.data
    data.data_type = 'hagen'
    
    loss = config.loss
    loss.loss_type = LossType.MSE

    model = config.model
    model.model_type = ModelType.Swin2SR

    training = config.training
    training.lr = 0.001
    training.precision = 16
    
     
    return config