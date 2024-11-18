import os

from configs.default_config import get_default_config
from core.data_type import DataType
from core.loss_type import LossType
from core.model_type import ModelType
from core.sampler_type import SamplerType


def get_config():
    config = get_default_config()
    
    configs  = {
    "data_type": "biosr",    
    "data": {'noisy_data': True, 'poisson_factor': 0, 'gaussian_factor': 3400},
    "data_shape": (5,1004,1004),
    "learning_rate": 0.001,
    "upscale": 1,
    "in_chans": 1,
    "patch_size": 4,
    "img_size": (256, 256),
    "window_size": 16,
    "img_range": 1.0,
    "depths":  [6, 6, 6, 6], 
    "embed_dim": 60,
    "num_heads": [6, 6, 6, 6],
    "mlp_ratio": 2,
    "upsampler": "pixelshuffledirect",
    
    }
    data = config.data
    data.data_type = configs['data_type']
    data.noisy = configs['data']['noisy_data']
    data.poisson_factor = configs['data']['poisson_factor']
    data.gaussian_factor = configs['data']['gaussian_factor']
    data.data_shape = configs['data_shape']
    
    loss = config.loss
    loss.loss_type = LossType.MSE

    model = config.model
    model.model_type = ModelType.Swin2SR
    model.upsampler = configs['upsampler']
    model.upscale = configs['upscale']
    model.in_chans = configs['in_chans']
    model.patch_size = configs['patch_size']
    model.img_size = configs['img_size']
    model.window_size = configs['window_size']
    model.img_range = configs['img_range']
    model.depths = configs['depths']
    model.embed_dim = configs['embed_dim']
    model.num_heads = configs['num_heads']
    model.mlp_ratio = configs['mlp_ratio']

    training = config.training
    training.lr = configs["learning_rate"]
    training.precision = 16
     
    return config


def save_config_to_json(config, experiment_directory):
    """
    Saves a ConfigDict to a JSON file.
    
    Args:
        config (ConfigDict): The configuration object to save.
        experiment_directory (str): The directory where the config.json will be saved.
    """
    # Ensure the directory exists
    os.makedirs(experiment_directory, exist_ok=True)
    
    # Convert ConfigDict to JSON string
    json_str = config.to_json(indent=4)  # Use `indent=4` for pretty formatting
    
    # Save the JSON string to a file
    with open(os.path.join(experiment_directory, 'config.json'), 'w') as f:
        f.write(json_str)
