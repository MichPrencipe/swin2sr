from sklearn.model_selection import ParameterSampler, ParameterGrid
import numpy as np
from tests.training import Swin2SRModule

param_dist = {
    'learning_rate': [0.0001, 0.0005, 0.001],
    'batch_size': [1, 2, 3],
    'num_heads':[[6, 6, 6, 6], [3 , 3]],
    'depths' : [[6, 6, 6, 6], [3, 3]],
    'gaussian_factor': [1000, 2000, 3000],
    'noise_factor': [1000, 1500, 800]
}

random_search = ParameterSampler(param_dist, n_iter=10, random_state=42)
grid = ParameterGrid(param_dist)

for params in random_search:
    model = Swin2SRModule(*param_dist, )  
    #train with the params

for params in grid:
    model = Swin2SRModule(...)  # Initialize your model
    #train the model with the params