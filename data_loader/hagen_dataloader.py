import os
import numpy as np
import torch
from data_loader.read_mrc import read_mrc
from torch.utils.data import Dataset
from skimage.transform import resize
import matplotlib.pyplot as plt


class HagenDataloader(Dataset):    
    
    def hagendataloader(self, directory, mode='Train'):
        if mode == 'Train':
            train = np.load(os.path.join(directory, 'train_data.npy'))
            return train
        elif mode =='Val':
            val = np.load(os.path.join(directory, 'val_data.npy'))
            return val
        elif mode =='Test':
            test = np.load(os.path.join(directory, 'test_data.npy'))
            return test     
        
    
    """Dataset class to load images from MRC files in multiple folders."""
    def __init__(self, patch_size=64, transform=None, mode='Train'): #TODO
        """
        Args:
            root_dir (string): Root directory containing subdirectories of MRC files.
            transform (callable, optional): Optional transform to be applied on a sample.
            resize_to_shape: For development, we can downscale the data to fit in the meomory constraints
        """
        self.transform = transform  
        self.mode = mode  
        self.c1_data_hagen = self.hagendataloader(directory = '/group/jug/ashesh/TrainValTestSplit/hagen', mode = self.mode)[..., 0:1]
        self.c2_data_hagen = self.hagendataloader(directory = '/group/jug/ashesh/TrainValTestSplit/hagen', mode = self.mode)[..., 1:2]
        
        self.c1_data_hagen = np.transpose(self.c1_data_hagen, (1,2,0))
        self.c2_data_hagen = np.transpose(self.c2_data_hagen, (1,2,0))
        
        self.patch_size = patch_size      
  
        self.c1_min = np.min(self.c1_data_hagen) 
        self.c2_min = np.min(self.c2_data_hagen)
        self.c1_max = np.max(self.c1_data_hagen)
        self.c2_max = np.max(self.c2_data_hagen) 
        self.input_min = np.min(self.c1_data_hagen[:,:,:self.c1_data_hagen.shape[-1]]+self.c2_data_hagen[:, :, :self.c1_data_hagen.shape[-1]])
        self.input_max = np.max(self.c1_data_hagen[:,:,:self.c2_data_hagen.shape[-1]]+self.c2_data_hagen[:, :,:self.c2_data_hagen.shape[-1]])
        
        print("Norm Param: ", self.c1_min, self.c2_min, self.c1_max, self.c2_max, self.input_max, self.input_min)
        
        print(f"c1_data shape: {self.c1_data_hagen.shape}")
        print(f"c2_data shape: {self.c2_data_hagen.shape}")
        
    def __len__(self):
        # Use the first dimension to determine the number of images
        return min(self.c1_data_hagen.shape[-1], self.c2_data_hagen.shape[-1])

    def __getitem__(self, idx):
        n_idx, h, w = self.patch_location(idx)
        
        data_channel1 = self.c1_data_hagen[h:h+self.patch_size,w:w+self.patch_size, n_idx].astype(np.float32)
        data_channel2 = self.c2_data_hagen[h:h+self.patch_size,w:w+self.patch_size, n_idx].astype(np.float32)       

        sample1 = {'image': data_channel1}
        sample2 = {'image': data_channel2}      
        
        if self.transform:
            transformed = self.transform(image = sample1['image'], image0=sample2['image'])
            
            sample1['image'] = transformed['image']
            sample2['image'] = transformed['image0']
                           
        input_image = sample1['image'] + sample2['image']
        
        input_image = (input_image - self.input_min) / (self.input_max - self.input_min) 
        input_image = input_image.astype(np.float32)
        sample1['image'] = (sample1['image'] - self.c1_min) / (self.c1_max - self.c1_min)  # Min-Max Normalization
        sample2['image'] = (sample2['image'] - self.c2_min) / (self.c2_max - self.c2_min)  # Min-Max Normalization
        
        target = np.stack((sample1['image'], sample2['image']))        
        target = target.astype(np.float32)        
        return input_image, target
    
    def get_normalization_params(self):
        return self.c1_min, self.c1_max, self.c2_min, self.c2_max   
    
    def patch_location(self, index):
        # it just ignores the index and returns a random location
        n_idx = np.random.randint(0,len(self))
        h = np.random.randint(0, self.c1_data_hagen.shape[0]-self.patch_size) 
        w = np.random.randint(0, self.c1_data_hagen.shape[1]-self.patch_size)
        return (n_idx, h, w)
   
    
    

    