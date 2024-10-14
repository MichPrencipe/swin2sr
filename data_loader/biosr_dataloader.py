import os
import numpy as np
import torch
from data_loader.read_mrc import read_mrc
from torch.utils.data import Dataset
from skimage.transform import resize
import matplotlib.pyplot as plt


class BioSRDataloader(Dataset):    
    
    def biosrdataloader(self, directory=None, mode='Train'):
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
    def __init__(self, patch_size=64, transform=None, noisy_data = False, noise_factor = 1000, gaus_factor = 2000, mode = 'Train'): #TODO
        """
        Args:
            root_dir (string): Root directory containing subdirectories of MRC files.
            transform (callable, optional): Optional transform to be applied on a sample.
            resize_to_shape: For development, we can downscale the data to fit in the meomory constraints
        """
        self.transform = transform     
        self.mode = mode   
        
        self.c1_data_biosr = self.biosrdataloader(directory = '/group/jug/ashesh/TrainValTestSplit/biosr', mode = 'Train')[..., 0:1]
        self.c2_data_biosr = self.biosrdataloader(directory = '/group/jug/ashesh/TrainValTestSplit/biosr', mode = 'Train')[..., 1:2]
        
        self.c1_data_biosr = np.squeeze(self.c1_data_biosr, axis = -1)
        self.c2_data_biosr = np.squeeze(self.c2_data_biosr, axis = -1)

        self.c1_data_biosr = np.transpose(self.c1_data_biosr, (1,2,0))
        self.c2_data_biosr = np.transpose(self.c2_data_biosr, (1,2,0))
        
        self.patch_size = patch_size
        self.noisy_data = noisy_data
        self.noise_factor = noise_factor   
        self.gaus_factor = gaus_factor           
             
        
        if self.noisy_data: 
            self.poisson_noise_channel_1 = np.random.poisson(self.c1_data_biosr / self.noise_factor) * self.noise_factor
            self.gaussian_noise_channel_1= np.random.normal(0,self.gaus_factor, (self.poisson_noise_channel_1.shape))
            self.poisson_noise_channel_2= np.random.poisson(self.c2_data_biosr / self.noise_factor) * self.noise_factor
            self.gaussian_noise_channel_2 = np.random.normal(0,self.gaus_factor, (self.poisson_noise_channel_2.shape))
            self.c1_data_noisy = self.poisson_noise_channel_1 + self.gaussian_noise_channel_1
            self.c2_data_noisy = self.poisson_noise_channel_2 + self.gaussian_noise_channel_2    
        if noisy_data:            
            self.c1_min = np.min(self.c1_data_noisy) 
            self.c2_min = np.min(self.c2_data_noisy)
            self.c1_max = np.max(self.c1_data_noisy)
            self.c2_max = np.max(self.c2_data_noisy)
        else:
            self.c1_min = np.min(self.c1_data_biosr) 
            self.c2_min = np.min(self.c2_data_biosr)
            self.c1_max = np.max(self.c1_data_biosr)
            self.c2_max = np.max(self.c2_data_biosr) 
        self.input_min = np.min(self.c1_data_biosr[:,:,:self.c1_data_biosr.shape[-1]]+self.c2_data_biosr[:, :, :self.c1_data_biosr.shape[-1]])
        self.input_max = np.max(self.c1_data_biosr[:,:,:self.c2_data_biosr.shape[-1]]+self.c2_data_biosr[:, :,:self.c2_data_biosr.shape[-1]]) #TODO da cambiare trovare un modeo per trovare la lunghezza
        
        print("Norm Param: ", self.c1_min, self.c2_min, self.c1_max, self.c2_max,self.input_max, self.input_min)
        
        print(f"c1_data shape: {self.c1_data_biosr.shape}")
        print(f"c2_data shape: {self.c2_data_biosr.shape}")
        
    def __len__(self):
        # Use the first dimension to determine the number of images
        return min(self.c1_data_biosr.shape[-1], self.c2_data_biosr.shape[-1])

    def __getitem__(self, idx):
        n_idx, h, w = self.patch_location(idx)
        
        data_channel1 = self.c1_data_biosr[h:h+self.patch_size,w:w+self.patch_size, n_idx].astype(np.float32)
        data_channel2 = self.c2_data_biosr[h:h+self.patch_size,w:w+self.patch_size, n_idx].astype(np.float32) 
        
        data_channel1_noisy = self.c1_data_noisy[h:h+self.patch_size,w:w+self.patch_size, n_idx].astype(np.float32)
        data_channel2_noisy = self.c2_data_noisy[h:h+self.patch_size,w:w+self.patch_size, n_idx].astype(np.float32)

        sample1 = {'image': data_channel1}
        sample2 = {'image': data_channel2}
        noisy_sample_1 = {'image': data_channel1_noisy}
        noisy_sample_2 = {'image': data_channel2_noisy}        
        
        if self.transform:
            transformed = self.transform(image = sample1['image'], image0=sample2['image'], noisy_image_1=noisy_sample_1['image'], noisy_image_2=noisy_sample_2['image'])
            
            sample1['image'] = transformed['image']
            sample2['image'] = transformed['image0']
            noisy_sample_1['image'] = transformed['noisy_image_1']
            noisy_sample_2['image'] = transformed['noisy_image_2'] 
                           
        if self.noisy_data:
            input_image = noisy_sample_1['image'] + noisy_sample_2['image']
        else:
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
        h = np.random.randint(0, self.c1_data_biosr.shape[0]-self.patch_size) 
        w = np.random.randint(0, self.c1_data_biosr.shape[1]-self.patch_size)
        return (n_idx, h, w)
   
    
    

    