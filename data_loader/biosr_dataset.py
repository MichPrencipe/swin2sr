import os
import numpy as np
import torch
from data_loader.read_mrc import read_mrc
from torch.utils.data import Dataset
from skimage.transform import resize

def downscale(data, shape):
    """
    HxWxC -> H/2 x W/2 x C
    """
    dtype = data.dtype
    new_shape = (*shape, data.shape[-1])
    return resize(data*1.0, new_shape).astype(dtype)

class BioSRDataLoader(Dataset):
    
    """Dataset class to load images from MRC files in multiple folders."""
    def __init__(self, root_dir, patch_size=64, transform=None, resize_to_shape=None, noisy_data = False, noise_factor = 0.1):
        """
        Args:
            root_dir (string): Root directory containing subdirectories of MRC files.
            transform (callable, optional): Optional transform to be applied on a sample.
            resize_to_shape: For development, we can downscale the data to fit in the meomory constraints
        """
        self.root_dir = root_dir
        self.transform = transform
        self.c1_data = read_mrc(os.path.join(self.root_dir, 'ER/', 'GT_all.mrc'), filetype='image')[1]
        self.c2_data = read_mrc(os.path.join(self.root_dir, 'CCPs/', 'GT_all.mrc'), filetype='image')[1]
        self.patch_size = patch_size
        self.noisy_data = noisy_data
        self.noise_factor = noise_factor
        
        
        self.c1_min = np.min(self.c1_data) 
        self.c2_min = np.min(self.c2_data)
        self.c1_max = np.max(self.c1_data)
        self.c2_max = np.max(self.c2_data)
        print("Norm Param: ", self.c1_min, self.c2_min, self.c1_max, self.c2_max)
       
        # Ensure c1_data and c2_data are NumPy arrays
        if isinstance(self.c1_data, tuple):
            self.c1_data = np.array(self.c1_data)
        if isinstance(self.c2_data, tuple):
            self.c2_data = np.array(self.c2_data)

        # Debug print to check the shape of the data
        if resize_to_shape is not None:
            print(f"Resizing to shape {resize_to_shape}. MUST BE REMOVED IN PRODUCTION!")
            self.c1_data = downscale(self.c1_data, resize_to_shape)
            self.c2_data = downscale(self.c2_data, resize_to_shape)                
        

        print(f"c1_data shape: {self.c1_data.shape}")
        print(f"c2_data shape: {self.c2_data.shape}")
        
    def __len__(self):
        # Use the first dimension to determine the number of images
        return min(self.c1_data.shape[-1], self.c2_data.shape[-1])

    def __getitem__(self, idx):
        data_channel1 = self.c1_data[:, :, idx]
        data_channel2 = self.c2_data[:, :, idx]        

        # Convert data to float32 and normalize (example normalization)
        data_channel1 = data_channel1.astype(np.float32)
        data_channel2 = data_channel2.astype(np.float32)


        sample1 = {'image': data_channel1}
        sample2 = {'image': data_channel2}

        if self.transform:
            sample1 = self.transform(sample1)
            sample2 = self.transform(sample2)           
                           

        input_image = sample1['image'] + sample2['image']
        
        if self.noisy_data:         
            poisson_data = np.random.poisson(input_image / self.noise_factor) * self.noise_factor 
            gaussian_data = np.random.normal(0, np.std(poisson_data), (poisson_data.shape)) #change the noise_factor and the standard deviation
            input_image = poisson_data + gaussian_data 
        
        input_image = (input_image - np.min(input_image)) / (np.max(input_image) - np.min(input_image)) 
        input_image = input_image.astype(np.float32)
        sample1['image'] = (sample1['image'] - self.c1_min) / (self.c1_max - self.c1_min)  # Min-Max Normalization
        sample2['image'] = (sample2['image'] - self.c2_min) / (self.c2_max - self.c2_min)  # Min-Max Normalization
        
        target = np.stack((sample1['image'], sample2['image']))        
        target = target.astype(np.float32)        
        return input_image, target
    
    def get_normalization_params(self):
        return self.c1_min, self.c1_max, self.c2_min, self.c2_max   
   

    