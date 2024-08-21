import os
import numpy as np
import torch
from data_loader.read_mrc import read_mrc
from torch.utils.data import Dataset
from skimage.transform import resize

def downscale(data, shape):
    """
    HXWXC -> H/2 x W/2 x C
    """
    new_shape = (*shape, data.shape[-1])
    return resize(data, new_shape)

class BioSRDataset(Dataset):
    
    """Dataset class to load images from MRC files in multiple folders."""
    def __init__(self, root_dir, patch_size=64, transform=None, resize_to_shape=None):
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
        data_channel1 = (data_channel1 - np.min(data_channel1)) / (np.max(data_channel1) - np.min(data_channel1))  # Min-Max Normalization

        data_channel2 = data_channel2.astype(np.float32)
        data_channel2 = (data_channel2 - np.min(data_channel2)) / (np.max(data_channel2) - np.min(data_channel2))  # Min-Max Normalization

        sample1 = {'image': data_channel1}
        sample2 = {'image': data_channel2}

        if self.transform:
            sample1 = self.transform(sample1)
            sample2 = self.transform(sample2)

        input_image = sample1['image'] + sample2['image']
        target = np.stack((sample1['image'], sample2['image']))
        
        return input_image, target
   

    