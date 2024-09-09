from torchvision import transforms
import numpy as np

from torchvision import transforms


class Augmentations:
    def __init__(self):
        self.transforms = transforms.Compose([
            transforms.Lambda(self.random_flip),
            transforms.Lambda(self.random_rotation)
        ])

    def random_flip(self, sample):
        img = sample['image']  # Access the image data from the dictionary
        if img.ndim == 3:  # Handle 3D images (e.g., color images)
            img = np.flip(img, axis=1)  # Flip horizontally
        elif img.ndim == 2:  # Handle 2D images (e.g., grayscale images)
            img = np.flip(img, axis=1)  # Flip horizontally
        else:
            raise ValueError(f"Unexpected number of dimensions: {img.ndim}")

        sample['image'] = img
        return sample

    def random_rotation(self, sample):
        k = np.random.randint(0, 4)  # Randomly choose rotation angle: 90, 180, or 270 degrees
        img = sample['image']
        
        if img.ndim == 3:  # Handle 3D images (e.g., color images)
            img = np.rot90(img, k, axes=(0, 1))
        elif img.ndim == 2:  # Handle 2D images (e.g., grayscale images)
            img = np.rot90(img, k)
        else:
            raise ValueError(f"Unexpected number of dimensions: {img.ndim}")

        sample['image'] = img
        return sample

    def __call__(self, sample):
        return self.transforms(sample)



