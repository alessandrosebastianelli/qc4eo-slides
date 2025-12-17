"""
EuroSAT Dataset class for satellite image classification
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class EuroSATDataset(Dataset):
    """
    PyTorch Dataset for EuroSAT satellite imagery.
    
    This dataset loads satellite images directly without dimensionality reduction,
    allowing the CNN to extract features from the raw images.
    
    Args:
        images (numpy.ndarray): Array of images with shape (N, H, W, C)
        labels (numpy.ndarray): Array of numeric labels with shape (N,)
        transform (callable, optional): Transform to apply to images
    """
    
    def __init__(self, images, labels, transform=None):
        """
        Initialize the dataset.
        
        Args:
            images: NumPy array of images
            labels: NumPy array of corresponding labels (numeric)
            transform: Optional torchvision transforms
        """
        self.images = images
        self.labels = labels
        self.transform = transform
        
        # Default transform if none provided
        if self.transform is None:
            self.transform = self.get_default_transform()
    
    def __len__(self):
        """Return the total number of samples"""
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            tuple: (image_tensor, label) where image_tensor is a torch.Tensor
        """
        # Get image and label
        img = self.images[idx]
        label = self.labels[idx]
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        
        return img, int(label)
    
    @staticmethod
    def get_default_transform(image_size=(64, 64)):
        """
        Get default image transformations.
        
        This includes:
        1. Converting numpy array to PIL Image
        2. Resizing to target size
        3. Converting to tensor
        4. Normalizing with ImageNet statistics
        
        Args:
            image_size: Target image size (height, width)
            
        Returns:
            transforms.Compose: Composition of transforms
        """
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            # ImageNet normalization (standard for pre-trained models)
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def get_class_distribution(self):
        """
        Calculate the distribution of classes in the dataset.
        
        Returns:
            dict: Mapping from label to count
        """
        unique, counts = torch.unique(
            torch.tensor(self.labels), 
            return_counts=True
        )
        return dict(zip(unique.tolist(), counts.tolist()))
    
    def __repr__(self):
        """String representation of the dataset"""
        return (f"EuroSATDataset(samples={len(self)}, "
                f"image_shape={self.images[0].shape}, "
                f"num_classes={len(set(self.labels))})")
