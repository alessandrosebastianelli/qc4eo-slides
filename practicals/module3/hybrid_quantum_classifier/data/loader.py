"""
Utilities for downloading and loading the EuroSAT dataset
"""

import os
import zipfile
import requests
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import DataLoader, random_split

from .dataset import EuroSATDataset


def download_eurosat(data_dir="EuroSAT"):
    """
    Download the EuroSAT dataset if not already present.
    
    The dataset contains RGB satellite images from Sentinel-2 satellite,
    organized into 10 land use classes.
    
    Args:
        data_dir: Directory where the dataset will be stored
    """
    if os.path.exists(data_dir):
        print(f"Dataset already available at: {data_dir}")
        return
    
    # EuroSAT dataset URL
    url = "http://madm.dfki.de/files/sentinel/EuroSAT.zip"
    zip_path = "EuroSAT.zip"
    
    print("Downloading EuroSAT dataset (~90MB)...")
    print(f"URL: {url}")
    
    # Download with progress bar
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(zip_path, 'wb') as f:
        downloaded = 0
        for data in response.iter_content(chunk_size=8192):
            f.write(data)
            downloaded += len(data)
            # Simple progress bar
            done = int(50 * downloaded / total_size)
            print(f"\r[{'=' * done}{' ' * (50 - done)}] "
                  f"{downloaded/1e6:.1f}/{total_size/1e6:.1f} MB", 
                  end='', flush=True)
    
    print("\n\nExtracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    # Clean up zip file
    os.remove(zip_path)
    print("Dataset downloaded and extracted successfully!")


def load_images(data_dir="EuroSAT/2750", max_per_class=300):
    """
    Load images from the EuroSAT dataset.
    
    The EuroSAT dataset structure:
    EuroSAT/2750/
        ├── AnnualCrop/
        ├── Forest/
        ├── HerbaceousVegetation/
        ├── Highway/
        ├── Industrial/
        ├── Pasture/
        ├── PermanentCrop/
        ├── Residential/
        ├── River/
        └── SeaLake/
    
    Args:
        data_dir: Path to the dataset directory
        max_per_class: Maximum number of images to load per class
                      (useful for quick experimentation)
    
    Returns:
        tuple: (images, labels, class_names)
            - images: numpy array of shape (N, H, W, C)
            - labels: numpy array of shape (N,) with string class names
            - class_names: list of unique class names
    """
    images = []
    labels = []
    class_names = []
    
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}\n"
            f"Please run download_eurosat() first."
        )
    
    # Iterate through class directories
    for class_dir in sorted(data_path.iterdir()):
        if class_dir.is_dir():
            class_name = class_dir.name
            class_names.append(class_name)
            
            # Load images from this class
            image_files = list(class_dir.glob("*.jpg"))[:max_per_class]
            
            print(f"Loading {len(image_files)} images from class: {class_name}")
            
            for img_path in image_files:
                # Load image as numpy array
                img = np.array(Image.open(img_path))
                images.append(img)
                labels.append(class_name)
    
    images = np.array(images)
    labels = np.array(labels)
    
    print(f"\nLoaded {len(images)} images from {len(class_names)} classes")
    print(f"Image shape: {images[0].shape}")
    print(f"Classes: {class_names}")
    
    return images, labels, class_names


def create_label_mapping(class_names):
    """
    Create a mapping from class names to numeric indices.
    
    Args:
        class_names: List of class names
        
    Returns:
        dict: Mapping from class name to index
    """
    label_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
    return label_to_idx


def labels_to_numeric(labels, label_to_idx):
    """
    Convert string labels to numeric indices.
    
    Args:
        labels: Array of string labels
        label_to_idx: Dictionary mapping labels to indices
        
    Returns:
        numpy.ndarray: Array of numeric labels
    """
    return np.array([label_to_idx[label] for label in labels])


def create_dataloaders(images, labels, class_names, config):
    """
    Create PyTorch DataLoaders for training, validation, and testing.
    
    This function:
    1. Converts string labels to numeric indices
    2. Creates the full dataset
    3. Splits into train/val/test sets
    4. Creates DataLoader objects for each split
    
    Args:
        images: NumPy array of images
        labels: NumPy array of string labels
        class_names: List of class names
        config: Configuration object with batch_size, train_ratio, etc.
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, label_to_idx)
    """
    # Create label mapping
    label_to_idx = create_label_mapping(class_names)
    print(f"\nLabel mapping: {label_to_idx}")
    
    # Convert labels to numeric
    labels_numeric = labels_to_numeric(labels, label_to_idx)
    
    # Create dataset
    transform = EuroSATDataset.get_default_transform(
        image_size=config.IMAGE_SIZE
    )
    full_dataset = EuroSATDataset(images, labels_numeric, transform=transform)
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(config.TRAIN_RATIO * total_size)
    val_size = int(config.VAL_RATIO * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        persistent_workers=True if config.NUM_WORKERS > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        persistent_workers=True if config.NUM_WORKERS > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        persistent_workers=True if config.NUM_WORKERS > 0 else False
    )
    
    print("DataLoaders created successfully!")
    
    return train_loader, val_loader, test_loader, label_to_idx
