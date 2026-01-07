"""
CNN Feature Extractor for satellite images

This CNN replaces traditional dimensionality reduction (like PCA) by learning
to extract relevant features from satellite images through training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNFeatureExtractor(nn.Module):
    """
    Convolutional Neural Network for extracting features from satellite images.
    
    Architecture:
    - 3 Convolutional blocks (Conv → BatchNorm → ReLU → MaxPool)
    - Global Average Pooling
    - Fully connected layer
    - Dropout for regularization
    
    This network learns to extract spatial features from images that are
    relevant for classification, replacing hand-crafted feature extraction
    methods like PCA.
    
    Args:
        output_dim: Dimension of the output feature vector
    """
    
    def __init__(self, output_dim=256):
        """
        Initialize the CNN feature extractor.
        
        Args:
            output_dim: Size of the output feature vector (default: 256)
        """
        super(CNNFeatureExtractor, self).__init__()
        
        self.output_dim = output_dim
        
        # First convolutional block
        # Input: (batch, 3, 64, 64) → RGB images
        # Output: (batch, 32, 32, 32)
        self.conv1 = nn.Conv2d(
            in_channels=3,      # RGB input
            out_channels=32,    # 32 feature maps
            kernel_size=3,      # 3x3 convolution
            padding=1           # Same padding
        )
        self.bn1 = nn.BatchNorm2d(32)  # Batch normalization for stability
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample by 2
        
        # Second convolutional block
        # Input: (batch, 32, 32, 32)
        # Output: (batch, 64, 16, 16)
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Third convolutional block
        # Input: (batch, 64, 16, 16)
        # Output: (batch, 128, 8, 8)
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Global Average Pooling
        # Reduces (batch, 128, 8, 8) → (batch, 128, 1, 1)
        # This is more robust than flattening
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layer
        # Maps 128 features to desired output dimension
        self.fc = nn.Linear(128, output_dim)
        
        # Dropout for regularization (prevents overfitting)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        """
        Forward pass through the CNN.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Tensor of shape (batch_size, output_dim) with extracted features
        """
        # First block: Conv → BatchNorm → ReLU → MaxPool
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Second block: Conv → BatchNorm → ReLU → MaxPool
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Third block: Conv → BatchNorm → ReLU → MaxPool
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Global pooling: (batch, 128, 8, 8) → (batch, 128, 1, 1)
        x = self.global_pool(x)
        
        # Flatten: (batch, 128, 1, 1) → (batch, 128)
        x = x.view(x.size(0), -1)
        
        # Fully connected: (batch, 128) → (batch, output_dim)
        x = self.fc(x)
        
        # Apply dropout and ReLU activation
        x = F.relu(x)
        x = self.dropout(x)
        
        return x
    
    def get_num_parameters(self):
        """
        Calculate the total number of parameters in the CNN.
        
        Returns:
            int: Total number of parameters
        """
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_parameters(self):
        """
        Calculate the number of trainable parameters.
        
        Returns:
            int: Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def print_architecture(self):
        """Print the CNN architecture in a readable format"""
        print("\n" + "=" * 60)
        print("CNN FEATURE EXTRACTOR ARCHITECTURE")
        print("=" * 60)
        print(f"Output dimension: {self.output_dim}")
        print("\nLayers:")
        print("  1. Conv2d(3 → 32, 3x3) + BatchNorm + ReLU + MaxPool")
        print("  2. Conv2d(32 → 64, 3x3) + BatchNorm + ReLU + MaxPool")
        print("  3. Conv2d(64 → 128, 3x3) + BatchNorm + ReLU + MaxPool")
        print("  4. Global Average Pooling")
        print(f"  5. Fully Connected(128 → {self.output_dim}) + ReLU + Dropout(0.3)")
        print("\nParameters:")
        print(f"  Total: {self.get_num_parameters():,}")
        print(f"  Trainable: {self.get_num_trainable_parameters():,}")
        print("=" * 60 + "\n")


def test_cnn_extractor():
    """
    Test function to verify CNN works correctly.
    Creates a dummy input and passes it through the network.
    """
    print("Testing CNN Feature Extractor...")
    
    # Create a dummy batch of images (4 images, 3 channels, 64x64)
    dummy_input = torch.randn(4, 3, 64, 64)
    
    # Create model
    model = CNNFeatureExtractor(output_dim=256)
    model.print_architecture()
    
    # Forward pass
    output = model(dummy_input)
    
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("\n✓ CNN Feature Extractor test passed!")
    
    return model


if __name__ == "__main__":
    # Run test if this file is executed directly
    test_cnn_extractor()
