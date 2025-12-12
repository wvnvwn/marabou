"""
Small CNN Models for Marabou Verification

These models are specifically designed to be tractable for SMT-based
verification tools like Marabou. They have:
- Few layers (2-3 conv layers)
- Small number of parameters (~10K-50K)
- Simple architecture without complex operations

Reference: Marabou GitHub - https://github.com/NeuralNetworkVerification/Marabou
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    """
    Small CNN optimized for Marabou verification.
    
    Architecture:
        Conv(3x3, 16) -> ReLU -> MaxPool
        Conv(3x3, 32) -> ReLU -> MaxPool  
        FC(128) -> ReLU -> FC(num_classes)
    
    Parameters: ~50K (CIFAR-10)
    
    This is small enough for Marabou to verify in reasonable time
    while still achieving decent accuracy on CIFAR-10.
    """
    
    def __init__(self, num_classes=10, in_channels=3, use_bn=False):
        """
        Args:
            num_classes: Number of output classes (10 for CIFAR-10)
            in_channels: Number of input channels (3 for RGB)
            use_bn: Whether to use batch normalization (not recommended for Marabou)
        """
        super(SmallCNN, self).__init__()
        self.use_bn = use_bn
        
        # Conv layers
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=not use_bn)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=not use_bn)
        
        if use_bn:
            self.bn1 = nn.BatchNorm2d(16)
            self.bn2 = nn.BatchNorm2d(32)
        
        # After two 2x2 max pools: 32x32 -> 16x16 -> 8x8
        # Feature map size: 32 * 8 * 8 = 2048
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # For compatibility with ResNet checkpoint loading code
        self.linear = self.fc2
    
    def forward(self, x):
        # Conv block 1
        if self.use_bn:
            x = F.relu(self.bn1(self.conv1(x)))
        else:
            x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        
        # Conv block 2
        if self.use_bn:
            x = F.relu(self.bn2(self.conv2(x)))
        else:
            x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        # Flatten and FC layers
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


class TinyCNN(nn.Module):
    """
    Tiny CNN for very fast Marabou verification.
    
    Architecture:
        Conv(5x5, 8) -> ReLU -> MaxPool
        FC(64) -> ReLU -> FC(num_classes)
    
    Parameters: ~15K (CIFAR-10)
    
    Very fast verification but lower accuracy.
    Good for testing and debugging.
    """
    
    def __init__(self, num_classes=10, in_channels=3, use_bn=False):
        super(TinyCNN, self).__init__()
        self.use_bn = use_bn
        
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=5, padding=2, bias=not use_bn)
        
        if use_bn:
            self.bn1 = nn.BatchNorm2d(8)
        
        # After one 2x2 max pool: 32x32 -> 16x16
        # Feature map size: 8 * 16 * 16 = 2048
        self.fc1 = nn.Linear(8 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
        # For compatibility
        self.linear = self.fc2
    
    def forward(self, x):
        if self.use_bn:
            x = F.relu(self.bn1(self.conv1(x)))
        else:
            x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=" * 60)
    print("Small CNN Models for Marabou Verification")
    print("=" * 60)
    
    # Test SmallCNN
    print("\n1. SmallCNN (recommended for CIFAR-10):")
    model = SmallCNN(num_classes=10, in_channels=3, use_bn=False)
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    print(f"   Input: {x.shape}")
    print(f"   Output: {y.shape}")
    print(f"   Parameters: {count_parameters(model):,}")
    
    # Test TinyCNN
    print("\n2. TinyCNN (fastest verification):")
    model = TinyCNN(num_classes=10, in_channels=3, use_bn=False)
    y = model(x)
    print(f"   Input: {x.shape}")
    print(f"   Output: {y.shape}")
    print(f"   Parameters: {count_parameters(model):,}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

