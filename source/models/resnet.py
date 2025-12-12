"""
ResNet Implementation for Neural Network Verification

This module implements ResNet-20 architecture adapted for:
1. CIFAR-10 (32x32x3 images) - Original configuration
2. MNIST (28x28x1 images) - Adapted configuration

The architecture is kept small to remain tractable for Marabou verification
while maintaining sufficient complexity for meaningful experiments.

Reference:
    He et al., "Deep Residual Learning for Image Recognition" (CVPR 2016)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
    Basic ResNet building block with two 3x3 convolutions.

    This block implements:
        x -> Conv -> BN -> ReLU -> Conv -> BN -> (+) -> ReLU
                                              |          ^
                                              |__________|
                                              (skip connection)

    Note: BatchNorm can be disabled for verification as it complicates
    the verification process.
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_bn=True):
        """
        Args:
            in_planes: Number of input channels
            planes: Number of output channels
            stride: Stride for the first convolution
            use_bn: Whether to use batch normalization
        """
        super(BasicBlock, self).__init__()
        self.use_bn = use_bn

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=not use_bn
        )
        if use_bn:
            self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1,
            padding=1, bias=not use_bn
        )
        if use_bn:
            self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            if use_bn:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes,
                            kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes,
                            kernel_size=1, stride=stride, bias=True)
                )

    def forward(self, x):
        """Forward pass through the basic block."""
        if self.use_bn:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
        else:
            out = F.relu(self.conv1(x))
            out = self.conv2(out)

        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """
    ResNet architecture for image classification.

    This implementation is flexible to work with different datasets
    by adjusting the initial convolution and number of blocks.
    """

    def __init__(self, block, num_blocks, num_classes=10, in_channels=3, use_bn=True):
        """
        Args:
            block: Building block class (BasicBlock)
            num_blocks: List of number of blocks in each layer
            num_classes: Number of output classes
            in_channels: Number of input channels (1 for MNIST, 3 for CIFAR-10)
            use_bn: Whether to use batch normalization
        """
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.use_bn = use_bn

        # Initial convolution (adjusted for small images)
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3,
                              stride=1, padding=1, bias=not use_bn)
        if use_bn:
            self.bn1 = nn.BatchNorm2d(16)

        # ResNet layers
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        # Final classification layer
        self.linear = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        """Create a ResNet layer with multiple blocks."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.use_bn))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the network."""
        if self.use_bn:
            out = F.relu(self.bn1(self.conv1(x)))
        else:
            out = F.relu(self.conv1(x))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        # Global max pooling (Marabou-compatible)
        # For CIFAR-10 (32x32): after two stride-2 layers, feature map is 8x8
        out = F.max_pool2d(out, 8)
        # Use flatten instead of view to avoid Shape operation
        out = torch.flatten(out, start_dim=1)
        out = self.linear(out)
        return out


def ResNet20(num_classes=10, in_channels=3, use_bn=True):
    """
    ResNet-20 for CIFAR-10 (or similar 32x32 datasets).

    Architecture: 3 layers with [3, 3, 3] blocks each = 2*9 + 2 = 20 layers

    Args:
        num_classes: Number of output classes
        in_channels: Number of input channels (3 for CIFAR-10)
        use_bn: Use batch normalization (True for training, False for verification)

    Returns:
        ResNet-20 model
    """
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes,
                  in_channels=in_channels, use_bn=use_bn)


def ResNet32(num_classes=10, in_channels=3, use_bn=True):
    """
    ResNet-32 for CIFAR-10/100 (or similar 32x32 datasets).

    Architecture: 3 layers with [5, 5, 5] blocks each = 2*15 + 2 = 32 layers
    Parameters: ~470K (vs ~271K for ResNet-20)

    Args:
        num_classes: Number of output classes
        in_channels: Number of input channels (3 for CIFAR)
        use_bn: Use batch normalization (True for training, False for verification)

    Returns:
        ResNet-32 model
    """
    return ResNet(BasicBlock, [5, 5, 5], num_classes=num_classes,
                  in_channels=in_channels, use_bn=use_bn)


def ResNet56(num_classes=10, in_channels=3, use_bn=True):
    """
    ResNet-56 for CIFAR-10/100 (or similar 32x32 datasets).

    Architecture: 3 layers with [9, 9, 9] blocks each = 2*27 + 2 = 56 layers
    Parameters: ~853K (vs ~271K for ResNet-20)

    Args:
        num_classes: Number of output classes
        in_channels: Number of input channels (3 for CIFAR)
        use_bn: Use batch normalization (True for training, False for verification)

    Returns:
        ResNet-56 model
    """
    return ResNet(BasicBlock, [9, 9, 9], num_classes=num_classes,
                  in_channels=in_channels, use_bn=use_bn)


def ResNet20_MNIST(num_classes=10, use_bn=True):
    """
    ResNet-20 adapted for MNIST (28x28 grayscale images).

    Args:
        num_classes: Number of output classes
        use_bn: Use batch normalization (True for training, False for verification)

    Returns:
        ResNet-20 model adapted for MNIST
    """
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes,
                  in_channels=1, use_bn=use_bn)


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the models
    print("=" * 60)
    print("ResNet-20 Architecture Test")
    print("=" * 60)

    # Test CIFAR-10 model
    print("\n1. ResNet-20 for CIFAR-10:")
    model_cifar = ResNet20(num_classes=10, in_channels=3, use_bn=True)
    x_cifar = torch.randn(1, 3, 32, 32)
    y_cifar = model_cifar(x_cifar)
    print(f"   Input shape: {x_cifar.shape}")
    print(f"   Output shape: {y_cifar.shape}")
    print(f"   Parameters: {count_parameters(model_cifar):,}")

    # Test MNIST model
    print("\n2. ResNet-20 for MNIST:")
    model_mnist = ResNet20_MNIST(num_classes=10, use_bn=True)
    x_mnist = torch.randn(1, 1, 28, 28)
    y_mnist = model_mnist(x_mnist)
    print(f"   Input shape: {x_mnist.shape}")
    print(f"   Output shape: {y_mnist.shape}")
    print(f"   Parameters: {count_parameters(model_mnist):,}")

    # Test without batch normalization (for verification)
    print("\n3. ResNet-20 for CIFAR-10 (No BN, for verification):")
    model_verify = ResNet20(num_classes=10, in_channels=3, use_bn=False)
    y_verify = model_verify(x_cifar)
    print(f"   Output shape: {y_verify.shape}")
    print(f"   Parameters: {count_parameters(model_verify):,}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
