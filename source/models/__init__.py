"""
Neural Network Models for Verification

This module provides model architectures optimized for formal verification.
Includes:
- SmallCNN/TinyCNN: Recommended for Marabou verification (fast)
- ResNet variants: For more complex experiments (slower verification)
"""

from .resnet import (
    ResNet20, ResNet32, ResNet56,
    ResNet20_MNIST,
    BasicBlock,
    count_parameters
)

from .small_cnn import (
    SmallCNN,
    TinyCNN,
)

__all__ = [
    # Small models (recommended for Marabou)
    'SmallCNN', 'TinyCNN',
    # ResNet variants
    'ResNet20', 'ResNet32', 'ResNet56',
    'ResNet20_MNIST',
    'BasicBlock',
    'count_parameters'
]
