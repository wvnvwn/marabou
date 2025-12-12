"""
Dataset Management Module

Provides unified interface for loading MNIST and CIFAR-10 datasets.
"""

from .dataset import load_dataset, get_dataloaders

__all__ = ['load_dataset', 'get_dataloaders']
