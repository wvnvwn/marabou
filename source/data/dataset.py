"""
Dataset Loading and Preprocessing

Handles MNIST and CIFAR-10 dataset loading with proper transformations.
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_transforms(dataset_name='cifar100', augment=True):
    """
    Get data transforms for the specified dataset.

    Args:
        dataset_name: 'cifar100'
        augment: Whether to apply data augmentation (for training)

    Returns:
        Tuple of (train_transform, test_transform)
    """
    if dataset_name.lower() == 'cifar100':
        # CIFAR-100 transforms
        # Normalization values from https://github.com/kuangliu/pytorch-cifar
        if augment:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408),
                    (0.2675, 0.2565, 0.2761)
                ),
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408),
                    (0.2675, 0.2565, 0.2761)
                ),
            ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408),
                (0.2675, 0.2565, 0.2761)
            ),
        ])

    elif dataset_name.lower() == 'cifar10':
        # CIFAR-10 transforms
        if augment:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010)
                ),
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010)
                ),
            ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)
            ),
        ])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return train_transform, test_transform


def load_dataset(dataset_name='cifar100', data_root='./data', augment=True):
    """
    Load the specified dataset.

    Args:
        dataset_name: 'cifar100'
        data_root: Root directory for dataset storage
        augment: Whether to apply data augmentation

    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    train_transform, test_transform = get_transforms(dataset_name, augment)

    if dataset_name.lower() == 'cifar100':
        train_dataset = datasets.CIFAR100(
            root=data_root,
            train=True,
            download=True,
            transform=train_transform
        )
        test_dataset = datasets.CIFAR100(
            root=data_root,
            train=False,
            download=True,
            transform=test_transform
        )

    elif dataset_name.lower() == 'cifar10':
        train_dataset = datasets.CIFAR10(
            root=data_root,
            train=True,
            download=True,
            transform=train_transform
        )
        test_dataset = datasets.CIFAR10(
            root=data_root,
            train=False,
            download=True,
            transform=test_transform
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return train_dataset, test_dataset


def get_dataloaders(dataset_name='cifar100', data_root='./data',
                   batch_size=128, num_workers=2, augment=True):
    """
    Get data loaders for training and testing.

    Args:
        dataset_name: 'cifar100'
        data_root: Root directory for dataset storage
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        augment: Whether to apply data augmentation

    Returns:
        Tuple of (train_loader, test_loader)
    """
    train_dataset, test_dataset = load_dataset(dataset_name, data_root, augment)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader


if __name__ == "__main__":
    # Test dataset loading
    print("Testing dataset loading...")

    print("\nLoading CIFAR-100...")
    train_loader, test_loader = get_dataloaders('cifar100', batch_size=128)
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Test batches: {len(test_loader)}")

    print("\nDataset loading test passed!")
