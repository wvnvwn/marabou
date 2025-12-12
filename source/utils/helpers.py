"""
Helper Utilities for Training and Evaluation

Common utility functions used throughout the project.
"""

import torch
import torch.nn as nn
import random
import numpy as np
import os


def set_seed(seed=42):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """
    Get the appropriate device (CUDA if available, otherwise CPU).

    Returns:
        torch.device object
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, optimizer, epoch, accuracy, path):
    """
    Save model checkpoint.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        accuracy: Model accuracy
        path: Path to save checkpoint
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer=None, path=None):
    """
    Load model checkpoint.

    Args:
        model: PyTorch model
        optimizer: Optimizer (optional)
        path: Path to checkpoint file

    Returns:
        Tuple of (model, optimizer, epoch, accuracy)
    """
    if path is None or not os.path.exists(path):
        print(f"Checkpoint not found at {path}")
        return model, optimizer, 0, 0.0

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    accuracy = checkpoint.get('accuracy', 0.0)

    print(f"Checkpoint loaded from {path}")
    print(f"  Epoch: {epoch}, Accuracy: {accuracy:.2f}%")

    return model, optimizer, epoch, accuracy


def get_correctly_classified_sample(model, test_loader, device):
    """
    Get a correctly classified test sample.

    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run on

    Returns:
        Tuple of (image, true_label, predicted_label)
    """
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)

            for i in range(len(target)):
                if pred[i] == target[i]:
                    return data[i].cpu(), target[i].item(), pred[i].item()
    return None, None, None


if __name__ == "__main__":
    print("Testing utility functions...")

    # Test seed setting
    set_seed(42)
    print(f"✓ Seed set to 42")

    # Test device
    device = get_device()
    print(f"✓ Device: {device}")

    # Test parameter counting
    model = nn.Linear(10, 5)
    params = count_parameters(model)
    print(f"✓ Parameter counting: {params} parameters")

    print("\nAll utility tests passed!")
