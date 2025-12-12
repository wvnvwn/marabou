"""
Training Module for ResNet Models

Provides training and evaluation functions for neural networks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    Train the model for one epoch.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })

    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def evaluate(model, test_loader, criterion, device):
    """
    Evaluate the model on test set.

    Args:
        model: PyTorch model
        test_loader: Test data loader
        criterion: Loss function
        device: Device to evaluate on

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    avg_loss = running_loss / len(test_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def train_model(model, train_loader, test_loader, num_epochs=50,
                learning_rate=0.01, device='cuda', checkpoint_dir='./checkpoints',
                use_bn=False, model_name=None):
    """
    Train a model with standard training loop.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        test_loader: Test data loader
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        use_bn: Whether batch normalization is used
        model_name: Name of model architecture (for checkpoint saving)

    Returns:
        Trained model
    """
    model = model.to(device)

    # Detect num_classes from model
    num_classes = model.linear.out_features
    criterion = nn.CrossEntropyLoss()
    
    # Use Adam for small models (faster convergence), SGD for ResNets
    is_small_model = model_name in ['smallcnn', 'tinycnn'] if model_name else False
    
    if is_small_model:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        # Simple step scheduler for small models
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                             momentum=0.9, weight_decay=5e-4)
        # Multi-step scheduler for ResNets
        milestones = [int(num_epochs * 0.5), int(num_epochs * 0.75), int(num_epochs * 0.9)]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    best_acc = 0.0

    print(f"\n{'='*60}")
    print(f"Training Configuration")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Optimizer: {'Adam' if is_small_model else 'SGD'}")
    print(f"Batch size: {train_loader.batch_size}")
    print(f"{'='*60}\n")

    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Evaluate
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )

        # Learning rate step
        scheduler.step()

        print(f"Epoch {epoch:3d} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            
            # Auto-detect model_name if not provided
            if model_name is None:
                state_keys = list(model.state_dict().keys())
                if any('layer1.' in k for k in state_keys):
                    # ResNet model
                    layer1_blocks = sum(1 for k in state_keys if k.startswith('layer1.') and k.endswith('.conv1.weight'))
                    if layer1_blocks <= 3:
                        model_name_save = 'resnet20'
                    elif layer1_blocks <= 5:
                        model_name_save = 'resnet32'
                    else:
                        model_name_save = 'resnet56'
                elif 'conv2.weight' in state_keys:
                    model_name_save = 'smallcnn'
                else:
                    model_name_save = 'tinycnn'
            else:
                model_name_save = model_name
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': test_acc,
                'use_bn': use_bn,
                'num_classes': num_classes,
                'model_name': model_name_save,
            }, checkpoint_path)
            print(f"  ✓ Best model saved (acc={best_acc:.2f}%)")

    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best test accuracy: {best_acc:.2f}%")
    print(f"{'='*60}\n")

    return model


if __name__ == "__main__":
    # Simple training test
    print("Training module loaded successfully.")
