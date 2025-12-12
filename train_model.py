#!/usr/bin/env python3
"""
Training Script - Train CNN models on CIFAR-10/100

Supports:
- SmallCNN, TinyCNN: Recommended for Marabou verification (fast)
- ResNet-20/32/56: For more complex experiments (slower verification)

Default: SmallCNN on CIFAR-10 (optimized for Marabou)
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'source'))

from models import SmallCNN, TinyCNN, ResNet20, ResNet32, ResNet56, count_parameters
from data import get_dataloaders
from utils import set_seed, get_device
from train import train_model


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train CNN on CIFAR-10/100 for Marabou Verification'
    )

    # Model selection
    parser.add_argument('--model', type=str, default='smallcnn',
                       choices=['smallcnn', 'tinycnn', 'resnet20', 'resnet32', 'resnet56'],
                       help='Model architecture (default: smallcnn, recommended for Marabou)')
    
    # Dataset selection
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'cifar100'],
                       help='Dataset (default: cifar10, recommended for Marabou)')
    
    parser.add_argument('--data-root', type=str, default='./data',
                       help='Root directory for datasets')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate (default: 0.01)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers (default: 4)')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--use-bn', action='store_true', default=False,
                       help='Use batch normalization (not recommended for Marabou)')

    return parser.parse_args()


def main():
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Get device
    device = get_device()

    # Determine num_classes from dataset
    num_classes = 10 if args.dataset == 'cifar10' else 100

    # Select model
    model_dict = {
        'smallcnn': SmallCNN,
        'tinycnn': TinyCNN,
        'resnet20': ResNet20,
        'resnet32': ResNet32,
        'resnet56': ResNet56,
    }
    model_class = model_dict[args.model.lower()]

    print(f"\n{'='*60}")
    print(f"Training {args.model.upper()} on {args.dataset.upper()}")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Batch Normalization: {args.use_bn}")

    # Create model
    model = model_class(num_classes=num_classes, in_channels=3, use_bn=args.use_bn)
    print(f"Parameters: {count_parameters(model):,}")
    
    # Marabou compatibility note
    if args.model in ['smallcnn', 'tinycnn']:
        print(f"  → Optimized for Marabou verification")
    else:
        print(f"  ⚠ Large model - may cause timeout in Marabou")
    print(f"{'='*60}\n")

    # Load data
    print(f"Loading {args.dataset.upper()} dataset...")
    train_loader, test_loader = get_dataloaders(
        args.dataset, args.data_root, args.batch_size,
        num_workers=args.num_workers
    )
    print(f"✓ Dataset loaded ({num_classes} classes)\n")

    # Train
    model = train_model(
        model, train_loader, test_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        use_bn=args.use_bn,
        model_name=args.model
    )

    # Save final model
    final_path = os.path.join(args.checkpoint_dir, 'final_model.pth')
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    num_classes = model.linear.out_features
    torch.save({
        'model_state_dict': model.state_dict(),
        'use_bn': args.use_bn,
        'num_classes': num_classes,
        'model_name': args.model,
        'dataset': args.dataset,
    }, final_path)

    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Model saved to: {args.checkpoint_dir}/")
    print(f"  - best_model.pth (best test accuracy)")
    print(f"  - final_model.pth (final epoch)")
    print(f"{'='*60}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
