#!/usr/bin/env python3
"""
Verification Script - Convert to ONNX and Verify with Marabou
Can be run on CPU nodes (Marabou doesn't use GPU)
"""

import os
import sys
import argparse
import torch
import torch.onnx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'source'))

from models import SmallCNN, TinyCNN, ResNet20, ResNet32, ResNet56
from data import get_dataloaders
from utils import set_seed, get_correctly_classified_sample
from verification import verify_local_robustness


def parse_args():
    parser = argparse.ArgumentParser(description='Verify ResNet with Marabou')

    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--model', type=str, default=None,
                       choices=['smallcnn', 'tinycnn', 'resnet20', 'resnet32', 'resnet56'],
                       help='Model architecture (auto-detected from checkpoint if not specified)')
    parser.add_argument('--data-root', type=str, default='./data',
                       help='Root directory for datasets')
    parser.add_argument('--epsilon', type=float, default=0.02,
                       help='Perturbation bound (default: 0.02)')
    parser.add_argument('--timeout', type=int, default=60,
                       help='Timeout per query in seconds (default: 60)')
    parser.add_argument('--onnx-path', type=str, default=None,
                       help='Path to save ONNX model (default: auto)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')

    return parser.parse_args()


def detect_model_name(state_dict, checkpoint):
    """
    Detect model architecture from checkpoint.
    
    Args:
        state_dict: Model state dictionary
        checkpoint: Full checkpoint dictionary
    
    Returns:
        Model name string ('smallcnn', 'tinycnn', 'resnet20', 'resnet32', or 'resnet56')
    """
    # First, check if model_name is saved in checkpoint
    model_name = checkpoint.get('model_name', None)
    if model_name is not None:
        return model_name
    
    state_keys = list(state_dict.keys())
    
    # Check if it's a ResNet (has layer1)
    if any('layer1.' in k for k in state_keys):
        # Auto-detect ResNet variant from layer1 block count
        layer1_blocks = sum(1 for k in state_keys 
                           if k.startswith('layer1.') and k.endswith('.conv1.weight'))
        
        if layer1_blocks <= 3:
            return 'resnet20'
        elif layer1_blocks <= 5:
            return 'resnet32'
        else:
            return 'resnet56'
    
    # Check for SmallCNN vs TinyCNN
    if 'conv2.weight' in state_keys:
        return 'smallcnn'
    else:
        return 'tinycnn'


def convert_to_onnx(model, onnx_path):
    """Convert PyTorch model to ONNX format."""
    print(f"\n{'='*60}")
    print(f"Converting to ONNX")
    print(f"{'='*60}")

    model.eval()
    model = model.cpu()

    dummy_input = torch.randn(1, 3, 32, 32)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    file_size = os.path.getsize(onnx_path) / 1024
    print(f"✓ ONNX model saved: {onnx_path}")
    print(f"  Size: {file_size:.2f} KB")
    print(f"  Opset: 10 (Marabou compatible)")
    print(f"{'='*60}")

    return onnx_path


def main():
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    print(f"\n{'='*60}")
    print(f"Neural Network Verification with Marabou")
    print(f"{'='*60}")
    print(f"Model: {args.model_path}")
    print(f"Epsilon: {args.epsilon}")
    print(f"Timeout: {args.timeout}s")
    print(f"{'='*60}")

    # Load model
    print(f"\nLoading trained model...")
    checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)

    # Auto-detect parameters from checkpoint
    state_dict = checkpoint['model_state_dict']

    # Auto-detect use_bn
    use_bn = checkpoint.get('use_bn', None)
    if use_bn is None:
        has_bn = any('bn' in key for key in state_dict.keys())
        use_bn = has_bn
        print(f"  Auto-detected use_bn={use_bn} from checkpoint")

    # Auto-detect num_classes from linear layer shape
    num_classes = checkpoint.get('num_classes', None)
    if num_classes is None:
        # linear.weight shape: [num_classes, 64]
        num_classes = state_dict['linear.weight'].shape[0]
        print(f"  Auto-detected num_classes={num_classes} from checkpoint")

    # Detect or use specified model architecture
    if args.model is not None:
        model_name = args.model
        print(f"  Using specified model: {model_name}")
    else:
        model_name = detect_model_name(state_dict, checkpoint)
        print(f"  Auto-detected model: {model_name}")

    # Create model with correct architecture
    model_dict = {
        'smallcnn': SmallCNN,
        'tinycnn': TinyCNN,
        'resnet20': ResNet20,
        'resnet32': ResNet32,
        'resnet56': ResNet56,
    }
    model_class = model_dict[model_name]
    model = model_class(num_classes=num_classes, in_channels=3, use_bn=use_bn)
    model.load_state_dict(state_dict)
    model.eval()

    accuracy = checkpoint.get('accuracy', 'N/A')
    print(f"✓ Model loaded ({model_name.upper()}, accuracy: {accuracy}%, use_bn={use_bn})")

    # Convert to ONNX
    if args.onnx_path is None:
        args.onnx_path = args.model_path.replace('.pth', '.onnx')

    convert_to_onnx(model, args.onnx_path)

    # Load test data
    print(f"\n{'='*60}")
    print(f"Loading test data")
    print(f"{'='*60}")

    # Auto-detect dataset from num_classes
    dataset_name = 'cifar100' if num_classes == 100 else 'cifar10'
    print(f"Dataset: {dataset_name.upper()} (detected from num_classes={num_classes})")

    _, test_loader = get_dataloaders(
        dataset_name, args.data_root, batch_size=1000, num_workers=2
    )
    print(f"✓ Test data loaded")

    # Get test sample
    print(f"\nSelecting test sample...")
    model_cpu = model.cpu()
    test_image, true_label, pred_label = get_correctly_classified_sample(
        model_cpu, test_loader, torch.device('cpu')
    )

    if test_image is None:
        print("✗ No correctly classified samples found!")
        return 1

    print(f"✓ Selected sample: label={true_label}")

    # Verify with Marabou
    is_robust, counterexample_class, verification_time = verify_local_robustness(
        args.onnx_path, test_image, true_label,
        epsilon=args.epsilon,
        timeout=args.timeout,
        verbose=True
    )

    # Save results
    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, 'verification_results.txt')

    with open(results_path, 'w') as f:
        f.write(f"Verification Results\n")
        f.write(f"{'='*60}\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"ONNX: {args.onnx_path}\n")
        f.write(f"Test sample label: {true_label}\n")
        f.write(f"Epsilon: {args.epsilon}\n")
        f.write(f"Timeout: {args.timeout}s\n\n")

        if is_robust is None:
            f.write(f"Result: SKIPPED (Marabou not available)\n")
        elif is_robust:
            f.write(f"Result: ROBUST\n")
            f.write(f"Verification time: {verification_time:.2f}s\n")
        else:
            f.write(f"Result: NOT ROBUST\n")
            f.write(f"Counterexample class: {counterexample_class}\n")
            f.write(f"Verification time: {verification_time:.2f}s\n")

        f.write(f"{'='*60}\n")

    print(f"\n{'='*60}")
    print(f"Verification completed!")
    print(f"Results saved to: {results_path}")
    print(f"{'='*60}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
