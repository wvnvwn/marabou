#!/bin/bash

# Complete Pipeline: Training + ONNX Conversion + Marabou Verification
# Optimized for Marabou: SmallCNN on CIFAR-10 by default

set -e

# Parse arguments
MODEL=${1:-smallcnn}   # Default: smallcnn (recommended for Marabou)
DATASET=${2:-cifar10}  # Default: cifar10 (10 classes = faster verification)

# Validate model choice
if [[ ! "$MODEL" =~ ^(smallcnn|tinycnn|resnet20|resnet32|resnet56)$ ]]; then
    echo "✗ Invalid model: $MODEL"
    echo ""
    echo "Usage: bash run_all.sh [MODEL] [DATASET]"
    echo ""
    echo "  MODEL (default: smallcnn):"
    echo "    smallcnn  - Small CNN (~50K params, recommended for Marabou)"
    echo "    tinycnn   - Tiny CNN (~15K params, fastest verification)"
    echo "    resnet20  - ResNet-20 (~270K params)"
    echo "    resnet32  - ResNet-32 (~470K params)"
    echo "    resnet56  - ResNet-56 (~850K params, may timeout)"
    echo ""
    echo "  DATASET (default: cifar10):"
    echo "    cifar10   - 10 classes (recommended, 9 verification queries)"
    echo "    cifar100  - 100 classes (99 verification queries, slow)"
    echo ""
    echo "Examples:"
    echo "  bash run_all.sh                    # SmallCNN + CIFAR-10 (recommended)"
    echo "  bash run_all.sh tinycnn            # TinyCNN + CIFAR-10 (fastest)"
    echo "  bash run_all.sh resnet20 cifar10   # ResNet-20 + CIFAR-10"
    exit 1
fi

# Validate dataset choice
if [[ ! "$DATASET" =~ ^(cifar10|cifar100)$ ]]; then
    echo "✗ Invalid dataset: $DATASET"
    echo "  Valid options: cifar10, cifar100"
    exit 1
fi

MODEL_UPPER=$(echo "$MODEL" | tr '[:lower:]' '[:upper:]')
DATASET_UPPER=$(echo "$DATASET" | tr '[:lower:]' '[:upper:]')

echo "============================================================"
echo "Neural Network Verification - Assignment 3"
echo "Complete Pipeline: $MODEL_UPPER on $DATASET_UPPER"
echo "============================================================"
echo ""

# Warning for large models
if [[ "$MODEL" =~ ^(resnet32|resnet56)$ ]]; then
    echo "⚠ Warning: $MODEL_UPPER is large and may cause timeout in Marabou."
    echo "  Consider using smallcnn or tinycnn for faster verification."
    echo ""
fi

if [ "$DATASET" = "cifar100" ]; then
    echo "⚠ Warning: CIFAR-100 has 100 classes (99 verification queries)."
    echo "  Consider using cifar10 (9 queries) for faster verification."
    echo ""
fi

# Step 1: Training
echo "STEP 1: Training $MODEL_UPPER on $DATASET_UPPER"
echo "------------------------------------------------------------"
bash run_train.sh $MODEL $DATASET
echo ""

# Step 2: Verification
echo "============================================================"
echo "STEP 2: ONNX Conversion + Marabou Verification"
echo "------------------------------------------------------------"
bash run_verify.sh $MODEL
echo ""

# Summary
echo "============================================================"
echo "✓ Complete pipeline finished successfully!"
echo "============================================================"
echo ""
echo "Results:"
echo "  1. Training:"
echo "     - Model: $MODEL_UPPER on $DATASET_UPPER"
echo "     - Weights: checkpoints/best_model.pth"
echo ""
echo "  2. Verification:"
echo "     - ONNX: checkpoints/best_model.onnx"
echo "     - Results: results/verification_results.txt"
echo ""
echo "============================================================"
