#!/bin/bash

# Training Script for Neural Network Verification
# Optimized for Marabou: SmallCNN on CIFAR-10 by default

set -e

# Parse arguments
MODEL=${1:-smallcnn}  # Default: smallcnn (recommended for Marabou)
DATASET=${2:-cifar10}  # Default: cifar10 (10 classes = faster verification)

# Validate model choice
if [[ ! "$MODEL" =~ ^(smallcnn|tinycnn|resnet20|resnet32|resnet56)$ ]]; then
    echo "✗ Invalid model: $MODEL"
    echo ""
    echo "Usage: bash run_train.sh [MODEL] [DATASET]"
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
    echo "  bash run_train.sh                    # SmallCNN + CIFAR-10 (recommended)"
    echo "  bash run_train.sh tinycnn            # TinyCNN + CIFAR-10 (fastest)"
    echo "  bash run_train.sh resnet20 cifar10   # ResNet-20 + CIFAR-10"
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
echo "Training $MODEL_UPPER on $DATASET_UPPER"
echo "============================================================"
echo ""

# Check GPU
GPU_INFO=$(python -c "
import torch
if torch.cuda.is_available():
    try:
        _ = torch.zeros(1).cuda()
        props = torch.cuda.get_device_properties(0)
        print(f'{props.name}|{props.total_memory/1024**3:.1f}')
    except:
        print('None|0')
else:
    print('None|0')
" 2>/dev/null)

GPU_NAME=$(echo "$GPU_INFO" | cut -d'|' -f1)
GPU_MEM=$(echo "$GPU_INFO" | cut -d'|' -f2)

if [ "$GPU_NAME" != "None" ]; then
    echo "GPU: $GPU_NAME (${GPU_MEM}GB)"
    # Auto-adjust batch size
    GPU_MEM_INT=${GPU_MEM%.*}
    if [ "$GPU_MEM_INT" -ge 8 ]; then
        BATCH_SIZE=128
    else
        BATCH_SIZE=64
    fi
else
    echo "GPU: Not available (using CPU)"
    BATCH_SIZE=64
fi

# Configuration
EPOCHS=100
if [[ "$MODEL" =~ ^(smallcnn|tinycnn)$ ]]; then
    LR=0.01
else
    LR=0.1
fi

SEED=42
NUM_WORKERS=4

echo ""
echo "Configuration:"
echo "  Model: $MODEL_UPPER"
echo "  Dataset: $DATASET_UPPER"
echo "  Epochs: $EPOCHS"
echo "  Learning rate: $LR"
echo "  Batch Size: $BATCH_SIZE"
echo "  Workers: $NUM_WORKERS"
echo "  Seed: $SEED"
echo "============================================================"
echo ""

# Run training
python train_model.py \
    --model $MODEL \
    --dataset $DATASET \
    --epochs $EPOCHS \
    --lr $LR \
    --batch-size $BATCH_SIZE \
    --num-workers $NUM_WORKERS \
    --seed $SEED

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ Training completed successfully!"
    echo ""
    echo "Next step: Run verification"
    echo "  bash run_verify.sh"
else
    echo ""
    echo "✗ Training failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
