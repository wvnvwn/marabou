#!/bin/bash

# Verification Script - ONNX Conversion + Marabou Verification
# Can run on CPU nodes (Marabou doesn't use GPU)

set -e

# Parse arguments
MODEL=${1:-}  # Optional: smallcnn, tinycnn, resnet20, etc. (auto-detected if not specified)

echo "============================================================"
echo "ONNX Conversion + Marabou Verification"
echo "============================================================"
echo ""

# Configuration
MODEL_PATH="./checkpoints/best_model.pth"
EPSILON=0.02
TIMEOUT=60
SEED=42

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "✗ Model not found: $MODEL_PATH"
    echo ""
    echo "Please run training first:"
    echo "  bash run_train.sh"
    exit 1
fi

echo "Model checkpoint: $MODEL_PATH"
if [ -n "$MODEL" ]; then
    echo "Architecture: $MODEL (specified)"
else
    echo "Architecture: (auto-detect from checkpoint)"
fi
echo "Epsilon: $EPSILON"
echo "Timeout: ${TIMEOUT}s per query"
echo "============================================================"
echo ""

# Build command
CMD="python verify_model.py --model-path $MODEL_PATH --epsilon $EPSILON --timeout $TIMEOUT --seed $SEED"
if [ -n "$MODEL" ]; then
    CMD="$CMD --model $MODEL"
fi

# Run verification
eval $CMD

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ Verification completed successfully!"
    echo ""
    echo "Results saved to: results/verification_results.txt"
else
    echo ""
    echo "✗ Verification failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
