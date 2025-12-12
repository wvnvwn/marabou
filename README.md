# Personal Project for MARABOU

---

## 1. Project Overview

This project demonstrates neural network verification using [Marabou](https://github.com/NeuralNetworkVerification/Marabou), an SMT-based verification tool for neural networks. The implementation includes:

1. **Training** a custom CNN model on CIFAR-10
2. **Converting** the trained model to ONNX format
3. **Verifying** local robustness using Marabou

### Key Features
- Custom small CNN architectures optimized for Marabou verification
- Automatic model architecture detection from checkpoints
- Single-command execution for the complete pipeline
- Detailed verification results with timing information

---

## 2. Environment & Requirements

### 2.1 Python Environment

```
Python 3.10 (recommended)
```

### 2.2 Required Packages

```
# Deep Learning
torch>=2.0.0
torchvision>=0.15.0

# ONNX Export
onnx>=1.14.0
onnxruntime>=1.15.0

# Verification Tool
maraboupy>=2.0.0

# Utilities
numpy>=1.21.0,<2.0
tqdm>=4.65.0
```

### 2.3 Setup Instructions

**Option A: Using Conda (Recommended)**
```bash
conda env create -f environment.yml
conda activate marabou
```

**Option B: Using pip**
```bash
pip install -r requirements.txt
```

---

## 3. Model and Dataset Selection

### 3.1 Dataset: CIFAR-10

**Why CIFAR-10?**
- **10 classes** → Only 9 verification queries needed (vs. 99 for CIFAR-100)
- **Standard benchmark** for image classification
- **Reasonable image size** (32×32×3) for verification
- **Not in Marabou resources directory** (satisfies assignment requirement)

### 3.2 Model: SmallCNN

**Available Models:**

| Model | Parameters | Marabou Suitability | Accuracy |
|-------|------------|---------------------|----------|
| **SmallCNN** | ~50K | ✅ Recommended | ~70-75% |
| **TinyCNN** | ~15K | ✅ Fastest | ~60-65% |
| ResNet-20 | ~270K | ⚠️ Slow | ~85-90% |
| ResNet-56 | ~850K | ❌ May timeout | ~90%+ |

**Why SmallCNN?**
- **Appropriately sized** for Marabou (as required by assignment)
- **Fast verification** (minutes, not hours)
- **Good balance** between accuracy and verifiability
- **Simple architecture** without complex operations

**SmallCNN Architecture:**
```
Input (3×32×32)
  ↓
Conv2D(16, 3×3) → ReLU → MaxPool(2×2)
  ↓
Conv2D(32, 3×3) → ReLU → MaxPool(2×2)
  ↓
Flatten → FC(128) → ReLU → FC(10)
  ↓
Output (10 classes)
```

---

## 4. Quick Start (Single Execution)

### Run Complete Pipeline

```bash
cd /path/to/assignment3

# Default: SmallCNN on CIFAR-10 (Recommended)
bash run_all.sh

# Alternative: TinyCNN for fastest verification
bash run_all.sh tinycnn
```

This single command will:
1. Download CIFAR-10 dataset (if not present)
2. Train the model for 50 epochs
3. Convert to ONNX format
4. Run Marabou verification
5. Save results to `results/verification_results.txt`

### Expected Output

```
============================================================
Neural Network Verification - Assignment 3
Complete Pipeline: SMALLCNN on CIFAR10
============================================================

STEP 1: Training SMALLCNN on CIFAR10
...
✓ Training completed successfully!

STEP 2: ONNX Conversion + Marabou Verification
...
✓ Verification completed successfully!

Results:
  1. Training:
     - Model: SMALLCNN on CIFAR10
     - Weights: checkpoints/best_model.pth

  2. Verification:
     - ONNX: checkpoints/best_model.onnx
     - Results: results/verification_results.txt
============================================================
```

---

## 5. Step-by-Step Process

### Step 1: Training

```bash
# Train SmallCNN on CIFAR-10 (default)
bash run_train.sh

# Or specify model and dataset
bash run_train.sh smallcnn cifar10
bash run_train.sh tinycnn cifar10
```

**Output Files:**
- `checkpoints/best_model.pth` - Best validation accuracy checkpoint
- `checkpoints/final_model.pth` - Final epoch checkpoint

**Training Configuration:**
- Optimizer: Adam (lr=0.01)
- Epochs: 50
- Batch size: 128 (auto-adjusted based on GPU memory)

### Step 2: ONNX Conversion & Verification

```bash
bash run_verify.sh
```

**What it does:**
1. Loads trained model from `checkpoints/best_model.pth`
2. Auto-detects model architecture (SmallCNN, TinyCNN, ResNet, etc.)
3. Converts to ONNX format (opset 10 for Marabou compatibility)
4. Selects a correctly classified test sample
5. Runs Marabou verification for local robustness

**Verification Property:**
- **Local Robustness**: For a given input x with true label y, verify that no adversarial perturbation within ε-ball can change the prediction.
- **ε (epsilon)**: 0.02 (L∞ norm)
- **Timeout**: 60 seconds per query

---

## 6. File Structure

```
assignment3/
├── run_all.sh              # Complete pipeline (single execution)
├── run_train.sh            # Training only
├── run_verify.sh           # Verification only
├── train_model.py          # Training script
├── verify_model.py         # ONNX conversion + Marabou verification
├── source/
│   ├── models/
│   │   ├── small_cnn.py    # SmallCNN, TinyCNN architectures
│   │   └── resnet.py       # ResNet-20/32/56 architectures
│   ├── data/
│   │   └── dataset.py      # CIFAR-10/100 data loading
│   ├── verification/
│   │   └── marabou_verifier.py  # Marabou wrapper
│   ├── train.py            # Training loop
│   └── utils/              # Utility functions
├── checkpoints/            # Saved models
├── results/                # Verification results
├── requirements.txt        # Python dependencies
├── environment.yml         # Conda environment
└── README.md               # This file
```

---

## 7. Verification Results

### Sample Output (`results/verification_results.txt`)

```
Verification Results
============================================================
Model: ./checkpoints/best_model.pth
ONNX: ./checkpoints/best_model.onnx
Test sample label: 3
Epsilon: 0.02
Timeout: 60s

Result: ROBUST
Verification time: 45.23s
============================================================
```

### Interpreting Results

| Result | Meaning |
|--------|---------|
| **ROBUST** | No adversarial example found within ε-ball |
| **NOT ROBUST** | Counterexample found (model can be fooled) |
| **TIMEOUT** | Verification incomplete (model may be too large) |
| **SKIPPED** | Marabou not installed |

---

## 8. Advanced Usage

### Custom Training Parameters

```bash
python train_model.py \
    --model smallcnn \
    --dataset cifar10 \
    --epochs 50 \
    --lr 0.01 \
    --batch-size 128 \
    --seed 42
```

### Custom Verification Parameters

```bash
python verify_model.py \
    --model-path ./checkpoints/best_model.pth \
    --epsilon 0.02 \
    --timeout 60 \
    --seed 42
```

### Using Different Models

```bash
# TinyCNN - Fastest verification (~15K params)
bash run_all.sh tinycnn

# SmallCNN - Recommended (~50K params)
bash run_all.sh smallcnn

# ResNet-20 - Higher accuracy, slower verification (~270K params)
bash run_all.sh resnet20 cifar10
```

---

## 8. Expected Execution Time

| Task | GPU (RTX 3090) | CPU |
|------|----------------|-----|
| Training (50 epochs) | ~20-30 min | ~2-3 hours |
| ONNX Conversion | ~5 sec | ~5 sec |
| Marabou Verification | ~1-5 min | ~1-5 min |
| **Total** | **~30-40 min** | **~2-3 hours** |
