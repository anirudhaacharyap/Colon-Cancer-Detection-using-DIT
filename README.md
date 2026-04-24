# 🔬 Colon Cancer Detection using Hybrid BOA-WOA + DiT

A deep learning pipeline for **binary classification of colon histopathology images** (Benign vs. Adenocarcinoma) using a novel **Hybrid Butterfly-Whale Optimization Algorithm (BOA-WOA)** for feature selection and a **Diffusion Transformer (DiT)** classifier.

Built on the [LC25000 Lung and Colon Histopathological Image Dataset](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images).

---

## 📋 Table of Contents

- [Architecture Overview](#architecture-overview)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Dataset Preparation](#dataset-preparation)
- [Commands & Usage](#commands--usage)
- [Generated Outputs & Graphs](#generated-outputs--graphs)
- [Configuration](#configuration)
- [Hardware Optimization](#hardware-optimization)

---

## 🏗️ Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                        INPUT IMAGES                              │
│                  (224×224 Histopathology)                         │
└──────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│              FEATURE EXTRACTION (3 Pretrained Backbones)         │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────────┐      │
│  │  ResNet-50   │  │  DenseNet-121 │  │   ViT-B/16       │      │
│  │  (2048-dim)  │  │  (1024-dim)   │  │   (768-dim)      │      │
│  └──────┬───────┘  └──────┬────────┘  └────────┬─────────┘      │
│         └─────────────────┼────────────────────┘                 │
│                           ▼                                      │
│                  Concatenated: 3840-dim                           │
└──────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│           HYBRID BOA-WOA FEATURE SELECTION                       │
│  ┌─────────────────────┐    ┌─────────────────────┐              │
│  │  Butterfly Optim.   │───▶│  Whale Optim.       │              │
│  │  (Global + Local)   │    │  (Spiral + Encircle)│              │
│  └─────────────────────┘    └─────────────────────┘              │
│         Output: Binary mask selecting optimal features           │
└──────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│              DiT CLASSIFIER (Diffusion Transformer)              │
│  ┌────────┐  ┌───────────────┐  ┌──────────┐  ┌──────────┐      │
│  │ Patch  │─▶│ DiT Blocks    │─▶│  CLS     │─▶│ Softmax  │      │
│  │ Embed  │  │ (adaLN-Zero)  │  │  Head    │  │ Output   │      │
│  └────────┘  └───────────────┘  └──────────┘  └──────────┘      │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Language** | Python 3.10+ | Core language |
| **Deep Learning** | PyTorch 2.x | Model training, AMP, `torch.compile` |
| **Vision Models** | torchvision | ResNet-50, DenseNet-121 (pretrained) |
| **ViT** | timm | Vision Transformer ViT-B/16 (pretrained) |
| **Optimization** | NumPy | Vectorized BOA/WOA metaheuristic algorithms |
| **Metrics** | scikit-learn | Accuracy, Precision, Recall, F1, AUC-ROC, MCC |
| **Statistics** | SciPy | 95% Confidence Intervals |
| **Visualization** | Matplotlib, Seaborn | Training curves, ROC, confusion matrix |
| **Data** | Pandas | Comparison tables |
| **Progress** | tqdm | Progress bars during extraction |

---

## 📁 Project Structure

```
boa_woa_dit/
├── config.py                          # All hyperparameters & hardware settings
├── train.py                           # Main training pipeline (entry point)
├── evaluate.py                        # Model evaluation & metrics
├── plot_roc_comparison.py             # ROC curve comparison across ablations
├── plot_tp_vs_fp.py                   # TP vs FP scatter plot comparison
├── requirements.txt                   # Python dependencies
├── check_env.py                       # Environment verification script
├── test_mock.py                       # Quick mock test
│
├── data/
│   ├── dataset.py                     # DataLoader with augmentation & splits
│   └── LC25000/                       # Dataset directory (not tracked in git)
│       ├── Train and Validation Set/
│       │   ├── colon_aca/             # Adenocarcinoma images
│       │   └── colon_n/               # Benign images
│       └── Test Set/
│           ├── colon_aca/
│           └── colon_n/
│
├── models/
│   └── dit_classifier.py             # DiT model with adaLN-Zero conditioning
│
├── optimization/
│   ├── hybrid_boa_woa.py             # Main hybrid optimization loop
│   ├── boa.py                        # Butterfly Optimization Algorithm (vectorized)
│   ├── woa.py                        # Whale Optimization Algorithm (vectorized)
│   └── fitness.py                    # Fitness evaluation (lightweight DiT)
│
├── feature_cache/                     # Cached extracted features (auto-generated)
├── checkpoints/                       # Saved model weights (auto-generated)
└── logs/                             # Training logs & plots (auto-generated)
```

---

## ⚙️ Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/anirudhaacharyap/Colon-Cancer-Detection-using-DIT.git
cd Colon-Cancer-Detection-using-DIT
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
```

### 3. Install PyTorch (CUDA)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Verify Environment

```bash
python check_env.py
```

---

## 📂 Dataset Preparation

1. Download the **LC25000** dataset from [Kaggle](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)
2. Extract and organize colon images into:

```
data/LC25000/
├── Train and Validation Set/
│   ├── colon_aca/    # Adenocarcinoma training images
│   └── colon_n/      # Benign/Normal training images
└── Test Set/
    ├── colon_aca/    # Adenocarcinoma test images
    └── colon_n/      # Benign/Normal test images
```

> Split ratio: ~70% Train, ~15% Validation (auto-split from Train folder), ~15% Test

---

## 🚀 Commands & Usage

### Full Pipeline (Train + Evaluate)

Runs the complete pipeline: Feature Extraction → BOA-WOA Optimization → DiT Training → Evaluation.

```bash
python train.py
```

### Ablation Studies

Run individual components to compare against the full hybrid approach:

```bash
# DiT classifier only (no feature selection)
python train.py --ablation dit_only

# BOA-only feature selection (no WOA phase)
python train.py --ablation boa_only

# WOA-only feature selection (no BOA phase)
python train.py --ablation woa_only
```

### Standalone Evaluation

Re-evaluate a previously trained model without retraining:

```bash
python evaluate.py
```

### Generate Comparison Graphs

```bash
# ROC curve comparison across all ablation variants
python plot_roc_comparison.py

# True Positives vs False Positives scatter plot
python plot_tp_vs_fp.py
```

### Environment Check

```bash
python check_env.py
```

---

## 📊 Generated Outputs & Graphs

### Metrics (saved to `logs/results.txt`)

| Metric | Description |
|--------|-------------|
| Accuracy | Overall classification accuracy (%) |
| Precision | Positive predictive value |
| Recall (Sensitivity) | True positive rate |
| F1-Score | Harmonic mean of precision and recall |
| AUC-ROC | Area under the ROC curve |
| Specificity | True negative rate |
| MCC | Matthews Correlation Coefficient |
| 95% CI | Confidence interval for accuracy |

### Graphs

| Graph | File | Generated By | Description |
|-------|------|-------------|-------------|
| **Training Curves** | `logs/full_run/training_curves.png` | `train.py` | Loss, Accuracy, and LR schedule over epochs |
| **Confusion Matrix** | `logs/full_run/confusion_matrix.png` | `evaluate.py` | Heatmap of TP/TN/FP/FN |
| **ROC Curve** | `logs/full_run/roc_curve.png` | `evaluate.py` | ROC curve with AUC score |
| **ROC Comparison** | `logs/roc_curve_comparison.png` | `plot_roc_comparison.py` | All 4 variants overlaid |
| **TP vs FP** | `tp_vs_fp_comparison.png` | `plot_tp_vs_fp.py` | Scatter plot comparing model performance |

### Checkpoints

| File | Description |
|------|-------------|
| `checkpoints/best_dit_model.pth` | Best model weights (lowest val loss) |
| `feature_cache/features_train.npy` | Cached 3840-dim training features |
| `feature_cache/features_val.npy` | Cached validation features |
| `feature_cache/features_test.npy` | Cached test features |
| `feature_cache/optimal_mask.npy` | Best feature selection mask from BOA-WOA |

---

## 🔧 Configuration

All hyperparameters are in [`config.py`](config.py). Key settings:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `BATCH_SIZE` | 128 | Feature extraction batch size |
| `POPULATION_SIZE` | 50 | BOA-WOA search agents |
| `MAX_ITER` | 75 | Optimization iterations |
| `FINAL_DIT_HIDDEN_DIM` | 768 | Transformer hidden dimension |
| `FINAL_DIT_DEPTH` | 12 | Number of DiT blocks |
| `FINAL_EPOCHS` | 150 | Max training epochs |
| `FINAL_LR` | 3e-4 | Learning rate (with cosine warmup) |
| `USE_AMP` | True | Mixed precision training (FP16) |
| `COMPILE_MODEL` | True | torch.compile() for faster kernels |
| `EARLY_STOP_PATIENCE` | 20 | Early stopping patience |

---

## 🖥️ Hardware Optimization

This pipeline is optimized for high-end hardware:

| Component | Optimization |
|-----------|-------------|
| **64-core CPU** | 24 DataLoader workers, 64 PyTorch threads, vectorized BOA/WOA |
| **256 GB RAM** | Aggressive prefetch (factor=4), pinned memory |
| **RTX 3090 24GB** | AMP mixed precision, torch.compile, batch size 256, pre-allocated GPU tensors |

---

## 📄 License

This project is for academic/research purposes.

## 👤 Author

**Anirudha Acharya P**  
[GitHub](https://github.com/anirudhaacharyap)
