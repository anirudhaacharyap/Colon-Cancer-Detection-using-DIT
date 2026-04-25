# 🔬 Colon Cancer Detection using Hybrid BOA-WOA + DiT

A deep learning pipeline for **binary classification of colon histopathology images** (Benign vs. Adenocarcinoma) using a novel **Hybrid Butterfly-Whale Optimization Algorithm (BOA-WOA)** for feature selection and a **Diffusion Transformer (DiT)** classifier.

Built on the [LC25000 Lung and Colon Histopathological Image Dataset](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images).

### Key Features

- **Multi-backbone Feature Extraction**: ResNet-50 + DenseNet-121 + ViT-B/16 → 3840-dim features
- **Hybrid Metaheuristic Optimization**: BOA + WOA for optimal feature selection
- **DiT Classifier**: Diffusion Transformer with adaLN-Zero conditioning
- **Stochastic Weight Averaging (SWA)**: Improved generalization via weight averaging
- **Mixup Augmentation**: Regularization through input interpolation (α=0.2)
- **Temperature Scaling**: Post-hoc calibration with LBFGS-fitted temperature
- **Test-Time Augmentation (TTA)**: Multi-pass inference with Gaussian noise for stability
- **Label Smoothing**: Cross-entropy with 0.1 smoothing to prevent overconfidence
- **Comprehensive Evaluation**: ECE calibration plots, bootstrap CIs, DeLong tests, Pareto analysis

---

## 📋 Table of Contents

- [Architecture Overview](#architecture-overview)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Dataset Preparation](#dataset-preparation)
- [Commands & Usage](#commands--usage)
- [Generated Outputs & Graphs](#generated-outputs--graphs)
- [Training Pipeline Features](#training-pipeline-features)
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
└──────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│              POST-HOC CALIBRATION & EVALUATION                   │
│  ┌────────────┐  ┌───────────┐  ┌──────────────────────────┐    │
│  │ Temp Scale │─▶│    TTA    │─▶│ Metrics + Calibration    │    │
│  │  (LBFGS)   │  │ (7-pass) │  │ (ECE, Bootstrap, DeLong) │    │
│  └────────────┘  └───────────┘  └──────────────────────────┘    │
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
| **Statistics** | SciPy | 95% CIs, DeLong tests, Bootstrap |
| **Visualization** | Matplotlib, Seaborn | Training curves, ROC, calibration, Pareto |
| **Data** | Pandas | Comparison tables, prediction I/O |
| **Progress** | tqdm | Progress bars during extraction |
| **SWA** | torch.optim.swa_utils | Stochastic Weight Averaging |

---

## 📁 Project Structure

```
boa_woa_dit/
├── config.py                          # All hyperparameters & hardware settings
├── train.py                           # Main training pipeline (mixup, SWA, EMA)
├── evaluate.py                        # Evaluation (temp scaling, TTA, calibration)
├── plot_roc_comparison.py             # ROC comparison with zoom inset (real data)
├── plot_all_comparison_graphs.py      # Pareto plot + Bootstrap CI + DeLong tests
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
│   └── {run_name}/                   # Per-run checkpoints (acc, loss, SWA, latest)
└── logs/                             # Training logs & plots (auto-generated)
    ├── full_run/                     # Full hybrid pipeline results
    ├── ablation_dit_only/            # DiT-only ablation results
    ├── ablation_boa_only/            # BOA-only ablation results
    └── ablation_woa_only/            # WOA-only ablation results
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
python evaluate.py --ablation dit_only
```

### Generate Comparison Graphs

Run these **after all 4 variants** have been trained and evaluated:

```bash
# ROC curve comparison with zoom inset (uses real predictions.csv)
python plot_roc_comparison.py

# Pareto plot (accuracy vs training time) + Bootstrap CI + DeLong tests
python plot_all_comparison_graphs.py

# True Positives vs False Positives scatter plot
python plot_tp_vs_fp.py
```

### Environment Check

```bash
python check_env.py
```

---

## 📊 Generated Outputs & Graphs

### Metrics (saved to `logs/{run}/results.txt`)

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
| Temperature | Fitted temperature scaling parameter |
| ECE (before/after) | Expected Calibration Error pre/post temp scaling |
| Threshold | Optimally tuned decision threshold |

### Graphs

| Graph | File | Generated By | Description |
|-------|------|-------------|-------------|
| **Training Curves** | `logs/{run}/training_curves.png` | `train.py` | Loss (log scale), Accuracy (5-ep rolling avg), LR schedule with warmup/SWA markers |
| **Confusion Matrix** | `logs/{run}/confusion_matrix.png` | `evaluate.py` | Heatmap of TP/TN/FP/FN |
| **ROC Curve** | `logs/{run}/roc_curve.png` | `evaluate.py` | Single-model ROC curve with AUC |
| **Calibration Plot** | `logs/{run}/calibration_plot.png` | `evaluate.py` | ECE before/after temperature scaling with shaded gap |
| **ROC Comparison** | `logs/roc_curve_comparison.png` | `plot_roc_comparison.py` | All 4 ablation ROC curves with zoom inset |
| **Pareto Plot** | `logs/pareto_plot.png` | `plot_all_comparison_graphs.py` | Accuracy vs real training duration |
| **Bootstrap CI** | `logs/bootstrap_ci_plot.png` | `plot_all_comparison_graphs.py` | AUC CIs (n=2000) with DeLong p-values |
| **TP vs FP** | `tp_vs_fp_comparison.png` | `plot_tp_vs_fp.py` | Scatter plot comparing model performance |

### Checkpoints (per run)

| File | Description |
|------|-------------|
| `checkpoints/{run}/best_dit_model_acc.pth` | Best model weights (highest val accuracy) |
| `checkpoints/{run}/best_dit_model_loss.pth` | Best model weights (lowest val loss) |
| `checkpoints/{run}/best_dit_model_swa.pth` | SWA-averaged model weights |
| `checkpoints/{run}/latest_dit_model.pth` | Latest epoch model weights |
| `checkpoints/swa_dit_model.pth` | Top-level SWA checkpoint |
| `feature_cache/features_{split}.npy` | Cached 3840-dim features per split |
| `feature_cache/optimal_mask_{run}.npy` | Best feature selection mask per run |

---

## 🧪 Training Pipeline Features

| Feature | Description |
|---------|-------------|
| **Mixup Augmentation** | Input interpolation with α=0.2 Beta distribution; original labels for accuracy, mixed for loss |
| **Label Smoothing** | CrossEntropyLoss with 0.1 smoothing to prevent overconfident predictions |
| **Stochastic Weight Averaging** | Averages model weights from epoch 140+ for flatter minima and better generalization |
| **EMA (Exponential Moving Average)** | Shadow weights with 0.999 decay saved as checkpoints |
| **Cosine Warmup LR** | 10-epoch linear warmup → cosine decay to 1e-7 |
| **Temperature Scaling** | Post-hoc calibration: LBFGS-fitted single parameter on validation logits |
| **Test-Time Augmentation** | 7-pass Gaussian noise (σ=0.008) with averaged softmax probabilities |
| **Threshold Tuning** | Optimal decision threshold via F1-sweep on validation set |
| **Calibration Analysis** | 15-bin ECE computed before and after temperature scaling |
| **DeLong Test** | Statistical significance testing for AUC comparison between ablations |
| **Bootstrap CI** | 2000-sample bootstrap with seed=42 for reproducible confidence intervals |

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
| `FINAL_EPOCHS` | 200 | Max training epochs |
| `FINAL_LR` | 1e-4 | Learning rate (with cosine warmup) |
| `WARMUP_EPOCHS` | 10 | Linear LR warmup duration |
| `MIXUP_ALPHA` | 0.2 | Mixup augmentation strength |
| `LABEL_SMOOTHING` | 0.1 | Cross-entropy label smoothing |
| `USE_AMP` | True | Mixed precision training (FP16) |
| `USE_SWA` | True | Stochastic Weight Averaging |
| `SWA_START_EPOCH` | 140 | Epoch to begin SWA averaging |
| `TTA_ENABLED` | True | Test-Time Augmentation (7-pass) |
| `COMPILE_MODEL` | True | torch.compile() for faster kernels |
| `EARLY_STOP_PATIENCE` | 25 | Early stopping patience (disabled by default) |

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
