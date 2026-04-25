import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             matthews_corrcoef, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as st
from sklearn.metrics import balanced_accuracy_score

from config import Config
from models.dit_classifier import DiTClassifier


# CHANGE 1: Temperature Scaling for calibration
class TemperatureScaling(nn.Module):
    """Post-hoc temperature scaling for model calibration.
    
    Learns a single scalar temperature parameter that divides the logits
    before softmax, improving probability calibration without affecting accuracy.
    """
    def __init__(self, init_temp=1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_temp)

    def forward(self, logits):
        return logits / self.temperature


def compute_ece(probs, targets, n_bins=15):
    """Compute Expected Calibration Error with given number of bins."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_accs = []
    bin_confs = []
    bin_counts = []
    
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (probs > lo) & (probs <= hi) if i > 0 else (probs >= lo) & (probs <= hi)
        count = in_bin.sum()
        if count > 0:
            bin_acc = targets[in_bin].mean()
            bin_conf = probs[in_bin].mean()
            ece += (count / len(probs)) * abs(bin_acc - bin_conf)
            bin_accs.append(bin_acc)
            bin_confs.append(bin_conf)
            bin_counts.append(count)
        else:
            bin_accs.append(0)
            bin_confs.append(0)
            bin_counts.append(0)
    
    bin_midpoints = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    return ece, bin_accs, bin_confs, bin_counts, bin_midpoints


def fit_temperature(model, val_loader, device, use_amp):
    """Fit temperature scaling on validation set using LBFGS."""
    temp_model = TemperatureScaling(init_temp=1.5).to(device)
    nll_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([temp_model.temperature], lr=0.01, max_iter=50)
    
    # Collect all validation logits
    all_logits = []
    all_labels = []
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device, non_blocking=True)
            logits = model(x_batch)
            all_logits.append(logits.float())  # ensure float32 for LBFGS
            all_labels.append(y_batch.to(device))
    
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Fit temperature
    def closure():
        optimizer.zero_grad()
        scaled_logits = temp_model(all_logits)
        loss = nll_criterion(scaled_logits, all_labels)
        loss.backward()
        return loss
    
    optimizer.step(closure)
    
    final_temp = temp_model.temperature.item()
    print(f"Temperature scaling fitted: T = {final_temp:.4f}")
    return temp_model


def maybe_compile_model(model: torch.nn.Module, device: str) -> torch.nn.Module:
    """Compile model when supported; otherwise safely fall back to eager."""
    if not Config.COMPILE_MODEL:
        return model
    if device != "cuda":
        print("Skipping torch.compile(): non-CUDA device.")
        return model
    if os.name == "nt":
        print("Skipping torch.compile() on Windows; using eager mode.")
        return model
    try:
        import triton  # noqa: F401
    except Exception:
        print("Skipping torch.compile(): Triton is missing or incompatible.")
        return model
    try:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        return torch.compile(model)
    except Exception as e:
        print(f"torch.compile() failed; falling back to eager mode. Reason: {e}")
        return model


def tta_predict(model, x, n_aug, noise_std, device, use_amp, temp_model=None):
    """
    Test Time Augmentation over feature vectors.
    Runs n_aug forward passes with small Gaussian noise,
    averages softmax probabilities for a more stable prediction.
    Applies temperature scaling if temp_model is provided.
    """
    accumulated = None
    x = x.to(device, non_blocking=True)

    for _ in range(n_aug):
        noise = torch.randn_like(x) * noise_std
        x_aug = x + noise
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
            logits = model(x_aug)
            if temp_model is not None:
                logits = temp_model(logits.float())
            probs = torch.softmax(logits, dim=1).cpu()
        accumulated = probs if accumulated is None else accumulated + probs

    return accumulated / n_aug


def collect_probs(model, loader, device, use_amp, temp_model=None):
    all_probs = []
    all_targets = []
    if Config.TTA_ENABLED:
        for x_batch, y_batch in loader:
            avg_probs = tta_predict(
                model, x_batch,
                n_aug=Config.TTA_N_AUG,
                noise_std=Config.TTA_NOISE_STD,
                device=device,
                use_amp=use_amp,
                temp_model=temp_model
            )
            all_probs.extend(avg_probs[:, 1].numpy())
            all_targets.extend(y_batch.numpy())
    else:
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(device, non_blocking=True)
                logits = model(x_batch)
                if temp_model is not None:
                    logits = temp_model(logits.float())
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                all_probs.extend(probs)
                all_targets.extend(y_batch.numpy())
    return np.array(all_probs), np.array(all_targets)


def tune_threshold(val_probs: np.ndarray, val_targets: np.ndarray) -> float:
    candidates = np.linspace(0.1, 0.9, 81)
    best_thr = Config.DEFAULT_DECISION_THRESHOLD
    best_score = -1.0

    for thr in candidates:
        preds = (val_probs >= thr).astype(np.int64)
        if Config.THRESHOLD_OBJECTIVE == "balanced_acc":
            score = balanced_accuracy_score(val_targets, preds)
        else:
            score = f1_score(val_targets, preds)
        if score > best_score:
            best_score = score
            best_thr = float(thr)
    return best_thr


def evaluate(ablation=None):
    print("========== Starting Evaluation ==========")
    device = Config.DEVICE

    # Resolve log dir — fallback to base if called standalone
    log_dir = Config.LOG_DIR if Config.LOG_DIR != "./logs" else "./logs/full_run/"
    os.makedirs(log_dir, exist_ok=True)
    if ablation == 'dit_only':
        run_name = "ablation_dit_only"
    elif ablation == 'boa_only':
        run_name = "ablation_boa_only"
    elif ablation == 'woa_only':
        run_name = "ablation_woa_only"
    elif "ablation_boa_only" in log_dir:
        run_name = "ablation_boa_only"
    elif "ablation_woa_only" in log_dir:
        run_name = "ablation_woa_only"
    elif "ablation_dit_only" in log_dir:
        run_name = "ablation_dit_only"
    else:
        run_name = "full_run"

    # 1. Load test features and optimal mask
    test_feat_path = os.path.join(Config.CACHE_DIR, "features_test.npy")
    test_lbl_path  = os.path.join(Config.CACHE_DIR, "labels_test.npy")
    mask_path      = os.path.join(Config.CACHE_DIR, f"optimal_mask_{run_name}.npy")

    if not os.path.exists(mask_path):
        fallback_mask = os.path.join(Config.CACHE_DIR, "optimal_mask.npy")
        if os.path.exists(fallback_mask):
            mask_path = fallback_mask

    if not os.path.exists(test_feat_path) or not os.path.exists(mask_path):
        print("Test features or optimal mask not found. Run train.py first.")
        return

    features_test = np.load(test_feat_path)
    labels_test   = np.load(test_lbl_path)
    optimal_mask  = np.load(mask_path)

    print(f"Loaded {features_test.shape[0]} test samples.")

    # Apply mask
    masked_test = features_test * optimal_mask

    # 2. DataLoader
    test_dataset = TensorDataset(
        torch.tensor(masked_test, dtype=torch.float32),
        torch.tensor(labels_test, dtype=torch.long)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.FINAL_BATCH_SIZE,
        shuffle=False,
        pin_memory=Config.PIN_MEMORY,
        num_workers=Config.FINAL_NUM_WORKERS,
        persistent_workers=Config.PERSISTENT_WORKERS and Config.FINAL_NUM_WORKERS > 0,
        prefetch_factor=Config.FINAL_PREFETCH_FACTOR if Config.FINAL_NUM_WORKERS > 0 else None
    )

    # 3. Load Model — CHANGE 4: Prioritize SWA checkpoint
    run_ckpt_dir = os.path.join(Config.CHECKPOINT_DIR, run_name)
    swa_top_level = os.path.join(Config.CHECKPOINT_DIR, "swa_dit_model.pth")
    preferred_checkpoints = [
        os.path.join(run_ckpt_dir, "best_dit_model_swa.pth"),  # SWA from run dir
        swa_top_level,                                          # SWA from top-level
        os.path.join(run_ckpt_dir, "best_dit_model_acc.pth"),
        os.path.join(run_ckpt_dir, "best_dit_model_loss.pth"),
        os.path.join(Config.CHECKPOINT_DIR, "best_dit_model.pth")
    ]
    model_path = None
    is_swa_checkpoint = False
    for cp in preferred_checkpoints:
        if os.path.exists(cp):
            model_path = cp
            is_swa_checkpoint = 'swa' in os.path.basename(cp).lower()
            break
    if model_path is None or not os.path.exists(model_path):
        print("Best model checkpoint not found. Run train.py first.")
        return

    model = DiTClassifier(
        feature_dim=Config.FEATURE_DIM,
        hidden_dim=Config.FINAL_DIT_HIDDEN_DIM,
        depth=Config.FINAL_DIT_DEPTH,
        num_heads=Config.FINAL_DIT_HEADS,
        mlp_ratio=Config.FINAL_DIT_MLP_RATIO,
        dropout=0.0  # No dropout during evaluation
    ).to(device)

    # Handle SWA checkpoint (wrapped in AveragedModel)
    if is_swa_checkpoint:
        from torch.optim.swa_utils import AveragedModel
        swa_wrapper = AveragedModel(model)
        swa_wrapper.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model = swa_wrapper.module
        print(f"Loaded SWA checkpoint: {model_path}")
    else:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"Loaded checkpoint: {model_path}")
    model.eval()
    model = maybe_compile_model(model, device)

    # Optional threshold tuning on validation probabilities.
    decision_threshold = Config.DEFAULT_DECISION_THRESHOLD
    temp_model = None  # will be set if temperature scaling succeeds

    # CHANGE 1: Load val set for temperature scaling + threshold tuning
    val_feat_path = os.path.join(Config.CACHE_DIR, "features_val.npy")
    val_lbl_path = os.path.join(Config.CACHE_DIR, "labels_val.npy")
    if os.path.exists(val_feat_path) and os.path.exists(val_lbl_path):
        features_val = np.load(val_feat_path)
        labels_val = np.load(val_lbl_path)
        masked_val = features_val * optimal_mask
        val_dataset = TensorDataset(
            torch.tensor(masked_val, dtype=torch.float32),
            torch.tensor(labels_val, dtype=torch.long)
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.FINAL_BATCH_SIZE,
            shuffle=False,
            pin_memory=Config.PIN_MEMORY,
            num_workers=Config.FINAL_NUM_WORKERS,
            persistent_workers=Config.PERSISTENT_WORKERS and Config.FINAL_NUM_WORKERS > 0,
            prefetch_factor=Config.FINAL_PREFETCH_FACTOR if Config.FINAL_NUM_WORKERS > 0 else None
        )

        # Collect pre-calibration probs for ECE "before"
        probs_before, targets_before = collect_probs(model, val_loader, device, Config.USE_AMP, temp_model=None)
        ece_before, _, _, _, _ = compute_ece(probs_before, targets_before, n_bins=15)
        print(f"ECE before temperature scaling: {ece_before:.4f}")

        # Fit temperature scaling
        temp_model = fit_temperature(model, val_loader, device, Config.USE_AMP)

        # Collect post-calibration probs for ECE "after"
        probs_after, targets_after = collect_probs(model, val_loader, device, Config.USE_AMP, temp_model=temp_model)
        ece_after, _, _, _, _ = compute_ece(probs_after, targets_after, n_bins=15)
        print(f"ECE after temperature scaling:  {ece_after:.4f}")

        # Threshold tuning (with temp-scaled probs)
        if Config.TUNE_THRESHOLD_ON_VAL:
            val_probs_for_thr, val_targets_for_thr = collect_probs(model, val_loader, device, Config.USE_AMP, temp_model=temp_model)
            decision_threshold = tune_threshold(val_probs_for_thr, val_targets_for_thr)
            print(f"Threshold tuned on validation: {decision_threshold:.3f} ({Config.THRESHOLD_OBJECTIVE})")

    # 4. Predict probabilities with temperature scaling applied.
    all_probs, all_targets = collect_probs(model, test_loader, device, Config.USE_AMP, temp_model=temp_model)
    all_preds = (all_probs >= decision_threshold).astype(np.int64)

    # 5. Metrics
    acc       = accuracy_score(all_targets, all_preds) * 100
    precision = precision_score(all_targets, all_preds)
    recall    = recall_score(all_targets, all_preds)
    f1        = f1_score(all_targets, all_preds)
    auc       = roc_auc_score(all_targets, all_probs)
    mcc       = matthews_corrcoef(all_targets, all_preds)

    TN, FP, FN, TP = confusion_matrix(all_targets, all_preds).ravel()
    specificity = TN / (TN + FP)

    n = len(all_targets)
    acc_raw = accuracy_score(all_targets, all_preds)
    ci_low, ci_high = st.binom.interval(0.95, n, acc_raw)

    # Save per-sample outputs for downstream ablation comparison graphs.
    pred_df = pd.DataFrame({
        "target": np.array(all_targets, dtype=np.int64),
        "pred": np.array(all_preds, dtype=np.int64),
        "prob": np.array(all_probs, dtype=np.float64)
    })
    predictions_path = os.path.join(log_dir, "predictions.csv")
    pred_df.to_csv(predictions_path, index=False)

    print("\n========== Test Set Metrics ==========")
    print(f"Accuracy:    {acc:.2f}%")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"F1-Score:    {f1:.4f}")
    print(f"AUC-ROC:     {auc:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"MCC:         {mcc:.4f}")
    print(f"95% CI:      ({ci_low/n:.4f}, {ci_high/n:.4f})")
    print(f"Threshold:   {decision_threshold:.3f}")
    print(f"TTA:         {'Enabled (' + str(Config.TTA_N_AUG) + ' passes)' if Config.TTA_ENABLED else 'Disabled'}")

    # Save results.txt (with ECE before/after)
    results_path = os.path.join(log_dir, "results.txt")
    with open(results_path, "w") as f:
        f.write(f"Accuracy:    {acc:.2f}%\n")
        f.write(f"Precision:   {precision:.4f}\n")
        f.write(f"Recall:      {recall:.4f}\n")
        f.write(f"F1-Score:    {f1:.4f}\n")
        f.write(f"AUC-ROC:     {auc:.4f}\n")
        f.write(f"Specificity: {specificity:.4f}\n")
        f.write(f"MCC:         {mcc:.4f}\n")
        f.write(f"95% CI:      ({ci_low/n:.4f}, {ci_high/n:.4f})\n")
        f.write(f"Threshold:   {decision_threshold:.3f}\n")
        f.write(f"TTA:         {'Enabled (' + str(Config.TTA_N_AUG) + ' passes)' if Config.TTA_ENABLED else 'Disabled'}\n")
        if temp_model is not None:
            f.write(f"Temperature: {temp_model.temperature.item():.4f}\n")
            f.write(f"ECE_before:  {ece_before:.4f}\n")
            f.write(f"ECE_after:   {ece_after:.4f}\n")

    # 6. Comparison Table
    print("\n========== Comparison with Previous Works ==========")
    data = [
        ["AlexNet",                    "89.90%"],
        ["CNN Global Attention",        "96.50%"],
        ["Multi-CNN+CCA",               "99.10%"],
        ["EfficientNet+Transformer",    "99.87%"],
        ["Hybrid BOA-WOA + DiT (Ours)", f"{acc:.2f}%"]
    ]
    df = pd.DataFrame(data, columns=["Model", "Accuracy"])
    print(df.to_string(index=False))

    # 7. Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Adenocarcinoma'],
                yticklabels=['Benign', 'Adenocarcinoma'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    cm_path = os.path.join(log_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()   # ← was missing, leaked into ROC plot

    # 8. ROC Curve
    fpr, tpr, _ = roc_curve(all_targets, all_probs)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color='#e6a532', lw=2,
             label=f'Hybrid BOA-WOA + DiT (AUC≈{auc:.3f})')
    plt.plot([0, 1], [0, 1], color='#1f77b4', lw=1.5, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc="lower right", fontsize=11)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    roc_path = os.path.join(log_dir, "roc_curve.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()

    print(f"\nSaved confusion matrix to {cm_path}")
    print(f"Saved ROC curve to {roc_path}")
    print(f"Saved results to {results_path}")
    print(f"Saved per-sample predictions to {predictions_path}")

    # CHANGE 6: ECE Calibration Plot (before/after temperature scaling)
    if temp_model is not None:
        fig, ax = plt.subplots(figsize=(8, 6))
        n_bins = 15
        
        # Before temp scaling
        ece_b, bin_accs_b, bin_confs_b, bin_counts_b, bin_mids = compute_ece(
            probs_before, targets_before, n_bins=n_bins)
        # After temp scaling  
        ece_a, bin_accs_a, bin_confs_a, bin_counts_a, _ = compute_ece(
            probs_after, targets_after, n_bins=n_bins)
        
        # Perfect diagonal
        ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfect calibration', alpha=0.6)
        
        # Shaded gap from diagonal for "after" curve
        ax.fill_between(bin_mids, bin_mids, bin_accs_a, alpha=0.15, color='#2ca02c', label='Calibration gap (after)')
        
        # Before curve (dashed)
        ax.plot(bin_mids, bin_accs_b, 'o--', color='#d62728', lw=2, markersize=6,
                label=f'Before T-scaling (ECE={ece_b:.4f})')
        
        # After curve (solid)
        ax.plot(bin_mids, bin_accs_a, 's-', color='#2ca02c', lw=2, markersize=6,
                label=f'After T-scaling (ECE={ece_a:.4f})')
        
        ax.set_xlabel('Mean Predicted Confidence', fontsize=12)
        ax.set_ylabel('Fraction of Positives (Accuracy)', fontsize=12)
        ax.set_title(f'Calibration Plot (T={temp_model.temperature.item():.3f})', fontsize=14)
        ax.legend(loc='upper left', fontsize=10)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        cal_path = os.path.join(log_dir, "calibration_plot.png")
        plt.savefig(cal_path, dpi=300)
        plt.close()
        print(f"Saved calibration plot to {cal_path}")

    print("==========================================")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ablation', type=str, default=None,
                        choices=['dit_only', 'boa_only', 'woa_only'],
                        help='Evaluate a specific ablation variant.')
    args = parser.parse_args()
    evaluate(ablation=args.ablation)