import os
import logging
import numpy as np
import torch
torch.set_num_threads(64)                   # Match 64 physical cores
torch.backends.cudnn.benchmark = True        # Auto-tune CUDA kernels
torch.backends.cudnn.deterministic = False   # Allow non-deterministic for speed
torch.set_float32_matmul_precision("high")
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import random

from config import Config
from feature_extraction.extractor import extract_and_save_features
from optimization.hybrid_boa_woa import run_hybrid_boa_woa
from models.dit_classifier import DiTClassifier

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(Config.LOG_DIR, "training.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }

    def update(self, model: torch.nn.Module):
        with torch.no_grad():
            current = model.state_dict()
            for k, v in current.items():
                if k in self.shadow:
                    self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def state_dict(self, model: torch.nn.Module):
        merged = model.state_dict()
        for k, v in self.shadow.items():
            merged[k] = v
        return merged


def set_reproducible_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def maybe_compile_model(model: torch.nn.Module, device: str) -> torch.nn.Module:
    """Compile model when supported; otherwise safely fall back to eager."""
    if not Config.COMPILE_MODEL:
        return model

    if device != "cuda":
        logger.info("Skipping torch.compile(): non-CUDA device.")
        return model

    # On Windows, torch.compile with inductor commonly fails due to Triton constraints.
    if os.name == "nt":
        logger.warning("Skipping torch.compile() on Windows; using eager mode.")
        return model

    try:
        import triton  # noqa: F401
    except Exception:
        logger.warning("Skipping torch.compile(): Triton is missing or incompatible.")
        return model

    try:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        logger.info("Compiling model with torch.compile()...")
        return torch.compile(model)
    except Exception as e:
        logger.warning(f"torch.compile() failed; falling back to eager mode. Reason: {e}")
        return model


def resolve_mask_paths(run_name: str):
    run_mask = os.path.join(Config.CACHE_DIR, f"optimal_mask_{run_name}.npy")
    legacy_mask = os.path.join(Config.CACHE_DIR, "optimal_mask.npy")
    return run_mask, legacy_mask

def main(ablation=None):
    set_reproducible_seed(Config.SEED)

    if Config.ALLOW_TF32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision(Config.MATMUL_PRECISION)

    if ablation == 'dit_only':
        Config.LOG_DIR = "./logs/ablation_dit_only/"
        run_name = "ablation_dit_only"
    elif ablation == 'boa_only':
        Config.LOG_DIR = "./logs/ablation_boa_only/"
        run_name = "ablation_boa_only"
    elif ablation == 'woa_only':
        Config.LOG_DIR = "./logs/ablation_woa_only/"
        run_name = "ablation_woa_only"
    else:
        Config.LOG_DIR = "./logs/full_run/"
        run_name = "full_run"
        
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(Config.LOG_DIR, "training.log")),
            logging.StreamHandler()
        ]
    )
    
    logger.info("========== Starting ML Pipeline ==========")
    logger.info(f"Hardware: {torch.cuda.get_device_name(0)} | "
                f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB | "
                f"CPU threads: {torch.get_num_threads()}")
    device = Config.DEVICE
    logger.info(f"Using device: {device}")

    run_ckpt_dir = os.path.join(Config.CHECKPOINT_DIR, run_name)
    os.makedirs(run_ckpt_dir, exist_ok=True)
    
    # Step 1: Extract and cache features
    logger.info("--- Step 1: Feature Extraction ---")
    extract_and_save_features(Config.TRAIN_EVAL_DIR, Config.TEST_DIR, Config.CACHE_DIR, Config.BATCH_SIZE, device)
    
    # Load features
    logger.info("Loading cached features...")
    features_train = np.load(os.path.join(Config.CACHE_DIR, "features_train.npy"))
    labels_train = np.load(os.path.join(Config.CACHE_DIR, "labels_train.npy"))
    
    features_val = np.load(os.path.join(Config.CACHE_DIR, "features_val.npy"))
    labels_val = np.load(os.path.join(Config.CACHE_DIR, "labels_val.npy"))
    
    features_test = np.load(os.path.join(Config.CACHE_DIR, "features_test.npy"))
    labels_test = np.load(os.path.join(Config.CACHE_DIR, "labels_test.npy"))
    
    logger.info(f"Train features shape: {features_train.shape}")
    logger.info(f"Val features shape: {features_val.shape}")
    logger.info(f"Test features shape: {features_test.shape}")
    
    # Step 2: Run hybrid BOA-WOA
    run_mask_path, legacy_mask_path = resolve_mask_paths(run_name)

    if ablation == 'dit_only':
        logger.info("--- Step 2: Skipping BOA-WOA (Ablation: dit_only) ---")
        optimal_mask = np.ones(Config.FEATURE_DIM)
        np.save(run_mask_path, optimal_mask)
    else:
        logger.info(f"--- Step 2: Hybrid BOA-WOA Feature Selection (Ablation: {ablation}) ---")

        cached_mask_path = None
        if os.path.exists(run_mask_path):
            cached_mask_path = run_mask_path
        elif os.path.exists(legacy_mask_path):
            cached_mask_path = legacy_mask_path

        if Config.FORCE_STEP2_RECOMPUTE:
            logger.info("FORCE_STEP2_RECOMPUTE=True, recomputing BOA-WOA mask.")
            optimal_mask = run_hybrid_boa_woa(
                features_train, labels_train, features_val, labels_val, device, ablation_mode=ablation
            )
            np.save(run_mask_path, optimal_mask)
        elif cached_mask_path is not None and Config.SKIP_STEP2_IF_MASK_EXISTS:
            logger.info(f"Cached mask found at {cached_mask_path}, skipping BOA-WOA optimization.")
            optimal_mask = np.load(cached_mask_path)
            if cached_mask_path != run_mask_path:
                np.save(run_mask_path, optimal_mask)
                logger.info(f"Copied cached mask to run-specific path: {run_mask_path}")
        else:
            logger.info("No cached mask found, starting optimization algorithm...")
            optimal_mask = run_hybrid_boa_woa(features_train, labels_train, features_val, labels_val, device, ablation_mode=ablation)
            np.save(run_mask_path, optimal_mask)
            
    num_selected = int(np.sum(optimal_mask))
    logger.info(f"Optimal mask selected {num_selected}/{Config.FEATURE_DIM} features.")
    
    # Step 3: Apply mask
    logger.info("--- Step 3: Applying Mask ---")
    masked_train = features_train * optimal_mask
    masked_val = features_val * optimal_mask
    masked_test = features_test * optimal_mask
    
    # Step 4: Train Full DiT Classifier
    logger.info("--- Step 4: Training Final DiT Classifier ---")
    logger.info(f"Config: hidden={Config.FINAL_DIT_HIDDEN_DIM}, depth={Config.FINAL_DIT_DEPTH}, "
                f"heads={Config.FINAL_DIT_HEADS}, batch={Config.FINAL_BATCH_SIZE}, "
                f"lr={Config.FINAL_LR}, epochs={Config.FINAL_EPOCHS}, AMP={Config.USE_AMP}")
    
    train_dataset = TensorDataset(torch.tensor(masked_train, dtype=torch.float32), 
                                  torch.tensor(labels_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(masked_val, dtype=torch.float32), 
                                torch.tensor(labels_val, dtype=torch.long))
                                
    train_loader = DataLoader(train_dataset, batch_size=Config.FINAL_BATCH_SIZE, shuffle=True,
                                                            pin_memory=Config.PIN_MEMORY, num_workers=Config.FINAL_NUM_WORKERS,
                                                            persistent_workers=Config.PERSISTENT_WORKERS and Config.FINAL_NUM_WORKERS > 0,
                                                            prefetch_factor=Config.FINAL_PREFETCH_FACTOR if Config.FINAL_NUM_WORKERS > 0 else None,
                                                            drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=Config.FINAL_BATCH_SIZE, shuffle=False,
                                                        pin_memory=Config.PIN_MEMORY, num_workers=Config.FINAL_NUM_WORKERS,
                                                        persistent_workers=Config.PERSISTENT_WORKERS and Config.FINAL_NUM_WORKERS > 0,
                                                        prefetch_factor=Config.FINAL_PREFETCH_FACTOR if Config.FINAL_NUM_WORKERS > 0 else None)
    
    model = DiTClassifier(
        feature_dim=Config.FEATURE_DIM, # keeping fixed 3840 since zero-masked
        hidden_dim=Config.FINAL_DIT_HIDDEN_DIM,
        depth=Config.FINAL_DIT_DEPTH,
        num_heads=Config.FINAL_DIT_HEADS,
        mlp_ratio=Config.FINAL_DIT_MLP_RATIO,
        dropout=Config.FINAL_DROPOUT
    ).to(device)
    
    # Compile only when backend/toolchain support is available.
    model = maybe_compile_model(model, device)
    
    # Log model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
    
    # AdamW with weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr=Config.FINAL_LR, weight_decay=Config.WEIGHT_DECAY)
    
    # Cosine Annealing LR with warmup
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, 
                                 total_iters=Config.WARMUP_EPOCHS)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=Config.FINAL_EPOCHS - Config.WARMUP_EPOCHS, 
                                         eta_min=Config.LR_ETA_MIN)
    swa_model = AveragedModel(model._orig_mod if hasattr(model, '_orig_mod') else model) if Config.USE_SWA else None
    swa_scheduler = SWALR(optimizer, swa_lr=Config.SWA_LR) if Config.USE_SWA else None
    ema = EMA(model._orig_mod if hasattr(model, '_orig_mod') else model, Config.EMA_DECAY) if Config.USE_EMA else None
    warmup_epochs = Config.WARMUP_EPOCHS

    # CHANGE 2: Warmup sanity assertion
    assert Config.WARMUP_EPOCHS < Config.FINAL_EPOCHS * 0.15, \
        f"WARMUP_EPOCHS ({Config.WARMUP_EPOCHS}) must be < 15% of FINAL_EPOCHS ({Config.FINAL_EPOCHS})"
    
    # AMP GradScaler
    scaler = torch.amp.GradScaler('cuda', enabled=Config.USE_AMP)
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_acc_for_checkpoint = 0.0
    best_model_loss_path = os.path.join(run_ckpt_dir, "best_dit_model_loss.pth")
    best_model_acc_path = os.path.join(run_ckpt_dir, "best_dit_model_acc.pth")
    latest_model_path = os.path.join(run_ckpt_dir, "latest_dit_model.pth")
    if run_name == "full_run":
        best_model_default_path = os.path.join(Config.CHECKPOINT_DIR, "best_dit_model.pth")
    else:
        best_model_default_path = os.path.join(Config.CHECKPOINT_DIR, f"best_dit_model_{run_name}.pth")
    patience_counter = 0

    if not Config.ENABLE_EARLY_STOPPING:
        logger.info("Early stopping is disabled; training will run for all configured epochs.")
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    lr_history = []
    
    # CHANGE 5: Mixup helper functions
    def mixup_batch(x, y, alpha):
        """Apply mixup augmentation to a batch."""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(criterion_fn, pred, y_a, y_b, lam):
        """Compute mixup loss using both label sets."""
        return lam * criterion_fn(pred, y_a) + (1 - lam) * criterion_fn(pred, y_b)

    for epoch in range(Config.FINAL_EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
            
            # CHANGE 5: Mixup augmentation
            if Config.MIXUP_ALPHA > 0:
                mixed_x, y_a, y_b, lam = mixup_batch(x_batch, y_batch, Config.MIXUP_ALPHA)
            else:
                mixed_x, y_a, y_b, lam = x_batch, y_batch, y_batch, 1.0

            # AMP autocast for mixed precision
            with torch.amp.autocast('cuda', enabled=Config.USE_AMP):
                logits = model(mixed_x)
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            
            # Scaled backward pass
            scaler.scale(loss).backward()
            
            # Gradient clipping for stability
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRADIENT_CLIP)
            
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model._orig_mod if hasattr(model, '_orig_mod') else model)
            
            running_loss += loss.item() * x_batch.size(0)
            # Use original labels for accuracy (not mixup labels)
            preds = torch.argmax(logits, dim=1)
            correct_train += (preds == y_batch).sum().item()
            total_train += x_batch.size(0)
            
        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train
        
        # Step schedulers explicitly to avoid SequentialLR epoch deprecation warnings.
        if Config.USE_SWA and epoch >= Config.SWA_START_EPOCH:
            swa_scheduler.step()
        elif epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=Config.USE_AMP):
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                
                val_loss += loss.item() * x_batch.size(0)
                preds = torch.argmax(logits, dim=1)
                correct_val += (preds == y_batch).sum().item()
                total_val += x_batch.size(0)
                
        epoch_val_loss = val_loss / total_val
        epoch_val_acc = correct_val / total_val
        
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accs.append(epoch_train_acc)
        val_accs.append(epoch_val_acc)
        
        logger.info(f"Epoch [{epoch+1}/{Config.FINAL_EPOCHS}] - "
                     f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
                     f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f} | "
                     f"LR: {current_lr:.6f}")

        if Config.USE_SWA and epoch >= Config.SWA_START_EPOCH:
            swa_model.update_parameters(model._orig_mod if hasattr(model, '_orig_mod') else model)
        
        # CHANGE 3: Save best model (primary gate = val accuracy, track loss for logging)
        acc_improved = epoch_val_acc > (best_acc_for_checkpoint + Config.CHECKPOINT_MIN_DELTA)
        if acc_improved:
            best_acc_for_checkpoint = epoch_val_acc
            best_val_acc = epoch_val_acc
            # Handle compiled model state dict
            model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
            state = ema.state_dict(model_to_save) if ema is not None else model_to_save.state_dict()
            torch.save(state, best_model_acc_path)
            torch.save(state, best_model_default_path)
            logger.info(f"--> Saved new best model with Val Acc: {best_val_acc:.4f}, Val Loss: {epoch_val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

        # Also track best val loss for logging only
        if epoch_val_loss < (best_val_loss - Config.CHECKPOINT_MIN_DELTA):
            best_val_loss = epoch_val_loss
            model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
            state = ema.state_dict(model_to_save) if ema is not None else model_to_save.state_dict()
            torch.save(state, best_model_loss_path)
            logger.info(f"     Also new best-loss checkpoint: Val Loss: {best_val_loss:.4f}")

        model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
        state = ema.state_dict(model_to_save) if ema is not None else model_to_save.state_dict()
        torch.save(state, latest_model_path)
            
        # Early stopping (based on val accuracy improvement)
        if Config.ENABLE_EARLY_STOPPING and patience_counter >= Config.EARLY_STOP_PATIENCE:
            logger.info(f"Early stopping triggered at epoch {epoch+1} (no val acc improvement for {Config.EARLY_STOP_PATIENCE} epochs)")
            break
            
    # CHANGE 7: Enhanced Training Curves (log scale, warmup line, rolling avg, annotations)
    logger.info("Saving training curves...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Rolling average helper
    def rolling_avg(data, window=5):
        if len(data) < window:
            return data
        kernel = np.ones(window) / window
        smoothed = np.convolve(data, kernel, mode='valid')
        # Pad front to align with original epoch indices
        pad = np.array(data[:window-1])
        return np.concatenate([pad, smoothed])

    epochs_axis = np.arange(1, len(train_losses) + 1)
    
    # --- Loss subplot (log scale) ---
    smooth_train_loss = rolling_avg(train_losses)
    smooth_val_loss = rolling_avg(val_losses)
    axes[0].plot(epochs_axis, train_losses, alpha=0.25, color='#1f77b4', linewidth=0.8)
    axes[0].plot(epochs_axis, val_losses, alpha=0.25, color='#ff7f0e', linewidth=0.8)
    axes[0].plot(epochs_axis, smooth_train_loss, label='Train Loss (5-ep avg)', color='#1f77b4', linewidth=2)
    axes[0].plot(epochs_axis, smooth_val_loss, label='Val Loss (5-ep avg)', color='#ff7f0e', linewidth=2)
    axes[0].axvline(x=Config.WARMUP_EPOCHS, color='red', linestyle='--', alpha=0.7, label=f'Warmup end (ep {Config.WARMUP_EPOCHS})')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (log scale)')
    axes[0].set_title('Loss Curve')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3, which='both')
    
    # --- Accuracy subplot ---
    smooth_train_acc = rolling_avg(train_accs)
    smooth_val_acc = rolling_avg(val_accs)
    axes[1].plot(epochs_axis, train_accs, alpha=0.25, color='#1f77b4', linewidth=0.8)
    axes[1].plot(epochs_axis, val_accs, alpha=0.25, color='#ff7f0e', linewidth=0.8)
    axes[1].plot(epochs_axis, smooth_train_acc, label='Train Acc (5-ep avg)', color='#1f77b4', linewidth=2)
    axes[1].plot(epochs_axis, smooth_val_acc, label='Val Acc (5-ep avg)', color='#ff7f0e', linewidth=2)
    axes[1].axvline(x=Config.WARMUP_EPOCHS, color='red', linestyle='--', alpha=0.7, label=f'Warmup end (ep {Config.WARMUP_EPOCHS})')
    # Annotate final val accuracy
    final_val_acc = val_accs[-1] if val_accs else 0
    axes[1].annotate(f'Final: {final_val_acc:.4f}', xy=(len(val_accs), final_val_acc),
                     xytext=(-80, -25), textcoords='offset points',
                     fontsize=9, fontweight='bold', color='#ff7f0e',
                     arrowprops=dict(arrowstyle='->', color='#ff7f0e', lw=1.2))
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curve')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    # --- LR subplot ---
    axes[2].plot(epochs_axis, lr_history, label='Learning Rate', color='#2ca02c', linewidth=1.5)
    axes[2].axvline(x=Config.WARMUP_EPOCHS, color='red', linestyle='--', alpha=0.7, label=f'Warmup end (ep {Config.WARMUP_EPOCHS})')
    if Config.USE_SWA:
        axes[2].axvline(x=Config.SWA_START_EPOCH, color='purple', linestyle=':', alpha=0.7, label=f'SWA start (ep {Config.SWA_START_EPOCH})')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('LR')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.LOG_DIR, "training_curves.png"), dpi=300)
    plt.close()
    
    # CHANGE 4: Save SWA model
    if Config.USE_SWA and Config.FINAL_EPOCHS > Config.SWA_START_EPOCH:
        logger.info("Updating batchnorm statistics for SWA model...")
        update_bn(train_loader, swa_model, device=device)
        torch.save(swa_model.state_dict(), os.path.join(run_ckpt_dir, "best_dit_model_swa.pth"))
        # Also save as swa_dit_model.pth at top-level checkpoint dir for evaluate.py
        torch.save(swa_model.state_dict(), os.path.join(Config.CHECKPOINT_DIR, "swa_dit_model.pth"))
        logger.info("Saved SWA model checkpoint.")

    logger.info(f"Best Val Loss: {best_val_loss:.4f}, Best Val Acc: {best_val_acc:.4f}")
    logger.info("========== Pipeline Completed ==========")
    from evaluate import evaluate
    evaluate(ablation=ablation)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ablation', type=str, default=None,
                        choices=['dit_only', 'boa_only', 'woa_only'],
                        help='Run ablation variant instead of full pipeline')
    args = parser.parse_args()
    main(args.ablation)
