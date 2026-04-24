import os
import logging
import numpy as np
import torch
torch.set_num_threads(64)                   # Match 64 physical cores
torch.backends.cudnn.benchmark = True        # Auto-tune CUDA kernels
torch.backends.cudnn.deterministic = False   # Allow non-deterministic for speed
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
import matplotlib.pyplot as plt

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

def main(ablation=None):
    if ablation == 'dit_only':
        Config.LOG_DIR = "./logs/ablation_dit_only/"
    elif ablation == 'boa_only':
        Config.LOG_DIR = "./logs/ablation_boa_only/"
    elif ablation == 'woa_only':
        Config.LOG_DIR = "./logs/ablation_woa_only/"
    else:
        Config.LOG_DIR = "./logs/full_run/"
        
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
                f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB | "
                f"CPU threads: {torch.get_num_threads()}")
    device = Config.DEVICE
    logger.info(f"Using device: {device}")
    
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
    if ablation == 'dit_only':
        logger.info("--- Step 2: Skipping BOA-WOA (Ablation: dit_only) ---")
        optimal_mask = np.ones(Config.FEATURE_DIM)
        mask_path = os.path.join(Config.CACHE_DIR, "optimal_mask.npy")
        np.save(mask_path, optimal_mask)
    else:
        logger.info(f"--- Step 2: Hybrid BOA-WOA Feature Selection (Ablation: {ablation}) ---")
        
        mask_path = os.path.join(Config.CACHE_DIR, "optimal_mask.npy")
        if os.path.exists(mask_path):
            logger.info("Optimal mask found in cache, loading...")
            optimal_mask = np.load(mask_path)
        else:
            logger.info("No cached mask found, starting optimization algorithm...")
            optimal_mask = run_hybrid_boa_woa(features_train, labels_train, features_val, labels_val, device, ablation_mode=ablation)
            np.save(mask_path, optimal_mask)
            
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
                              pin_memory=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.FINAL_BATCH_SIZE, shuffle=False,
                            pin_memory=True, num_workers=4)
    
    model = DiTClassifier(
        feature_dim=Config.FEATURE_DIM, # keeping fixed 3840 since zero-masked
        hidden_dim=Config.FINAL_DIT_HIDDEN_DIM,
        depth=Config.FINAL_DIT_DEPTH,
        num_heads=Config.FINAL_DIT_HEADS,
        mlp_ratio=Config.FINAL_DIT_MLP_RATIO,
        dropout=Config.FINAL_DROPOUT
    ).to(device)
    
    # torch.compile for kernel fusion speedup (20-40%)
    if Config.COMPILE_MODEL:
        logger.info("Compiling model with torch.compile()...")
        model = torch.compile(model)
    
    # Log model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    criterion = nn.CrossEntropyLoss()
    
    # AdamW with weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr=Config.FINAL_LR, weight_decay=Config.WEIGHT_DECAY)
    
    # Cosine Annealing LR with warmup
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, 
                                 total_iters=Config.WARMUP_EPOCHS)
    cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=Config.FINAL_EPOCHS - Config.WARMUP_EPOCHS, 
                                                    T_mult=1, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], 
                              milestones=[Config.WARMUP_EPOCHS])
    
    # AMP GradScaler
    scaler = torch.amp.GradScaler('cuda', enabled=Config.USE_AMP)
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_dit_model.pth")
    patience_counter = 0
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    lr_history = []
    
    for epoch in range(Config.FINAL_EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
            
            # AMP autocast for mixed precision
            with torch.amp.autocast('cuda', enabled=Config.USE_AMP):
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
            
            # Scaled backward pass
            scaler.scale(loss).backward()
            
            # Gradient clipping for stability
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRADIENT_CLIP)
            
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * x_batch.size(0)
            preds = torch.argmax(logits, dim=1)
            correct_train += (preds == y_batch).sum().item()
            total_train += x_batch.size(0)
            
        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train
        
        # Step the LR scheduler
        scheduler.step()
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
        
        # Save best model (based on val loss)
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_val_acc = epoch_val_acc
            # Handle compiled model state dict
            model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
            torch.save(model_to_save.state_dict(), best_model_path)
            logger.info(f"--> Saved new best model with Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= Config.EARLY_STOP_PATIENCE:
            logger.info(f"Early stopping triggered at epoch {epoch+1} (no improvement for {Config.EARLY_STOP_PATIENCE} epochs)")
            break
            
    # Plot Training Curves (3 subplots: loss, accuracy, LR)
    logger.info("Saving training curves...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].plot(train_losses, label='Train Loss', linewidth=1.5)
    axes[0].plot(val_losses, label='Validation Loss', linewidth=1.5)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(train_accs, label='Train Accuracy', linewidth=1.5)
    axes[1].plot(val_accs, label='Validation Accuracy', linewidth=1.5)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(lr_history, label='Learning Rate', color='green', linewidth=1.5)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('LR')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.LOG_DIR, "training_curves.png"), dpi=200)
    plt.close()
    
    logger.info(f"Best Val Loss: {best_val_loss:.4f}, Best Val Acc: {best_val_acc:.4f}")
    logger.info("========== Pipeline Completed ==========")
    from evaluate import evaluate
    evaluate()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ablation', type=str, default=None,
                        choices=['dit_only', 'boa_only', 'woa_only'],
                        help='Run ablation variant instead of full pipeline')
    args = parser.parse_args()
    main(args.ablation)
