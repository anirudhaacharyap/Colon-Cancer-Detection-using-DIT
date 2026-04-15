import os
import logging
import numpy as np
import torch
torch.set_num_threads(32)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
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
    
    train_dataset = TensorDataset(torch.tensor(masked_train, dtype=torch.float32), 
                                  torch.tensor(labels_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(masked_val, dtype=torch.float32), 
                                torch.tensor(labels_val, dtype=torch.long))
                                
    train_loader = DataLoader(train_dataset, batch_size=Config.FINAL_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.FINAL_BATCH_SIZE, shuffle=False)
    
    model = DiTClassifier(
        feature_dim=Config.FEATURE_DIM, # keeping fixed 3840 since zero-masked
        hidden_dim=Config.FINAL_DIT_HIDDEN_DIM,
        depth=Config.FINAL_DIT_DEPTH,
        num_heads=Config.FINAL_DIT_HEADS,
        mlp_ratio=Config.FINAL_DIT_MLP_RATIO
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.FINAL_LR)
    
    best_val_loss = float('inf')
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_dit_model.pth")
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(Config.FINAL_EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * x_batch.size(0)
            preds = torch.argmax(logits, dim=1)
            correct_train += (preds == y_batch).sum().item()
            total_train += x_batch.size(0)
            
        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train
        
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
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
        
        logger.info(f"Epoch [{epoch+1}/{Config.FINAL_EPOCHS}] - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
        
        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"--> Saved new best model with Val Loss: {best_val_loss:.4f}")
            
    # Plot Training Curves
    logger.info("Saving training curves...")
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.LOG_DIR, "training_curves.png"))
    plt.close()
    
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
