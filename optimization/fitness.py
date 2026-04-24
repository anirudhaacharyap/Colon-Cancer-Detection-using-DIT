import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
import numpy as np

from models.dit_classifier import DiTClassifier
from config import Config

def evaluate_fitness(mask: np.ndarray, train_features_gpu: torch.Tensor, train_labels_gpu: torch.Tensor,
                     val_features_gpu: torch.Tensor, val_labels_gpu: torch.Tensor, device: str) -> float:
    """
    Evaluates the fitness of a binary mask by training a lightweight DiT model
    on the masked features and returning the validation error.
    
    Optimized: Accepts pre-allocated GPU tensors, uses AMP, avoids redundant copies.
    
    Args:
        mask (np.ndarray): Binary mask of shape (3840,).
        train_features_gpu (torch.Tensor): Training features already on GPU.
        train_labels_gpu (torch.Tensor): Training labels already on GPU.
        val_features_gpu (torch.Tensor): Validation features already on GPU.
        val_labels_gpu (torch.Tensor): Validation labels already on GPU.
        device (str): Compute device.
        
    Returns:
        float: Fitness score (1 - validation accuracy). Lower is better.
    """
    # Convert mask to GPU tensor once
    mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device)
    
    # Apply zero-masking on GPU (avoids numpy→tensor conversion per call)
    masked_train = train_features_gpu * mask_tensor.unsqueeze(0)
    masked_val = val_features_gpu * mask_tensor.unsqueeze(0)
    
    # Create DataLoaders from GPU tensors (move to CPU for DataLoader compatibility)
    train_dataset = TensorDataset(masked_train.cpu(), train_labels_gpu.cpu())
    val_dataset = TensorDataset(masked_val.cpu(), val_labels_gpu.cpu())
    
    train_loader = DataLoader(train_dataset, batch_size=Config.FITNESS_BATCH_SIZE, shuffle=True,
                              pin_memory=True, num_workers=0)  # num_workers=0 since data is small
    val_loader = DataLoader(val_dataset, batch_size=Config.FITNESS_BATCH_SIZE, shuffle=False,
                            pin_memory=True, num_workers=0)
    
    # Instantiate lightweight DiT model
    model = DiTClassifier(
        feature_dim=Config.FEATURE_DIM,
        hidden_dim=Config.FITNESS_DIT_HIDDEN_DIM,
        depth=Config.FITNESS_DIT_DEPTH,
        num_heads=Config.FITNESS_DIT_HEADS,
        mlp_ratio=Config.FITNESS_MLP_RATIO,
        dropout=0.0  # No dropout for lightweight fitness model
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.FITNESS_LR)
    
    # AMP scaler for mixed precision training
    scaler = torch.amp.GradScaler('cuda', enabled=Config.USE_AMP)
    
    # Train for FITNESS_DIT_EPOCHS with AMP
    for epoch in range(Config.FITNESS_DIT_EPOCHS):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)  # Slightly faster than zero_grad()
            
            with torch.amp.autocast('cuda', enabled=Config.USE_AMP):
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
    # Evaluate on validation set with AMP
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=Config.USE_AMP):
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device, non_blocking=True)
            logits = model(x_batch)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y_batch.numpy())
    
    # Clean up model to free VRAM
    del model, optimizer, scaler
            
    val_accuracy = accuracy_score(all_targets, all_preds)
    fitness_score = 1.0 - val_accuracy
    
    return fitness_score
