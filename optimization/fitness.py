import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
import numpy as np

from models.dit_classifier import DiTClassifier
from config import Config

def evaluate_fitness(mask: np.ndarray, train_features: np.ndarray, train_labels: np.ndarray,
                     val_features: np.ndarray, val_labels: np.ndarray, device: str) -> float:
    """
    Evaluates the fitness of a binary mask by training a lightweight DiT model
    on the masked features and returning the validation error.
    
    Args:
        mask (np.ndarray): Binary mask of shape (3840,).
        train_features (np.ndarray): Training features.
        train_labels (np.ndarray): Training labels.
        val_features (np.ndarray): Validation features.
        val_labels (np.ndarray): Validation labels.
        device (str): Compute device.
        
    Returns:
        float: Fitness score (1 - validation accuracy). Lower is better.
    """
    # Apply zero-masking
    masked_train_features = train_features * mask
    masked_val_features = val_features * mask
    
    # Create DataLoaders
    train_dataset = TensorDataset(torch.tensor(masked_train_features, dtype=torch.float32), 
                                  torch.tensor(train_labels, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(masked_val_features, dtype=torch.float32), 
                                torch.tensor(val_labels, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=Config.FITNESS_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.FITNESS_BATCH_SIZE, shuffle=False)
    
    # Instantiate lightweight DiT model
    model = DiTClassifier(
        feature_dim=Config.FEATURE_DIM,
        hidden_dim=Config.FITNESS_DIT_HIDDEN_DIM,
        depth=Config.FITNESS_DIT_DEPTH,
        num_heads=Config.FITNESS_DIT_HEADS,
        mlp_ratio=Config.FITNESS_MLP_RATIO
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.FINAL_LR)
    
    # Train for FITNESS_DIT_EPOCHS
    for epoch in range(Config.FITNESS_DIT_EPOCHS):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
    # Evaluate on validation set
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y_batch.numpy())
            
    val_accuracy = accuracy_score(all_targets, all_preds)
    fitness_score = 1.0 - val_accuracy
    
    return fitness_score
