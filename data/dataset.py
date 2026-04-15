import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import numpy as np

def get_dataloaders(train_eval_dir: str, test_dir: str, batch_size: int = 32):
    """
    Creates and returns DataLoaders for train, validation, and test splits.
    Splits the train_eval_dir into Train and Validation (approx 82/18 to mimic 70/15 of total).
    Uses test_dir directly for the Test split.
    
    Args:
        train_eval_dir (str): Path to the directory containing training and eval class folders.
        test_dir (str): Path to the directory containing test class folders.
        batch_size (int): Number of inputs per batch.
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Define augmentations for training
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Define transforms for validation and testing (no augmentation)
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load Train/Eval data
    full_dataset_train = datasets.ImageFolder(root=train_eval_dir, transform=train_transforms)
    full_dataset_eval = datasets.ImageFolder(root=train_eval_dir, transform=test_transforms)
    
    # Load Test data
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transforms)
    
    # Extract labels for stratified split of train_eval
    targets = full_dataset_train.targets

    # Stratified split: split train_eval into train (82.35%) and val (17.65%)
    # This ratio is 15/(70+15) = 15/85 ~ 0.1765.
    indices = np.arange(len(targets))
    train_idx, val_idx, _, _ = train_test_split(
        indices, targets, test_size=0.1765, stratify=targets, random_state=42
    )

    # Create Subsets
    train_dataset = Subset(full_dataset_train, train_idx)
    val_dataset = Subset(full_dataset_eval, val_idx)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True)

    return train_loader, val_loader, test_loader
