import os
import torch
import torch.nn as nn
import torchvision.models as models
import timm
import numpy as np
from tqdm import tqdm
from data.dataset import get_dataloaders
from config import Config

class FeatureExtractor(nn.Module):
    """
    Combined feature extractor using ResNet50, DenseNet121, and ViT-B/16.
    Extracted features are concatenated into a 3840-dim vector.
    """
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        
        # Load ResNet50, remove FC
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1]) # 2048-dim
        
        # Load DenseNet121, remove classifier
        densenet = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        self.densenet = densenet.features # 1024-dim, needs adaptive pooling
        self.densenet_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Load ViT-B/16 via timm
        # We set num_classes=0 to get pooled features (CLS token) => 768-dim
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
            
        self.to(self.device)
        self.eval()
        
    def forward(self, x):
        """
        Forward pass to extract and concatenate features.
        
        Args:
            x (Tensor): Input images tensor of shape (B, 3, 224, 224).
            
        Returns:
            Tensor: Concatenated features of shape (B, 3840).
        """
        with torch.no_grad():
            x = x.to(self.device)
            
            # ResNet: (B, 2048, 1, 1) -> (B, 2048)
            f_res = self.resnet(x)
            f_res = torch.flatten(f_res, 1)
            
            # DenseNet: (B, 1024, 7, 7) -> (B, 1024, 1, 1) -> (B, 1024)
            f_dense = self.densenet(x)
            f_dense = self.densenet_pool(f_dense)
            f_dense = torch.flatten(f_dense, 1)
            
            # ViT: (B, 768)
            f_vit = self.vit(x)
            
            # Concatenate features
            f_concat = torch.cat([f_res, f_dense, f_vit], dim=1) # (B, 3840)
            
        return f_concat

def extract_and_save_features(train_eval_dir: str, test_dir: str, cache_dir: str, batch_size: int, device: str):
    """
    Extracts features for train, val, and test splits and saves them as numpy files.
    Skips extraction if files already exist.
    """
    train_feat_path = os.path.join(cache_dir, "features_train.npy")
    train_lbl_path = os.path.join(cache_dir, "labels_train.npy")
    val_feat_path = os.path.join(cache_dir, "features_val.npy")
    val_lbl_path = os.path.join(cache_dir, "labels_val.npy")
    test_feat_path = os.path.join(cache_dir, "features_test.npy")
    test_lbl_path = os.path.join(cache_dir, "labels_test.npy")
    
    if all(os.path.exists(p) for p in [train_feat_path, val_feat_path, test_feat_path]):
        print(f"Features already cached in {cache_dir}. Skipping extraction.")
        return
        
    print("Initializing DataLoaders for feature extraction...")
    train_loader, val_loader, test_loader = get_dataloaders(train_eval_dir, test_dir, batch_size=batch_size)
    
    print("Initializing Feature Extractor...")
    extractor = FeatureExtractor(device=device)
    
    def process_split(loader, split_name):
        print(f"Extracting features for {split_name} split...")
        features_list = []
        labels_list = []
        
        for images, labels in tqdm(loader, desc=f"{split_name} Batches"):
            features = extractor(images)
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.numpy())
            
        features_arr = np.concatenate(features_list, axis=0)
        labels_arr = np.concatenate(labels_list, axis=0)
        
        feat_path = os.path.join(cache_dir, f"features_{split_name}.npy")
        lbl_path = os.path.join(cache_dir, f"labels_{split_name}.npy")
        np.save(feat_path, features_arr)
        np.save(lbl_path, labels_arr)
        print(f"Saved {split_name} features shape: {features_arr.shape}")
        
    process_split(train_loader, "train")
    process_split(val_loader, "val")
    process_split(test_loader, "test")
