import torch
import os

class Config:
    """
    Configuration hyperparameters and global settings for the ML pipeline.
    """
    # Dataset Parameters
    TRAIN_EVAL_DIR = os.getenv("LC25000_TRAIN_EVAL_DIR", "./data/LC25000/Train and Validation Set")
    TEST_DIR = os.getenv("LC25000_TEST_DIR", "./data/LC25000/Test Set")
    CACHE_DIR = "./feature_cache"
    BATCH_SIZE = 64
    NUM_WORKERS = 16
    
    # Feature Extraction
    FEATURE_DIM = 3840 # 2048 (ResNet50) + 1024 (DenseNet121) + 768 (ViT-B/16)
    
    # Optimization (BOA/WOA)
    POPULATION_SIZE = 30
    MAX_ITER = 50
    BOA_SENSORY_MODALITY = 0.01
    BOA_POWER_EXPONENT = 0.1
    BOA_SWITCH_PROB = 0.8
    
    # DiT Classifier (Fitness Evaluation Lightweight)
    FITNESS_DIT_HIDDEN_DIM = 128
    FITNESS_DIT_DEPTH = 2
    FITNESS_DIT_HEADS = 4
    FITNESS_DIT_EPOCHS = 10
    FITNESS_BATCH_SIZE = 512
    FITNESS_MLP_RATIO = 4.0  # Kept from original to prevent crash
    
    # DiT Classifier (Final Training)
    FINAL_DIT_HIDDEN_DIM = 512
    FINAL_DIT_DEPTH = 8
    FINAL_DIT_HEADS = 8
    FINAL_EPOCHS = 100
    FINAL_LR = 1e-4
    FINAL_BATCH_SIZE = 128
    FINAL_DIT_MLP_RATIO = 4.0  # Kept from original to prevent crash
    
    # Training
    USE_AMP = True
    DEVICE = "cuda"
    
    # Logs and Checkpoints
    CHECKPOINT_DIR = "./checkpoints"
    LOG_DIR = "./logs"

# Ensure directories exist
os.makedirs(Config.CACHE_DIR, exist_ok=True)
os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(Config.LOG_DIR, exist_ok=True)
