import torch
import os

class Config:
    """
    Configuration hyperparameters and global settings for the ML pipeline.
    Optimized for: Threadripper PRO 5995WX (64-core), 256 GB RAM, RTX 3090 (24 GB).
    """
    # Dataset Parameters
    TRAIN_EVAL_DIR = os.getenv("LC25000_TRAIN_EVAL_DIR", "./data/LC25000/Train and Validation Set")
    TEST_DIR = os.getenv("LC25000_TEST_DIR", "./data/LC25000/Test Set")
    CACHE_DIR = "./feature_cache"
    BATCH_SIZE = 128               # Feature extraction batch — 3 backbones fit at 128 on 24 GB
    NUM_WORKERS = 24               # 64-core CPU — 24 workers leaves headroom
    PREFETCH_FACTOR = 4            # Aggressive prefetch with 256 GB RAM
    
    # Feature Extraction
    FEATURE_DIM = 3840 # 2048 (ResNet50) + 1024 (DenseNet121) + 768 (ViT-B/16)
    
    # Optimization (BOA/WOA)
    POPULATION_SIZE = 50           # Larger population for better exploration
    MAX_ITER = 75                  # More iterations for convergence
    BOA_SENSORY_MODALITY = 0.01
    BOA_POWER_EXPONENT = 0.1
    BOA_SWITCH_PROB = 0.8
    
    # DiT Classifier (Fitness Evaluation Lightweight)
    FITNESS_DIT_HIDDEN_DIM = 128
    FITNESS_DIT_DEPTH = 2
    FITNESS_DIT_HEADS = 4
    FITNESS_DIT_EPOCHS = 15        # More epochs for stable fitness signal
    FITNESS_BATCH_SIZE = 1024      # RTX 3090 handles this for lightweight model
    FITNESS_MLP_RATIO = 4.0        # Kept from original to prevent crash
    FITNESS_LR = 5e-4              # Dedicated LR for fitness evaluation
    
    # DiT Classifier (Final Training)
    FINAL_DIT_HIDDEN_DIM = 768     # Bigger model — 24 GB VRAM supports this
    FINAL_DIT_DEPTH = 12           # Deeper transformer for better feature interaction
    FINAL_DIT_HEADS = 12           # 768/12 = 64 dim per head (optimal)
    FINAL_EPOCHS = 150             # More epochs with cosine schedule + early stopping
    FINAL_LR = 3e-4                # Higher LR paired with cosine warmup + larger batch
    FINAL_BATCH_SIZE = 256         # Doubles throughput, well within VRAM budget
    FINAL_DIT_MLP_RATIO = 4.0     # Kept from original to prevent crash
    FINAL_DROPOUT = 0.1            # Regularization dropout for deeper model
    
    # Training Optimization
    USE_AMP = True                 # Mixed precision (FP16) for ~2× speedup
    DEVICE = "cuda"
    COMPILE_MODEL = True           # torch.compile() for kernel fusion (20-40% speedup)
    WARMUP_EPOCHS = 10             # Cosine warmup epochs
    WEIGHT_DECAY = 0.05            # AdamW weight decay for regularization
    EARLY_STOP_PATIENCE = 20       # Stop if no val improvement for 20 epochs
    GRADIENT_CLIP = 1.0            # Gradient clipping for stability
    
    # Logs and Checkpoints
    CHECKPOINT_DIR = "./checkpoints"
    LOG_DIR = "./logs"

# Ensure directories exist
os.makedirs(Config.CACHE_DIR, exist_ok=True)
os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(Config.LOG_DIR, exist_ok=True)
