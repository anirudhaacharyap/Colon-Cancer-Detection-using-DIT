import torch
import os

class Config:
    """
    Configuration hyperparameters and global settings for the ML pipeline.
    Optimized for: Threadripper PRO 5995WX (64-core), 256 GB RAM, RTX 3090 (24 GB).
    Target: > 99.87% accuracy.
    """
    # Dataset Parameters
    TRAIN_EVAL_DIR = os.getenv("LC25000_TRAIN_EVAL_DIR", "./data/LC25000/Train and Validation Set")
    TEST_DIR = os.getenv("LC25000_TEST_DIR", "./data/LC25000/Test Set")
    CACHE_DIR = "./feature_cache"
    BATCH_SIZE = 128
    NUM_WORKERS = 24
    PREFETCH_FACTOR = 4
    PIN_MEMORY = True
    PERSISTENT_WORKERS = True
    SEED = 42

    # Feature Extraction
    FEATURE_DIM = 3840

    # Optimization (BOA/WOA)
    POPULATION_SIZE = 50
    MAX_ITER = 75
    BOA_SENSORY_MODALITY = 0.01
    BOA_POWER_EXPONENT = 0.1
    BOA_SWITCH_PROB = 0.8

    # DiT Classifier (Fitness Evaluation Lightweight)
    FITNESS_DIT_HIDDEN_DIM = 128
    FITNESS_DIT_DEPTH = 2
    FITNESS_DIT_HEADS = 4
    FITNESS_DIT_EPOCHS = 15
    FITNESS_BATCH_SIZE = 4096
    FITNESS_MLP_RATIO = 4.0
    FITNESS_LR = 5e-4
    FITNESS_USE_SUBSET = False
    FITNESS_SUBSET_FRACTION = 0.6

    # Step 2 (BOA-WOA) cache behavior
    SKIP_STEP2_IF_MASK_EXISTS = True
    FORCE_STEP2_RECOMPUTE = False

    # DiT Classifier (Final Training)
    FINAL_DIT_HIDDEN_DIM = 768
    FINAL_DIT_DEPTH = 12
    FINAL_DIT_HEADS = 12
    FINAL_EPOCHS = 200             # was 150 — cosine schedule needs more room at low LR
    FINAL_LR = 1e-4                # was 3e-4 — lower LR pairs better with depth=12
    FINAL_BATCH_SIZE = 256         # throughput-oriented default for RTX 3090
    FINAL_DIT_MLP_RATIO = 4.0
    FINAL_DROPOUT = 0.15           # was 0.1 — slightly more regularization for depth=12
    FINAL_NUM_WORKERS = 24
    FINAL_PREFETCH_FACTOR = 4

    # Label Smoothing
    LABEL_SMOOTHING = 0.1          # was missing — prevents overconfidence on 2 hard cases

    # LR Scheduler
    LR_SCHEDULER = "cosine"
    LR_ETA_MIN = 1e-7              # was missing — let LR decay lower than 1e-6
    WARMUP_EPOCHS = 10             # Cosine warmup epochs

    # TTA — Test Time Augmentation
    TTA_ENABLED = True             # was missing — free accuracy at eval, no retraining
    TTA_N_AUG = 7                  # was missing — 7 passes, odd number avoids ties
    TTA_NOISE_STD = 0.008          # was missing — slightly less noise than 0.01 for stability

    # Training Optimization
    USE_AMP = True
    DEVICE = "cuda"
    COMPILE_MODEL = True
    ALLOW_TF32 = True
    MATMUL_PRECISION = "high"
    WEIGHT_DECAY = 0.05
    ENABLE_EARLY_STOPPING = False
    EARLY_STOP_PATIENCE = 25       # was 20 — give more room with 200 epochs
    GRADIENT_CLIP = 1.0
    MIXUP_ALPHA = 0.2              # Mixup augmentation alpha (Beta distribution)
    CHECKPOINT_MIN_DELTA = 1e-4
    SAVE_BEST_BY_ACC = True
    USE_EMA = True
    EMA_DECAY = 0.999
    USE_SWA = True
    SWA_START_EPOCH = 140
    SWA_LR = 5e-5
    TUNE_THRESHOLD_ON_VAL = True
    THRESHOLD_OBJECTIVE = "f1"
    DEFAULT_DECISION_THRESHOLD = 0.5

    # Logs and Checkpoints
    CHECKPOINT_DIR = "./checkpoints"
    LOG_DIR = "./logs"

# Ensure directories exist
os.makedirs(Config.CACHE_DIR, exist_ok=True)
os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(Config.LOG_DIR, exist_ok=True)