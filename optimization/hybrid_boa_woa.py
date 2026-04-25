import numpy as np
import logging
import torch
from optimization.fitness import evaluate_fitness
from optimization.boa import update_boa_population
from optimization.woa import update_woa_population
from config import Config

logger = logging.getLogger(__name__)

def sigmoid(x: np.ndarray) -> np.ndarray:
    # use np.clip to prevent overflow
    x_clipped = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x_clipped))

def get_binary_mask(population: np.ndarray) -> np.ndarray:
    return (sigmoid(population) >= 0.5).astype(np.float32)

def run_hybrid_boa_woa(train_features: np.ndarray, train_labels: np.ndarray, 
                       val_features: np.ndarray, val_labels: np.ndarray, 
                       device: str, ablation_mode: str = None) -> np.ndarray:
    """
    Executes the hybrid BOA-WOA algorithm to find the optimal feature selection mask.
    
    Optimized: Pre-converts features to GPU tensors once, strategic CUDA cache management.
    
    Args:
        train_features (np.ndarray): Cached training features.
        train_labels (np.ndarray): Cached training labels.
        val_features (np.ndarray): Cached validation features.
        val_labels (np.ndarray): Cached validation labels.
        device (str): Compute device.
        ablation_mode (str): Optional ablation mode.
        
    Returns:
        np.ndarray: The best binary mask found of shape (3840,).
    """
    pop_size = Config.POPULATION_SIZE
    dim = Config.FEATURE_DIM
    max_iter = Config.MAX_ITER
    c = Config.BOA_SENSORY_MODALITY
    a = Config.BOA_POWER_EXPONENT
    p_boa = Config.BOA_SWITCH_PROB
    
    # Pre-convert features to GPU tensors ONCE (avoids 7500+ numpy→tensor→GPU copies)
    logger.info("Pre-loading features to GPU for fitness evaluation...")
    train_features_gpu = torch.tensor(train_features, dtype=torch.float32, device=device)
    train_labels_gpu = torch.tensor(train_labels, dtype=torch.long, device=device)
    val_features_gpu = torch.tensor(val_features, dtype=torch.float32, device=device)
    val_labels_gpu = torch.tensor(val_labels, dtype=torch.long, device=device)
    if device == "cuda" and torch.cuda.is_available():
        logger.info(f"GPU memory allocated for features: {torch.cuda.memory_allocated(device) / 1e6:.1f} MB")
    
    # Initialize population randomly
    # Standard uniform distribution for initial continuous features
    population = np.random.uniform(low=-1.0, high=1.0, size=(pop_size, dim))
    fitness_values = np.zeros(pop_size)
    
    global_best_fitness = float('inf')
    global_best_pos = np.zeros(dim)
    
    logger.info(f"Evaluating initial population ({pop_size} agents, {dim} dimensions)...")
    for i in range(pop_size):
        mask = get_binary_mask(population[i])
        # If no features selected, heavily penalize
        if np.sum(mask) == 0:
            fitness_values[i] = 1.0
            continue
        try:
            fit = evaluate_fitness(mask, train_features_gpu, train_labels_gpu, 
                                   val_features_gpu, val_labels_gpu, device)
            fitness_values[i] = fit
        except Exception as e:
            logger.error(f"Error evaluating individual {i}: {e}")
            fitness_values[i] = 1.0
            
        if fitness_values[i] < global_best_fitness:
            global_best_fitness = fitness_values[i]
            global_best_pos = population[i].copy()
            
    logger.info(f"Initial Best Fitness: {global_best_fitness:.4f}")
            
    for iteration in range(max_iter):
        # --- BOA PHASE ---
        if ablation_mode != 'woa_only':
            population = update_boa_population(population, fitness_values, global_best_pos, c, a, p_boa)
            
            # Evaluate after BOA
            for i in range(pop_size):
                mask = get_binary_mask(population[i])
                if np.sum(mask) == 0:
                    fitness_values[i] = 1.0
                    continue
                fit = evaluate_fitness(mask, train_features_gpu, train_labels_gpu, 
                                       val_features_gpu, val_labels_gpu, device)
                fitness_values[i] = fit
                if fitness_values[i] < global_best_fitness:
                    global_best_fitness = fitness_values[i]
                    global_best_pos = population[i].copy()
                
        # --- WOA PHASE ---
        if ablation_mode != 'boa_only':
            population = update_woa_population(population, global_best_pos, iteration, max_iter)
            
            # Evaluate after WOA
            for i in range(pop_size):
                mask = get_binary_mask(population[i])
                if np.sum(mask) == 0:
                    fitness_values[i] = 1.0
                    continue
                fit = evaluate_fitness(mask, train_features_gpu, train_labels_gpu, 
                                       val_features_gpu, val_labels_gpu, device)
                fitness_values[i] = fit
                if fitness_values[i] < global_best_fitness:
                    global_best_fitness = fitness_values[i]
                    global_best_pos = population[i].copy()
                
        best_mask = get_binary_mask(global_best_pos)
        selected_features = int(np.sum(best_mask))
        logger.info(f"Iteration [{iteration+1}/{max_iter}] - Selected Features: {selected_features}/{dim} - Best Fitness: {global_best_fitness:.4f}")
        
        # Strategic CUDA cache cleanup every 10 iterations
        if device == "cuda" and torch.cuda.is_available() and (iteration + 1) % 10 == 0:
            torch.cuda.empty_cache()
    
    # Final cleanup — free GPU tensors
    del train_features_gpu, train_labels_gpu, val_features_gpu, val_labels_gpu
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return get_binary_mask(global_best_pos)
