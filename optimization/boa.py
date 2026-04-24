import numpy as np

def update_boa_population(population: np.ndarray, fitness_values: np.ndarray, global_best_pos: np.ndarray, 
                          c: float, a: float, p: float) -> np.ndarray:
    """
    Performs a single update iteration of the Butterfly Optimization Algorithm (BOA).
    Fully vectorized — no Python loops over population.
    
    Args:
        population (np.ndarray): Current population of shape (pop_size, dim). Values are continuous.
        fitness_values (np.ndarray): Fitness values for the population (pop_size,). Lower is better.
        global_best_pos (np.ndarray): Global best position found so far (dim,).
        c (float): Sensory modality.
        a (float): Power exponent.
        p (float): Switch probability.
        
    Returns:
        np.ndarray: Updated continuous population.
    """
    pop_size, dim = population.shape
    new_population = np.copy(population)
    
    # Calculate stimulus intensity I (vectorized)
    # Since fitness is an error rate (lower is better), we make intensity inversely proportional.
    intensity = 1.0 / (fitness_values + 1e-10)  # shape: (pop_size,)
    
    # Calculate fragrance f (vectorized)
    fragrance = c * np.power(intensity, a)  # shape: (pop_size,)
    
    # Random decision per individual: global vs local
    r1 = np.random.rand(pop_size)
    global_mask = r1 < p        # True = move towards global best
    local_mask = ~global_mask   # True = move randomly
    
    n_global = np.sum(global_mask)
    n_local = np.sum(local_mask)
    
    # --- Global move (towards global best) ---
    if n_global > 0:
        r2 = np.random.rand(n_global, 1)  # broadcast over dim
        frag_g = fragrance[global_mask].reshape(-1, 1)
        new_population[global_mask] = (
            population[global_mask] 
            + (r2 * r2 * global_best_pos[np.newaxis, :] - population[global_mask]) * frag_g
        )
    
    # --- Local move (random pair interaction) ---
    if n_local > 0:
        local_indices = np.where(local_mask)[0]
        # Random pairs j, k for each local individual
        jk = np.array([np.random.choice(pop_size, 2, replace=False) for _ in local_indices])
        j_idx = jk[:, 0]
        k_idx = jk[:, 1]
        r3 = np.random.rand(n_local, 1)
        frag_l = fragrance[local_mask].reshape(-1, 1)
        new_population[local_mask] = (
            population[local_mask]
            + (r3 * r3 * population[j_idx] - population[k_idx]) * frag_l
        )
            
    return new_population
