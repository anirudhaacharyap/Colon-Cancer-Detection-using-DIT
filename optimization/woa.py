import numpy as np

def update_woa_population(population: np.ndarray, global_best_pos: np.ndarray, 
                          current_iter: int, max_iter: int) -> np.ndarray:
    """
    Performs a single update iteration of the Whale Optimization Algorithm (WOA).
    Fully vectorized — no Python loops over population.
    
    Args:
        population (np.ndarray): Current population of shape (pop_size, dim). Values are continuous.
        global_best_pos (np.ndarray): Global best position found so far (dim,).
        current_iter (int): Current iteration number.
        max_iter (int): Maximum number of iterations.
        
    Returns:
        np.ndarray: Updated continuous population.
    """
    pop_size, dim = population.shape
    new_population = np.copy(population)
    
    # 'a' decreases linearly from 2 to 0
    a_coeff = 2.0 - current_iter * (2.0 / max_iter)
    
    # Generate all random numbers at once (vectorized)
    r1 = np.random.rand(pop_size)
    r2 = np.random.rand(pop_size)
    p_rand = np.random.rand(pop_size)
    l_rand = np.random.rand(pop_size) * 2.0 - 1.0  # [-1, 1]
    
    A = 2.0 * a_coeff * r1 - a_coeff  # shape: (pop_size,)
    C = 2.0 * r2                       # shape: (pop_size,)
    b = 1.0  # constant for logarithmic spiral
    
    # Classify individuals into 3 groups
    spiral_mask = p_rand >= 0.5                          # spiral update
    encircle_mask = (p_rand < 0.5) & (np.abs(A) < 1)    # encircling prey
    search_mask = (p_rand < 0.5) & (np.abs(A) >= 1)     # search (random)
    
    # --- Spiral update ---
    n_spiral = np.sum(spiral_mask)
    if n_spiral > 0:
        distance_to_best = np.abs(global_best_pos[np.newaxis, :] - population[spiral_mask])
        l_s = l_rand[spiral_mask].reshape(-1, 1)
        new_population[spiral_mask] = (
            distance_to_best * np.exp(b * l_s) * np.cos(2.0 * np.pi * l_s) 
            + global_best_pos[np.newaxis, :]
        )
    
    # --- Encircling prey ---
    n_encircle = np.sum(encircle_mask)
    if n_encircle > 0:
        C_e = C[encircle_mask].reshape(-1, 1)
        A_e = A[encircle_mask].reshape(-1, 1)
        D = np.abs(C_e * global_best_pos[np.newaxis, :] - population[encircle_mask])
        new_population[encircle_mask] = global_best_pos[np.newaxis, :] - A_e * D
    
    # --- Search for prey (random) ---
    n_search = np.sum(search_mask)
    if n_search > 0:
        rand_indices = np.random.randint(0, pop_size, size=n_search)
        rand_pos = population[rand_indices]
        C_s = C[search_mask].reshape(-1, 1)
        A_s = A[search_mask].reshape(-1, 1)
        D_rand = np.abs(C_s * rand_pos - population[search_mask])
        new_population[search_mask] = rand_pos - A_s * D_rand
            
    return new_population
