import numpy as np

def update_woa_population(population: np.ndarray, global_best_pos: np.ndarray, 
                          current_iter: int, max_iter: int) -> np.ndarray:
    """
    Performs a single update iteration of the Whale Optimization Algorithm (WOA).
    
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
    a = 2.0 - current_iter * (2.0 / max_iter)
    
    for i in range(pop_size):
        r1 = np.random.rand()
        r2 = np.random.rand()
        
        A = 2.0 * a * r1 - a
        C = 2.0 * r2
        
        b = 1.0  # constant for defining the shape of the logarithmic spiral
        l = (np.random.rand() * 2.0 - 1.0) # [-1, 1]
        
        p = np.random.rand()
        
        if p < 0.5:
            if abs(A) < 1:
                # Encircling prey
                D = abs(C * global_best_pos - population[i])
                new_population[i] = global_best_pos - A * D
            else:
                # Search for prey (random)
                rand_idx = np.random.randint(0, pop_size)
                rand_pos = population[rand_idx]
                D_rand = abs(C * rand_pos - population[i])
                new_population[i] = rand_pos - A * D_rand
        else:
            # Bubble-net attacking method (spiral update)
            distance_to_best = abs(global_best_pos - population[i])
            new_population[i] = distance_to_best * np.exp(b * l) * np.cos(2.0 * np.pi * l) + global_best_pos
            
    return new_population
