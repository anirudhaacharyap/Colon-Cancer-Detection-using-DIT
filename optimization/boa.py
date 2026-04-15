import numpy as np

def update_boa_population(population: np.ndarray, fitness_values: np.ndarray, global_best_pos: np.ndarray, 
                          c: float, a: float, p: float) -> np.ndarray:
    """
    Performs a single update iteration of the Butterfly Optimization Algorithm (BOA).
    
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
    
    # Calculate stimulus intensity I. 
    # Since fitness is an error rate (lower is better), we make intensity inversely proportional.
    # To avoid division by zero or negative values:
    intensity = 1.0 / (fitness_values + 1e-10)
    
    # Calculate fragrance f
    fragrance = c * np.power(intensity, a)
    
    for i in range(pop_size):
        r1 = np.random.rand()
        
        if r1 < p:
            # Move towards global best
            r2 = np.random.rand()
            new_population[i] = population[i] + (r2 * r2 * global_best_pos - population[i]) * fragrance[i]
        else:
            # Move randomly
            j, k = np.random.choice(pop_size, 2, replace=False)
            r3 = np.random.rand()
            new_population[i] = population[i] + (r3 * r3 * population[j] - population[k]) * fragrance[i]
            
    return new_population
