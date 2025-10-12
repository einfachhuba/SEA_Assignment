import numpy as np

def simple_hill_climbing(fitness_function: callable, bounds: tuple, step_size: float = 0.1,
                        max_iterations: int = 100, initial_point: float = None, 
                        minimize: bool = True, dimensions: int = 1) -> tuple:
    """
    Simple Hill Climbing: Moves to the first neighbor that improves the solution.
    
    Args:
        fitness_function: The objective function to optimize
        bounds: (lower_bound, upper_bound) tuple defining search space
        step_size: Size of the step to take when exploring neighbors
        max_iterations: Maximum number of iterations
        initial_point: Starting point (if None, random initialization)
        minimize: If True, minimize the function; if False, maximize it
        dimensions: 1 for 1D problems, 2 for 2D problems
    
    Returns:
        Tuple of (best_candidate, best_fitness, history)
    """
    lower_bound, upper_bound = bounds
    
    # Initialize starting point
    if initial_point is None:
        if dimensions == 1:
            current = np.random.uniform(lower_bound, upper_bound)
        else:
            current = np.array([
                np.random.uniform(lower_bound, upper_bound),
                np.random.uniform(lower_bound, upper_bound)
            ])
    else:
        current = np.array(initial_point) if dimensions == 2 else initial_point
    
    current_fitness = fitness_function(current)
    history = [(np.copy(current) if dimensions == 2 else current, current_fitness)]
    
    for _ in range(max_iterations):
        # Define neighbors based on dimensionality
        if dimensions == 1:
            neighbors = [
                current - step_size,
                current + step_size
            ]
        else:
            neighbors = [
                np.array([current[0] - step_size, current[1]]),  # Left
                np.array([current[0] + step_size, current[1]]),  # Right
                np.array([current[0], current[1] - step_size]),  # Down
                np.array([current[0], current[1] + step_size]),  # Up
            ]
        
        # Keep neighbors within bounds
        if dimensions == 1:
            neighbors = [n for n in neighbors if lower_bound <= n <= upper_bound]
        else:
            neighbors = [n for n in neighbors if 
                        lower_bound <= n[0] <= upper_bound and 
                        lower_bound <= n[1] <= upper_bound]
        
        # Find first improving neighbor
        improved = False
        for neighbor in neighbors:
            neighbor_fitness = fitness_function(neighbor)
            
            # Check if neighbor is better (minimization or maximization)
            is_better = neighbor_fitness < current_fitness if minimize else neighbor_fitness > current_fitness
            
            if is_better:
                current = neighbor if dimensions == 1 else np.copy(neighbor)
                current_fitness = neighbor_fitness
                history.append((np.copy(current) if dimensions == 2 else current, current_fitness))
                improved = True
                break
        
        if not improved:
            break
    
    return current, current_fitness, history


def adaptive_hill_climbing(fitness_function: callable, bounds: tuple, 
                          step_size: float = 0.1, max_iterations: int = 100, 
                          initial_point: float = None, 
                          acceleration: float = 1.2, deceleration: float = 0.5,
                          minimize: bool = True, dimensions: int = 1) -> tuple:
    """
    Adaptive Hill Climbing: Dynamically adjusts step size based on search progress.
    
    The algorithm increases step size after successful moves (to explore faster)
    and decreases it after failures (to refine the search locally).
    
    Args:
        fitness_function: The objective function to optimize
        bounds: (lower_bound, upper_bound) tuple defining search space
        step_size: Initial step size (will be adapted during search)
        max_iterations: Maximum number of iterations
        initial_point: Starting point (if None, random initialization)
        acceleration: Factor to increase step size after success (default: 1.2)
        deceleration: Factor to decrease step size after failure (default: 0.5)
        minimize: If True, minimize the function; if False, maximize it
        dimensions: 1 for 1D problems, 2 for 2D problems
    
    Returns:
        Tuple of (best_candidate, best_fitness, history)
    """
    lower_bound, upper_bound = bounds
    
    # Initialize starting point
    if initial_point is None:
        if dimensions == 1:
            current = np.random.uniform(lower_bound, upper_bound)
        else:
            current = np.array([
                np.random.uniform(lower_bound, upper_bound),
                np.random.uniform(lower_bound, upper_bound)
            ])
    else:
        current = np.array(initial_point) if dimensions == 2 else initial_point
    
    current_fitness = fitness_function(current)
    current_step = step_size
    history = [(np.copy(current) if dimensions == 2 else current, current_fitness)]
    
    for _ in range(max_iterations):
        # Define neighbors with current adaptive step size
        if dimensions == 1:
            neighbors = [
                current - current_step,
                current + current_step
            ]
        else:
            neighbors = [
                np.array([current[0] - current_step, current[1]]),  # Left
                np.array([current[0] + current_step, current[1]]),  # Right
                np.array([current[0], current[1] - current_step]),  # Down
                np.array([current[0], current[1] + current_step]),  # Up
            ]
        
        # Keep neighbors within bounds
        if dimensions == 1:
            neighbors = [n for n in neighbors if lower_bound <= n <= upper_bound]
        else:
            neighbors = [n for n in neighbors if 
                        lower_bound <= n[0] <= upper_bound and 
                        lower_bound <= n[1] <= upper_bound]
        
        # Find best neighbor
        best_neighbor = None
        best_neighbor_fitness = current_fitness
        
        for neighbor in neighbors:
            neighbor_fitness = fitness_function(neighbor)
            
            # Check if neighbor is better (minimization or maximization)
            is_better = neighbor_fitness < best_neighbor_fitness if minimize else neighbor_fitness > best_neighbor_fitness
            
            if is_better:
                best_neighbor = neighbor if dimensions == 1 else np.copy(neighbor)
                best_neighbor_fitness = neighbor_fitness
        
        # Adaptive step size adjustment based on success/failure
        if best_neighbor is not None:
            current = best_neighbor
            current_fitness = best_neighbor_fitness
            current_step = min(current_step * acceleration, (upper_bound - lower_bound) / 2)
            history.append((np.copy(current) if dimensions == 2 else current, current_fitness))
        else:
            current_step = current_step * deceleration
            
            # If step size becomes too small, stop (converged)
            if current_step < 1e-6:
                break
    
    return current, current_fitness, history