import numpy as np

def random_search(fitness_function: callable, bounds: tuple, iterations: bool = True, 
    max_iterations: int = 100) -> tuple[float, float, list[tuple[float, float]]]:
    lower_bound, upper_bound = bounds
    best_candidate = None
    best_fitness = -np.inf
    candidate_history = []

    if iterations:
        for _ in range(max_iterations):
            candidate = np.random.uniform(lower_bound, upper_bound)
            candidate_fitness = fitness_function(candidate)
            candidate_history.append((candidate, candidate_fitness))

            if candidate_fitness > best_fitness:
                best_candidate, best_fitness = candidate, candidate_fitness

    return best_candidate, best_fitness, candidate_history