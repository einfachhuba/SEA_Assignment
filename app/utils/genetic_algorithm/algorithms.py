import numpy as np
from typing import Callable, Tuple, List
import random


def initialize_population(population_size: int, bounds: List[Tuple[float, float]]) -> np.ndarray:
    """
    Initialize a random population within the given bounds.
    
    Args:
        population_size: Number of individuals in the population
        bounds: List of (min, max) tuples for each dimension
        
    Returns:
        np.ndarray: Population matrix of shape (population_size, n_dimensions)
    """
    n_dimensions = len(bounds)
    population = np.zeros((population_size, n_dimensions))
    
    for i, (min_val, max_val) in enumerate(bounds):
        population[:, i] = np.random.uniform(min_val, max_val, population_size)
    
    return population


def random_selection(population: np.ndarray, fitness_scores: np.ndarray) -> np.ndarray:
    """
    Select two parents randomly from the population.
    
    Args:
        population: Population matrix
        fitness_scores: Fitness values for each individual (not used but kept for consistency)
        
    Returns:
        np.ndarray: Two randomly selected parents
    """
    parent_indices = np.random.choice(len(population), 2, replace=False)
    return population[parent_indices].copy()


def single_point_crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform single-point crossover between two parents.
    
    Args:
        parent1: First parent chromosome
        parent2: Second parent chromosome
        
    Returns:
        Tuple of two offspring chromosomes
    """
    n_genes = len(parent1)
    
    # Choose crossover point (not at the ends)
    if n_genes <= 1:
        return parent1.copy(), parent2.copy()
    
    crossover_point = np.random.randint(1, n_genes)
    
    # Create offspring
    offspring1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
    offspring2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
    
    return offspring1, offspring2


def bit_flip_mutation(individual: np.ndarray, bounds: List[Tuple[float, float]], 
                     mutation_rate: float = 0.1) -> np.ndarray:
    """
    Apply bit-flip style mutation by randomly replacing genes with new random values.
    
    Args:
        individual: Individual to mutate
        bounds: List of (min, max) tuples for each dimension
        mutation_rate: Probability of mutating each gene
        
    Returns:
        np.ndarray: Mutated individual
    """
    mutated = individual.copy()
    
    for i, (min_val, max_val) in enumerate(bounds):
        if np.random.random() < mutation_rate:
            mutated[i] = np.random.uniform(min_val, max_val)
    
    return mutated


def genetic_algorithm(
    fitness_function: Callable[[np.ndarray], float],
    bounds: List[Tuple[float, float]],
    population_size: int = 50,
    max_generations: int = 100,
    mutation_rate: float = 0.1,
    elitism: bool = True,
    elite_size: int = 2,
    seed: int = None
) -> Tuple[np.ndarray, float, List[float], List[np.ndarray], List[float]]:
    """
    Main genetic algorithm implementation using assignment-required methods.
    
    Args:
        fitness_function: Function to evaluate individual fitness
        bounds: List of (min, max) tuples for each dimension
        population_size: Number of individuals in population
        max_generations: Maximum number of generations
        mutation_rate: Probability of mutation per gene
        mutation_strength: Not used for bit-flip mutation (kept for consistency)
        elitism: Whether to preserve best individuals
        elite_size: Number of elite individuals to preserve
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (best_individual, best_fitness, fitness_history, 
                 population_history, diversity_history)
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    # Initialize population
    population = initialize_population(population_size, bounds)
    
    # Track statistics
    fitness_history = []
    population_history = []
    diversity_history = []
    best_fitness = -np.inf
    best_individual = None
    
    # Choose functions
    crossover_func = single_point_crossover  # Only single-point crossover per assignment requirements
    mutation_func = bit_flip_mutation  # Only bit-flip mutation per assignment requirements
    selection_func = random_selection  # Only random selection is used per assignment requirements
    
    for generation in range(max_generations):
        # Evaluate fitness
        fitness_scores = np.array([fitness_function(ind) for ind in population])
        
        # Track best individual
        gen_best_idx = np.argmax(fitness_scores)
        gen_best_fitness = fitness_scores[gen_best_idx]
        
        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_individual = population[gen_best_idx].copy()
        
        # Track statistics
        fitness_history.append(gen_best_fitness)
        population_history.append(population.copy())
        
        # Calculate diversity (average pairwise distance)
        diversity = calculate_diversity(population)
        diversity_history.append(diversity)
        
        # Create new population
        new_population = []
        
        # Elitism: keep best individuals
        if elitism and elite_size > 0:
            elite_indices = np.argsort(fitness_scores)[-elite_size:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
        
        # Generate offspring to fill remaining slots
        while len(new_population) < population_size:
            # Select parents
            parents = selection_func(population, fitness_scores)
            
            # Crossover
            offspring1, offspring2 = crossover_func(parents[0], parents[1])
            
            # Mutation
            offspring1 = mutation_func(offspring1, bounds, mutation_rate)
            offspring2 = mutation_func(offspring2, bounds, mutation_rate)
            
            new_population.extend([offspring1, offspring2])
        
        # Ensure exact population size
        population = np.array(new_population[:population_size])
    
    return best_individual, best_fitness, fitness_history, population_history, diversity_history


def calculate_diversity(population: np.ndarray) -> float:
    """
    Calculate population diversity as average pairwise Euclidean distance.
    
    Args:
        population: Population matrix
        
    Returns:
        float: Average pairwise distance
    """
    n_individuals = len(population)
    if n_individuals < 2:
        return 0.0
    
    total_distance = 0.0
    count = 0
    
    for i in range(n_individuals):
        for j in range(i + 1, n_individuals):
            distance = np.linalg.norm(population[i] - population[j])
            total_distance += distance
            count += 1
    
    return total_distance / count if count > 0 else 0.0


def run_coffee_genetic_algorithm(
    population_size: int = 50,
    max_generations: int = 100,
    mutation_rate: float = 0.1,
    seed: int = None
) -> dict:
    """
    Convenience function to run GA on the coffee brewing problem using assignment-required methods.
    
    Args:
        population_size: Number of individuals in population
        max_generations: Maximum number of generations
        mutation_rate: Probability of mutation per gene
        mutation_strength: Not used for bit-flip mutation (kept for consistency)
        seed: Random seed for reproducibility
        
    Returns:
        dict: Results dictionary with all relevant information
    """
    from .functions import fitness_from_chromosome
    from .config import COFFEE_BOUNDS
    
    best_individual, best_fitness, fitness_history, population_history, diversity_history = genetic_algorithm(
        fitness_function=fitness_from_chromosome,
        bounds=COFFEE_BOUNDS,
        population_size=population_size,
        max_generations=max_generations,
        mutation_rate=mutation_rate,
        
        seed=seed
    )
    
    return {
        'best_individual': best_individual,
        'best_fitness': best_fitness,
        'fitness_history': fitness_history,
        'population_history': population_history,
        'diversity_history': diversity_history,
        'best_params': {
            'roast': best_individual[0],
            'blend': best_individual[1], 
            'grind': best_individual[2],
            'brew_time': best_individual[3]
        }
    }
