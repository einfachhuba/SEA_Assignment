"""
Genetic Algorithm implementation for Image Reconstruction with various strategy variations.
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Callable
import time
from skimage.metrics import structural_similarity as ssim


class ImageReconstructionGA:
    """
    Genetic Algorithm for image reconstruction with configurable strategies.
    """
    
    def __init__(
        self,
        target_image: np.ndarray,
        population_size: int = 100,
        max_generations: int = 500,
        mutation_rate: float = 0.01,
        crossover_rate: float = 0.8,
        tournament_size: int = 3,
        elite_size: int = 5,
        selection_strategy: str = 'tournament',
        crossover_strategy: str = 'uniform',
        mutation_strategy: str = 'gaussian',
        survivor_strategy: str = 'elitist',
        initialization_strategy: str = 'random',
        fitness_threshold: float = 0.98,
        max_stagnation: int = 50,
        seed: int = None
    ):
        """
        Initialize the Genetic Algorithm for image reconstruction.
        
        Args:
            target_image: The target image to reconstruct
            population_size: Number of individuals in population
            max_generations: Maximum number of generations
            mutation_rate: Probability of mutation per gene
            crossover_rate: Probability of crossover
            tournament_size: Size of tournament for tournament selection
            elite_size: Number of elite individuals to preserve
            selection_strategy: Parent selection strategy
            crossover_strategy: Crossover operator strategy
            mutation_strategy: Mutation operator strategy
            survivor_strategy: Survivor selection strategy
            initialization_strategy: Population initialization strategy
            fitness_threshold: Fitness threshold for early termination
            max_stagnation: Maximum generations without improvement
            seed: Random seed for reproducibility
        """
        self.target_image = target_image.flatten()
        self.image_shape = target_image.shape
        self.dimensions = len(self.target_image)
        
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elite_size = elite_size
        
        self.selection_strategy = selection_strategy
        self.crossover_strategy = crossover_strategy
        self.mutation_strategy = mutation_strategy
        self.survivor_strategy = survivor_strategy
        self.initialization_strategy = initialization_strategy
        
        self.fitness_threshold = fitness_threshold
        self.max_stagnation = max_stagnation
        
        if seed is not None:
            np.random.seed(seed)
        
        # Statistics tracking
        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'worst_fitness': [],
            'diversity': [],
            'mse': [],
            'psnr': [],
            'ssim': [],
            'fitness_evaluations': []
        }
        
        self.fitness_eval_count = 0
        self.best_individual = None
        self.best_fitness = -np.inf
    
    def initialize_population(self) -> np.ndarray:
        """
        Initialize the population using the selected strategy.
        
        Returns:
            np.ndarray: Initial population
        """
        population = np.zeros((self.population_size, self.dimensions))
        
        if self.initialization_strategy == 'random':
            population = np.random.uniform(0, 255, (self.population_size, self.dimensions))
            
        elif self.initialization_strategy == 'gaussian_noise':
            mean = np.mean(self.target_image)
            std = np.std(self.target_image)
            population = np.random.normal(mean, std, (self.population_size, self.dimensions))
            population = np.clip(population, 0, 255)
            
        elif self.initialization_strategy == 'oversampling_selection':
            # Create larger population and select best individuals
            oversample_factor = 3  # Create 3x more individuals
            large_population_size = self.population_size * oversample_factor
            
            # Generate random population
            large_population = np.random.uniform(0, 255, (large_population_size, self.dimensions))
            
            # Evaluate all individuals
            fitness_values = np.array([self.calculate_fitness(ind) for ind in large_population])
            
            # Select top population_size individuals
            best_indices = np.argsort(fitness_values)[-self.population_size:]
            population = large_population[best_indices]
            
        elif self.initialization_strategy == 'local_optimization':
            # Initialize randomly then apply local optimization to each individual
            population = np.random.uniform(0, 255, (self.population_size, self.dimensions))
            
            # Apply hill climbing to each individual
            for i in range(self.population_size):
                population[i] = self._local_hill_climbing(population[i], max_iterations=20)
                
        elif self.initialization_strategy == 'edge_based':
            # Initialize based on edge information
            target_2d = self.target_image.reshape(self.image_shape)
            # Simple edge detection (gradient magnitude)
            grad_y = np.gradient(target_2d, axis=0)
            grad_x = np.gradient(target_2d, axis=1)
            edges = np.sqrt(grad_x**2 + grad_y**2)
            edges_flat = edges.flatten()
            
            for i in range(self.population_size):
                # Start with mean value
                individual = np.full(self.dimensions, np.mean(self.target_image))
                # Add edge information with noise
                individual += edges_flat * np.random.uniform(0.5, 1.5, self.dimensions)
                # Add random noise
                individual += np.random.normal(0, 20, self.dimensions)
                population[i] = np.clip(individual, 0, 255)
        
        return population
    
    def _local_hill_climbing(self, individual: np.ndarray, max_iterations: int = 20) -> np.ndarray:
        """
        Apply local hill climbing optimization to an individual.
        
        Args:
            individual: Individual to optimize
            max_iterations: Maximum number of iterations
            
        Returns:
            np.ndarray: Optimized individual
        """
        current = individual.copy()
        current_fitness = self.calculate_fitness(current)
        
        for _ in range(max_iterations):
            # Generate neighbor by adding small random noise
            neighbor = current + np.random.normal(0, 10, self.dimensions)
            neighbor = np.clip(neighbor, 0, 255)
            
            # Evaluate neighbor
            neighbor_fitness = self.calculate_fitness(neighbor)
            
            # Accept if better
            if neighbor_fitness > current_fitness:
                current = neighbor
                current_fitness = neighbor_fitness
            else:
                # Small chance to accept worse solution (simulated annealing-like)
                if np.random.random() < 0.1:
                    current = neighbor
                    current_fitness = neighbor_fitness
        
        return current
    
    def calculate_fitness(self, individual: np.ndarray) -> float:
        """
        Calculate fitness based on normalized MSE.
        
        Args:
            individual: Individual to evaluate
            
        Returns:
            float: Fitness value (higher is better)
        """
        self.fitness_eval_count += 1
        mse = np.mean((individual - self.target_image) ** 2)
        max_mse = 255.0 ** 2
        fitness = 1.0 - (mse / max_mse)
        return fitness
    
    def evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """
        Evaluate fitness for entire population.
        
        Args:
            population: Population to evaluate
            
        Returns:
            np.ndarray: Fitness values
        """
        return np.array([self.calculate_fitness(ind) for ind in population])
    
    def select_parents(self, population: np.ndarray, fitness: np.ndarray, n_parents: int) -> np.ndarray:
        """
        Select parents using the chosen selection strategy.
        
        Args:
            population: Current population
            fitness: Fitness values
            n_parents: Number of parents to select
            
        Returns:
            np.ndarray: Selected parents
        """
        if self.selection_strategy == 'random':
            indices = np.random.choice(len(population), n_parents, replace=True)
            return population[indices]
            
        elif self.selection_strategy == 'tournament':
            parents = []
            for _ in range(n_parents):
                tournament_indices = np.random.choice(len(population), self.tournament_size, replace=False)
                tournament_fitness = fitness[tournament_indices]
                winner_idx = tournament_indices[np.argmax(tournament_fitness)]
                parents.append(population[winner_idx])
            return np.array(parents)
            
        elif self.selection_strategy == 'roulette':
            # Handle negative fitness values
            adjusted_fitness = fitness - np.min(fitness) + 1e-10
            probabilities = adjusted_fitness / np.sum(adjusted_fitness)
            indices = np.random.choice(len(population), n_parents, p=probabilities, replace=True)
            return population[indices]
            
        elif self.selection_strategy == 'rank':
            # Rank-based selection
            ranks = np.argsort(np.argsort(fitness)) + 1
            probabilities = ranks / np.sum(ranks)
            indices = np.random.choice(len(population), n_parents, p=probabilities, replace=True)
            return population[indices]
        
        return population[:n_parents]
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform crossover using the chosen strategy.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Tuple of two offspring
        """
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        if self.crossover_strategy == 'single_point':
            point = np.random.randint(1, len(parent1))
            offspring1 = np.concatenate([parent1[:point], parent2[point:]])
            offspring2 = np.concatenate([parent2[:point], parent1[point:]])
            
        elif self.crossover_strategy == 'two_point':
            point1, point2 = sorted(np.random.choice(len(parent1), 2, replace=False))
            offspring1 = parent1.copy()
            offspring2 = parent2.copy()
            offspring1[point1:point2] = parent2[point1:point2]
            offspring2[point1:point2] = parent1[point1:point2]
            
        elif self.crossover_strategy == 'uniform':
            mask = np.random.random(len(parent1)) < 0.5
            offspring1 = np.where(mask, parent1, parent2)
            offspring2 = np.where(mask, parent2, parent1)
            
        elif self.crossover_strategy == 'arithmetic':
            alpha = np.random.random()
            offspring1 = alpha * parent1 + (1 - alpha) * parent2
            offspring2 = (1 - alpha) * parent1 + alpha * parent2
        
        else:
            offspring1, offspring2 = parent1.copy(), parent2.copy()
        
        return offspring1, offspring2
    
    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """
        Mutate an individual using the chosen strategy.
        
        Args:
            individual: Individual to mutate
            
        Returns:
            np.ndarray: Mutated individual
        """
        mutated = individual.copy()
        
        if self.mutation_strategy == 'bit_flip':
            # Randomly flip pixels to random values
            mask = np.random.random(len(mutated)) < self.mutation_rate
            mutated[mask] = np.random.uniform(0, 255, np.sum(mask))
            
        elif self.mutation_strategy == 'gaussian':
            # Add Gaussian noise to genes
            mask = np.random.random(len(mutated)) < self.mutation_rate
            noise = np.random.normal(0, 15, len(mutated))
            mutated[mask] += noise[mask]
            
        elif self.mutation_strategy == 'uniform':
            # Add uniform noise
            mask = np.random.random(len(mutated)) < self.mutation_rate
            noise = np.random.uniform(-20, 20, len(mutated))
            mutated[mask] += noise[mask]
            
        elif self.mutation_strategy == 'insertion':
            # Insertion mutation: select random positions and insert at different positions
            num_mutations = max(1, int(len(mutated) * self.mutation_rate))
            
            for _ in range(num_mutations):
                # Select two random positions
                pos1, pos2 = np.random.choice(len(mutated), 2, replace=False)
                pos1, pos2 = min(pos1, pos2), max(pos1, pos2)
                
                if pos1 != pos2:
                    # Extract the value at pos1
                    value = mutated[pos1]
                    # Shift elements and insert at pos2
                    if pos1 < pos2:
                        mutated[pos1:pos2] = mutated[pos1+1:pos2+1]
                        mutated[pos2] = value
                    else:
                        mutated[pos2+1:pos1+1] = mutated[pos2:pos1]
                        mutated[pos2] = value
        
        return np.clip(mutated, 0, 255)
    
    def survivor_selection(
        self, 
        population: np.ndarray, 
        fitness: np.ndarray, 
        offspring: np.ndarray, 
        offspring_fitness: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select survivors for next generation.
        
        Args:
            population: Current population
            fitness: Current fitness values
            offspring: Generated offspring
            offspring_fitness: Offspring fitness values
            
        Returns:
            Tuple of (new_population, new_fitness)
        """
        if self.survivor_strategy == 'generational':
            # Replace entire population with offspring
            return offspring, offspring_fitness
            
        elif self.survivor_strategy == 'elitist':
            # Combine and select best
            combined = np.vstack([population, offspring])
            combined_fitness = np.concatenate([fitness, offspring_fitness])
            
            # Sort by fitness
            sorted_indices = np.argsort(combined_fitness)[::-1]
            new_population = combined[sorted_indices[:self.population_size]]
            new_fitness = combined_fitness[sorted_indices[:self.population_size]]
            
            return new_population, new_fitness
            
        elif self.survivor_strategy == 'steady_state':
            # Replace worst parents with best offspring
            n_replace = len(offspring)
            
            # Get worst parents
            worst_indices = np.argsort(fitness)[:n_replace]
            
            # Get best offspring
            best_offspring_indices = np.argsort(offspring_fitness)[-n_replace:]
            
            # Replace
            new_population = population.copy()
            new_fitness = fitness.copy()
            new_population[worst_indices] = offspring[best_offspring_indices]
            new_fitness[worst_indices] = offspring_fitness[best_offspring_indices]
            
            return new_population, new_fitness
            
        elif self.survivor_strategy == 'tournament_replacement':
            # Tournament between parent and offspring
            new_population = []
            new_fitness = []
            
            for i in range(min(len(population), len(offspring))):
                if offspring_fitness[i] > fitness[i]:
                    new_population.append(offspring[i])
                    new_fitness.append(offspring_fitness[i])
                else:
                    new_population.append(population[i])
                    new_fitness.append(fitness[i])
            
            # Fill remaining slots if needed
            while len(new_population) < self.population_size:
                idx = np.random.randint(len(population))
                new_population.append(population[idx])
                new_fitness.append(fitness[idx])
            
            return np.array(new_population[:self.population_size]), np.array(new_fitness[:self.population_size])
        
        return population, fitness
    
    def calculate_diversity(self, population: np.ndarray) -> float:
        """
        Calculate population diversity as average pairwise distance.
        
        Args:
            population: Current population
            
        Returns:
            float: Diversity measure
        """
        if len(population) < 2:
            return 0.0
        
        # Sample subset for efficiency
        sample_size = min(50, len(population))
        sample = population[np.random.choice(len(population), sample_size, replace=False)]
        
        distances = []
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                dist = np.mean(np.abs(sample[i] - sample[j]))
                distances.append(dist)
        
        return np.mean(distances) if distances else 0.0
    
    def calculate_metrics(self, individual: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive metrics for an individual.
        
        Args:
            individual: Individual to evaluate
            
        Returns:
            dict: Metrics (MSE, PSNR, SSIM)
        """
        # MSE
        mse = np.mean((individual - self.target_image) ** 2)
        
        # PSNR
        if mse == 0:
            psnr = 100
        else:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        
        # SSIM
        img1 = individual.reshape(self.image_shape)
        img2 = self.target_image.reshape(self.image_shape)
        
        # Ensure proper data range for SSIM
        ssim_value = ssim(img2, img1, data_range=255)
        
        return {
            'mse': mse,
            'psnr': psnr,
            'ssim': ssim_value
        }
    
    def run(self, progress_callback: Callable = None) -> Dict[str, Any]:
        """
        Run the genetic algorithm.
        
        Args:
            progress_callback: Optional callback function for progress updates
            
        Returns:
            dict: Results including best individual and statistics
        """
        start_time = time.time()
        
        # Initialize population
        population = self.initialize_population()
        fitness = self.evaluate_population(population)
        
        # Track best individual
        best_idx = np.argmax(fitness)
        self.best_individual = population[best_idx].copy()
        self.best_fitness = fitness[best_idx]
        
        stagnation_counter = 0
        
        for generation in range(self.max_generations):
            # Generate offspring
            offspring = []
            
            # Keep elite individuals
            elite_indices = np.argsort(fitness)[-self.elite_size:]
            
            # Generate offspring through crossover and mutation
            while len(offspring) < self.population_size:
                # Select parents
                parents = self.select_parents(population, fitness, 2)
                
                # Crossover
                child1, child2 = self.crossover(parents[0], parents[1])
                
                # Mutation
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                offspring.extend([child1, child2])
            
            offspring = np.array(offspring[:self.population_size])
            offspring_fitness = self.evaluate_population(offspring)
            
            # Survivor selection
            population, fitness = self.survivor_selection(population, fitness, offspring, offspring_fitness)
            
            # Update best individual
            current_best_idx = np.argmax(fitness)
            current_best_fitness = fitness[current_best_idx]
            
            if current_best_fitness > self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_individual = population[current_best_idx].copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            # Calculate metrics
            metrics = self.calculate_metrics(self.best_individual)
            diversity = self.calculate_diversity(population)
            
            # Record history
            self.history['best_fitness'].append(self.best_fitness)
            self.history['avg_fitness'].append(np.mean(fitness))
            self.history['worst_fitness'].append(np.min(fitness))
            self.history['diversity'].append(diversity)
            self.history['mse'].append(metrics['mse'])
            self.history['psnr'].append(metrics['psnr'])
            self.history['ssim'].append(metrics['ssim'])
            self.history['fitness_evaluations'].append(self.fitness_eval_count)
            
            # Progress callback
            if progress_callback:
                progress_callback(generation, self.max_generations, self.best_fitness, metrics)
            
            # Termination criteria
            if self.best_fitness >= self.fitness_threshold:
                break
            
            if stagnation_counter >= self.max_stagnation:
                break
        
        end_time = time.time()
        
        return {
            'best_individual': self.best_individual.reshape(self.image_shape),
            'best_fitness': self.best_fitness,
            'history': self.history,
            'generations': generation + 1,
            'fitness_evaluations': self.fitness_eval_count,
            'execution_time': end_time - start_time,
            'final_metrics': self.calculate_metrics(self.best_individual),
            'termination_reason': self._get_termination_reason(generation, stagnation_counter)
        }
    
    def _get_termination_reason(self, generation: int, stagnation_counter: int) -> str:
        """Get the reason for termination."""
        if self.best_fitness >= self.fitness_threshold:
            return f"Fitness threshold reached ({self.best_fitness:.4f} >= {self.fitness_threshold})"
        elif stagnation_counter >= self.max_stagnation:
            return f"Stagnation limit reached ({stagnation_counter} generations without improvement)"
        elif generation >= self.max_generations - 1:
            return f"Maximum generations reached ({self.max_generations})"
        else:
            return "Unknown"
