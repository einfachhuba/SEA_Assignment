"""
Differential Evolution algorithm for Neural Architecture Search.
"""

import numpy as np
import torch
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass

from .config import ARCHITECTURE_BOUNDS, DE_CONFIG
from .functions import evaluate_architecture


@dataclass
class DEResult:
    """Results from Differential Evolution search."""
    best_vector: np.ndarray
    best_fitness: float
    best_architecture: Dict
    best_metrics: Dict
    population_history: List[np.ndarray]
    fitness_history: List[np.ndarray]
    best_fitness_history: List[float]
    evaluation_cache: Dict
    generation_details: List[Dict]


class DifferentialEvolutionNAS:
    """
    Differential Evolution solver for Neural Architecture Search.
    
    Uses DE/rand/1/bin strategy with architecture-specific evaluation.
    """
    
    def __init__(
        self,
        train_loader,
        val_loader,
        device: torch.device,
        population_size: int = 12,
        generations: int = 8,
        mutation_factor: float = 0.6,
        crossover_rate: float = 0.8,
        seed: Optional[int] = None,
    ):
        """
        Initialize DE solver.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device for training
            population_size: Number of candidate architectures
            generations: Number of evolution iterations
            mutation_factor: F parameter for mutation
            crossover_rate: CR parameter for crossover
            seed: Random seed for reproducibility
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.population_size = population_size
        self.generations = generations
        self.mutation_factor = mutation_factor
        self.crossover_rate = crossover_rate
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        # Extract bounds
        self.bounds = list(ARCHITECTURE_BOUNDS.values())
        self.dim = len(self.bounds)
        
        # Cache for evaluated architectures
        self.cache = {}
        
        # History tracking
        self.population_history = []
        self.fitness_history = []
        self.best_fitness_history = []
        self.generation_details = []
    
    def initialize_population(self) -> np.ndarray:
        """
        Initialize population with uniform random sampling in [0, 1].
        
        Returns:
            population: Array of shape (population_size, dim)
        """
        population = np.random.rand(self.population_size, self.dim)
        return population
    
    def evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """
        Evaluate fitness for all individuals in population.
        
        Args:
            population: Array of shape (population_size, dim)
        
        Returns:
            fitness_scores: Array of shape (population_size,)
        """
        fitness_scores = np.zeros(self.population_size)
        
        for i in range(self.population_size):
            result = evaluate_architecture(
                population[i], 
                self.train_loader, 
                self.val_loader,
                self.device,
                cache=self.cache
            )
            fitness_scores[i] = result["fitness"]
        
        return fitness_scores
    
    def mutate(self, population: np.ndarray, target_idx: int) -> np.ndarray:
        """
        mutation: v = x_a + F * (x_b - x_c)
        
        Args:
            population: Current population
            target_idx: Index of target vector (to avoid in selection)
        
        Returns:
            mutant_vector: Mutated vector
        """
        # Select three distinct random individuals (excluding target)
        indices = [i for i in range(self.population_size) if i != target_idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        
        # Mutation
        mutant = population[a] + self.mutation_factor * (population[b] - population[c])
        
        # Clip to [0, 1]
        mutant = np.clip(mutant, 0, 1)
        
        return mutant
    
    def crossover(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        """
        Binomial crossover between target and mutant.
        
        Args:
            target: Target vector
            mutant: Mutant vector
        
        Returns:
            trial: Trial vector
        """
        trial = target.copy()
        
        # Ensure at least one dimension comes from mutant
        j_rand = np.random.randint(self.dim)
        
        for j in range(self.dim):
            if np.random.rand() < self.crossover_rate or j == j_rand:
                trial[j] = mutant[j]
        
        return trial
    
    def run(self, progress_callback: Optional[Callable] = None) -> DEResult:
        """
        Run the Differential Evolution algorithm.
        
        Args:
            progress_callback: Optional callback(gen, best_fitness, mean_fitness, cache_size)
        
        Returns:
            DEResult with best architecture and search history
        """
        # Initialize population
        population = self.initialize_population()
        fitness = self.evaluate_population(population)
        
        # Track best
        best_idx = np.argmax(fitness)
        best_vector = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        # Store initial state
        self.population_history.append(population.copy())
        self.fitness_history.append(fitness.copy())
        self.best_fitness_history.append(best_fitness)
        
        gen_detail = {
            "generation": 0,
            "best_fitness": best_fitness,
            "mean_fitness": np.mean(fitness),
            "std_fitness": np.std(fitness),
            "cache_size": len(self.cache),
        }
        self.generation_details.append(gen_detail)
        
        if progress_callback:
            progress_callback(0, best_fitness, np.mean(fitness), len(self.cache))
        
        # Evolution loop
        for gen in range(1, self.generations + 1):
            new_population = population.copy()
            
            for i in range(self.population_size):
                # Mutation
                mutant = self.mutate(population, i)
                
                # Crossover
                trial = self.crossover(population[i], mutant)
                
                # Evaluate trial
                trial_result = evaluate_architecture(
                    trial, self.train_loader, self.val_loader, 
                    self.device, cache=self.cache
                )
                trial_fitness = trial_result["fitness"]
                
                # Selection
                if trial_fitness >= fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                    
                    # Update global best
                    if trial_fitness > best_fitness:
                        best_fitness = trial_fitness
                        best_vector = trial.copy()
            
            # Update population
            population = new_population
            
            # Store history
            self.population_history.append(population.copy())
            self.fitness_history.append(fitness.copy())
            self.best_fitness_history.append(best_fitness)
            
            gen_detail = {
                "generation": gen,
                "best_fitness": best_fitness,
                "mean_fitness": np.mean(fitness),
                "std_fitness": np.std(fitness),
                "cache_size": len(self.cache),
            }
            self.generation_details.append(gen_detail)
            
            if progress_callback:
                progress_callback(gen, best_fitness, np.mean(fitness), len(self.cache))
        
        # Retrieve best architecture details
        best_metrics = self.cache[evaluate_architecture(
            best_vector, self.train_loader, self.val_loader,
            self.device, cache=self.cache
        )["arch_hash"]]
        
        return DEResult(
            best_vector=best_vector,
            best_fitness=best_fitness,
            best_architecture=best_metrics["arch_params"],
            best_metrics=best_metrics,
            population_history=self.population_history,
            fitness_history=self.fitness_history,
            best_fitness_history=self.best_fitness_history,
            evaluation_cache=self.cache,
            generation_details=self.generation_details,
        )
