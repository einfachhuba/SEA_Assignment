"""
Configuration settings for Genetic Algorithm Variations (Image Reconstruction)
"""

import numpy as np

# Image dimensions (start with 16x16, can be increased)
DEFAULT_IMAGE_SIZE = (16, 16)
PIXEL_BOUNDS = (0, 255)  # Grayscale pixel values

# GA Parameters
DEFAULT_POPULATION_SIZE = 100
DEFAULT_MAX_GENERATIONS = 500
DEFAULT_MUTATION_RATE = 0.01
DEFAULT_CROSSOVER_RATE = 0.8
DEFAULT_TOURNAMENT_SIZE = 3
DEFAULT_ELITE_SIZE = 5

# Termination criteria
FITNESS_THRESHOLD = 0.98  # If fitness reaches this value (as a fraction of max possible)
MAX_STAGNATION_GENERATIONS = 50  # Stop if no improvement for this many generations

# Selection strategies
SELECTION_STRATEGIES = {
    'random': 'Random Selection',
    'rank': 'Rank-based Selection',
    'roulette': 'Roulette Wheel Selection',
    'tournament': 'Tournament Selection'
}

# Crossover strategies
CROSSOVER_STRATEGIES = {
    'arithmetic': 'Arithmetic Crossover',
    'single_point': 'Single Point Crossover',
    'two_point': 'Two Point Crossover',
    'uniform': 'Uniform Crossover'
}

# Mutation strategies
MUTATION_STRATEGIES = {
    'bit_flip': 'Bit Flip Mutation',
    'gaussian': 'Gaussian Mutation',
    'insertion': 'Insertion Mutation',
    'uniform': 'Uniform Mutation'
}

# Survivor selection strategies
SURVIVOR_STRATEGIES = {
    'elitist': 'Elitist Replacement',
    'generational': 'Generational Replacement',
    'steady_state': 'Steady State',
    'tournament_replacement': 'Tournament Replacement'
}

# Initialization strategies
INITIALIZATION_STRATEGIES = {
    'edge_based': 'Edge-based Initialization',
    'gaussian_noise': 'Gaussian Noise Initialization',
    'local_optimization': 'Local Optimization (Hill Climbing)',
    'oversampling_selection': 'Oversampling and Selection',
    'random': 'Random Initialization'
}
