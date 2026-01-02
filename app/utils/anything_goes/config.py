"""
Configuration for Neural Architecture Search using Differential Evolution.
Defines the CNN architecture search space and DE hyperparameters.
"""

# Architecture Search Space Bounds (continuous encoding)
# Each dimension will be mapped to discrete choices via binning

ARCHITECTURE_BOUNDS = {
    "num_conv_layers": (1, 3),           # Number of convolutional blocks (1, 2, or 3)
    "filters_1": (8, 32),                # Filters in first conv layer (8, 16, 24, 32)
    "filters_2": (8, 32),                # Filters in second conv layer (if present)
    "filters_3": (8, 32),                # Filters in third conv layer (if present)
    "kernel_size_1": (0, 1),             # Kernel size for layer 1: 0->3x3, 1->5x5
    "kernel_size_2": (0, 1),             # Kernel size for layer 2
    "kernel_size_3": (0, 1),             # Kernel size for layer 3
    "pooling_1": (0, 1),                 # Pooling type: 0->max, 1->avg
    "pooling_2": (0, 1),                 # Pooling type layer 2
    "pooling_3": (0, 1),                 # Pooling type layer 3
    "dropout_1": (0, 1),                 # Dropout enabled: 0->no, 1->yes
    "dropout_2": (0, 1),                 # Dropout enabled layer 2
    "dropout_3": (0, 1),                 # Dropout enabled layer 3
    "dropout_rate_1": (0, 0.5),          # Dropout rate (if enabled)
    "dropout_rate_2": (0, 0.5),          # Dropout rate layer 2
    "dropout_rate_3": (0, 0.5),          # Dropout rate layer 3
    "fc_units": (32, 128),               # Fully connected layer units (32, 64, 96, 128)
}

# Discrete options for filter counts
FILTER_OPTIONS = [8, 16, 24, 32]

# Discrete options for kernel sizes
KERNEL_OPTIONS = [3, 5]

# Discrete options for pooling types
POOLING_OPTIONS = ["max", "avg"]

# Discrete options for FC units
FC_OPTIONS = [32, 64, 96, 128]

# Training Configuration
TRAINING_CONFIG = {
    "epochs": 4,                         # Small number for fast evaluation
    "learning_rate": 1e-3,               # Fixed learning rate as per assignment
    "batch_size": 128,                   # Reasonable batch size
    "num_workers": 0,                    # Set to 0 to avoid multiprocessing issues on Windows
    "device": "cuda",                    # Will fallback to CPU if not available
}

# Dataset Configuration
DATASET_CONFIG = {
    "train_samples": 15000,              # Subsample for faster training
    "val_samples": 3000,                 # Validation subset
    "test_samples": 2000,                # Test subset for final evaluation
    "num_classes": 10,                   # Fashion-MNIST has 10 classes
    "input_channels": 1,                 # Grayscale images
    "image_size": 28,                    # 28x28 pixels
}

# Differential Evolution Parameters
DE_CONFIG = {
    "population_size": 12,               # Number of candidate architectures
    "generations": 8,                    # Number of evolution iterations
    "mutation_factor": 0.6,              # F: differential weight (0.5-1.0 typical)
    "crossover_rate": 0.8,               # CR: crossover probability (0.7-0.9 typical)
    "seed": 42,                          # For reproducibility
}

# Fitness Function Weights
FITNESS_WEIGHTS = {
    "accuracy_weight": 1.0,              # Primary objective: maximize accuracy
    "param_penalty": 0.15,               # Penalty for model size (parameters)
    "time_penalty": 0.10,                # Penalty for training time
    "param_limit": 100000,               # Target parameter limit
    "time_limit": 30.0,                  # Target time per epoch (seconds)
}

# Fashion-MNIST Class Labels
FASHION_MNIST_LABELS = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]
