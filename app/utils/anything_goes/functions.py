"""
Functions for Neural Architecture Search: CNN construction, training, and fitness evaluation.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import Dict, Tuple, Optional
import json
import os

from .config import (
    ARCHITECTURE_BOUNDS, FILTER_OPTIONS, KERNEL_OPTIONS, POOLING_OPTIONS, FC_OPTIONS,
    TRAINING_CONFIG, DATASET_CONFIG, FITNESS_WEIGHTS
)


class CNNArchitecture(nn.Module):
    """
    Dynamically constructed CNN based on architecture parameters.
    """
    def __init__(self, arch_params: Dict):
        super(CNNArchitecture, self).__init__()
        self.arch_params = arch_params
        
        layers = []
        in_channels = DATASET_CONFIG["input_channels"]
        current_size = DATASET_CONFIG["image_size"]
        
        # Build convolutional blocks
        num_conv = arch_params["num_conv_layers"]
        for i in range(1, num_conv + 1):
            out_channels = arch_params[f"filters_{i}"]
            kernel_size = arch_params[f"kernel_size_{i}"]
            pooling_type = arch_params[f"pooling_{i}"]
            dropout_enabled = arch_params[f"dropout_{i}"]
            dropout_rate = arch_params[f"dropout_rate_{i}"]
            
            # Convolutional layer
            padding = kernel_size // 2  # Keep spatial dimensions
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            
            # Pooling layer (2x2)
            if pooling_type == "max":
                layers.append(nn.MaxPool2d(2))
            else:
                layers.append(nn.AvgPool2d(2))
            current_size = current_size // 2
            
            # Dropout
            if dropout_enabled:
                layers.append(nn.Dropout2d(dropout_rate))
            
            in_channels = out_channels
        
        self.features = nn.Sequential(*layers)
        
        # Calculate flattened size
        self.flat_size = in_channels * current_size * current_size
        
        # Fully connected layers
        fc_units = arch_params["fc_units"]
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_size, fc_units),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fc_units, DATASET_CONFIG["num_classes"])
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def decode_architecture(vector: np.ndarray) -> Dict:
    """
    Decode continuous DE vector [0,1] into discrete architecture parameters.
    
    Args:
        vector: Continuous vector with values in [0, 1]
    
    Returns:
        Dictionary of discrete architecture parameters
    """
    arch = {}
    idx = 0
    
    # Number of conv layers (1, 2, or 3)
    arch["num_conv_layers"] = int(np.clip(np.round(vector[idx] * 2 + 1), 1, 3))
    idx += 1
    
    # For each possible layer, decode parameters
    for i in range(1, 4):
        # Filters: map to [8, 16, 24, 32]
        filter_idx = int(np.clip(vector[idx] * len(FILTER_OPTIONS), 0, len(FILTER_OPTIONS) - 1))
        arch[f"filters_{i}"] = FILTER_OPTIONS[filter_idx]
        idx += 1
        
        # Kernel size: 0->3, 1->5
        kernel_idx = int(np.round(vector[idx]))
        arch[f"kernel_size_{i}"] = KERNEL_OPTIONS[kernel_idx]
        idx += 1
        
        # Pooling type: 0->max, 1->avg
        pooling_idx = int(np.round(vector[idx]))
        arch[f"pooling_{i}"] = POOLING_OPTIONS[pooling_idx]
        idx += 1
        
        # Dropout enabled: threshold at 0.5
        arch[f"dropout_{i}"] = vector[idx] > 0.5
        idx += 1
        
        # Dropout rate: continuous [0, 0.5]
        arch[f"dropout_rate_{i}"] = float(np.clip(vector[idx] * 0.5, 0, 0.5))
        idx += 1
    
    # FC units: map to [32, 64, 96, 128]
    fc_idx = int(np.clip(vector[idx] * len(FC_OPTIONS), 0, len(FC_OPTIONS) - 1))
    arch["fc_units"] = FC_OPTIONS[fc_idx]
    
    return arch


def get_architecture_hash(arch_params: Dict) -> str:
    """Create a unique string representation of architecture for caching."""
    # Only include parameters up to num_conv_layers to avoid unnecessary differences
    num_conv = arch_params["num_conv_layers"]
    key_parts = [f"conv{num_conv}"]
    for i in range(1, num_conv + 1):
        key_parts.append(f"f{arch_params[f'filters_{i}']}")
        key_parts.append(f"k{arch_params[f'kernel_size_{i}']}")
        key_parts.append(f"p{arch_params[f'pooling_{i}']}")
        key_parts.append(f"d{arch_params[f'dropout_{i}']}")
        if arch_params[f"dropout_{i}"]:
            key_parts.append(f"dr{arch_params[f'dropout_rate_{i}']:.2f}")
    key_parts.append(f"fc{arch_params['fc_units']}")
    return "_".join(key_parts)


def load_fashion_mnist(train_samples: int, val_samples: int, test_samples: int, 
                       batch_size: int = 128) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load Fashion-MNIST dataset with subsampling.
    
    Args:
        train_samples: Number of training samples to use
        val_samples: Number of validation samples
        test_samples: Number of test samples
        batch_size: Batch size for dataloaders
    
    Returns:
        train_loader, val_loader, test_loader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    
    # Load full datasets
    train_dataset = datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Create subsets
    train_indices = np.random.choice(len(train_dataset), train_samples, replace=False)
    val_indices = np.random.choice(len(train_dataset), val_samples, replace=False)
    test_indices = np.random.choice(len(test_dataset), test_samples, replace=False)
    
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)
    test_subset = Subset(test_dataset, test_indices)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, 
        num_workers=TRAINING_CONFIG["num_workers"]
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=TRAINING_CONFIG["num_workers"]
    )
    test_loader = DataLoader(
        test_subset, batch_size=batch_size, shuffle=False,
        num_workers=TRAINING_CONFIG["num_workers"]
    )
    
    return train_loader, val_loader, test_loader


def train_and_evaluate(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                       device: torch.device, epochs: int = 4, lr: float = 1e-3) -> Dict:
    """
    Train and evaluate a CNN model.
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        epochs: Number of training epochs
        lr: Learning rate
    
    Returns:
        Dictionary with training metrics
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_accuracies = []
    epoch_times = []
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100.0 * correct / total
        val_accuracies.append(val_acc)
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
    
    return {
        "train_losses": train_losses,
        "val_accuracies": val_accuracies,
        "epoch_times": epoch_times,
        "final_val_accuracy": val_accuracies[-1],
        "avg_epoch_time": np.mean(epoch_times),
    }


def calculate_fitness(arch_params: Dict, val_accuracy: float, num_params: int, 
                     avg_epoch_time: float) -> Tuple[float, Dict]:
    """
    Calculate fitness score with penalties for model size and training time.
    
    Fitness = accuracy - param_penalty - time_penalty
    
    Args:
        arch_params: Architecture parameters
        val_accuracy: Validation accuracy (0-100)
        num_params: Number of model parameters
        avg_epoch_time: Average time per epoch in seconds
    
    Returns:
        fitness_score, penalty_details
    """
    # Normalize accuracy to [0, 1]
    norm_accuracy = val_accuracy / 100.0
    
    # Parameter penalty (linear penalty above limit)
    param_ratio = num_params / FITNESS_WEIGHTS["param_limit"]
    param_penalty = FITNESS_WEIGHTS["param_penalty"] * max(0, param_ratio - 1.0)
    
    # Time penalty (linear penalty above limit)
    time_ratio = avg_epoch_time / FITNESS_WEIGHTS["time_limit"]
    time_penalty = FITNESS_WEIGHTS["time_penalty"] * max(0, time_ratio - 1.0)
    
    # Total fitness
    fitness = (FITNESS_WEIGHTS["accuracy_weight"] * norm_accuracy 
               - param_penalty - time_penalty)
    
    penalty_details = {
        "param_penalty": param_penalty,
        "time_penalty": time_penalty,
        "param_ratio": param_ratio,
        "time_ratio": time_ratio,
    }
    
    return fitness, penalty_details


def evaluate_architecture(vector: np.ndarray, train_loader: DataLoader, 
                         val_loader: DataLoader, device: torch.device,
                         cache: Optional[Dict] = None) -> Dict:
    """
    Complete evaluation pipeline for a candidate architecture.
    
    Args:
        vector: DE vector encoding the architecture
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        cache: Optional cache to avoid re-evaluating identical architectures
    
    Returns:
        Dictionary with all evaluation metrics
    """
    # Decode architecture
    arch_params = decode_architecture(vector)
    arch_hash = get_architecture_hash(arch_params)
    
    # Check cache
    if cache is not None and arch_hash in cache:
        return cache[arch_hash]
    
    # Build model
    try:
        model = CNNArchitecture(arch_params)
        num_params = model.count_parameters()
    except Exception as e:
        # Invalid architecture - return poor fitness
        return {
            "fitness": -1.0,
            "val_accuracy": 0.0,
            "num_params": 0,
            "avg_epoch_time": 0.0,
            "arch_params": arch_params,
            "arch_hash": arch_hash,
            "error": str(e),
        }
    
    # Train and evaluate
    training_results = train_and_evaluate(
        model, train_loader, val_loader, device,
        epochs=TRAINING_CONFIG["epochs"],
        lr=TRAINING_CONFIG["learning_rate"]
    )
    
    # Calculate fitness
    fitness, penalty_details = calculate_fitness(
        arch_params,
        training_results["final_val_accuracy"],
        num_params,
        training_results["avg_epoch_time"]
    )
    
    # Compile results
    result = {
        "fitness": fitness,
        "val_accuracy": training_results["final_val_accuracy"],
        "num_params": num_params,
        "avg_epoch_time": training_results["avg_epoch_time"],
        "arch_params": arch_params,
        "arch_hash": arch_hash,
        "train_losses": training_results["train_losses"],
        "val_accuracies": training_results["val_accuracies"],
        "epoch_times": training_results["epoch_times"],
        **penalty_details,
    }
    
    # Cache result
    if cache is not None:
        cache[arch_hash] = result
    
    return result


def test_final_model(arch_params: Dict, train_loader: DataLoader, test_loader: DataLoader,
                     device: torch.device, num_runs: int = 2) -> Dict:
    """
    Retrain and test the best architecture multiple times for performance validation.
    
    Args:
        arch_params: Architecture parameters
        train_loader: Training data loader
        test_loader: Test data loader
        device: Device to train on
        num_runs: Number of independent runs
    
    Returns:
        Dictionary with test performance statistics
    """
    test_accuracies = []
    test_times = []
    
    for run in range(num_runs):
        # Build fresh model
        model = CNNArchitecture(arch_params)
        
        # Train on full train+val data
        start_time = time.time()
        _ = train_and_evaluate(
            model, train_loader, test_loader, device,
            epochs=TRAINING_CONFIG["epochs"],
            lr=TRAINING_CONFIG["learning_rate"]
        )
        train_time = time.time() - start_time
        
        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_acc = 100.0 * correct / total
        test_accuracies.append(test_acc)
        test_times.append(train_time)
    
    return {
        "test_accuracies": test_accuracies,
        "mean_test_accuracy": np.mean(test_accuracies),
        "std_test_accuracy": np.std(test_accuracies),
        "test_times": test_times,
        "mean_test_time": np.mean(test_times),
    }


def get_predictions_and_examples(model: nn.Module, test_loader: DataLoader, 
                                 device: torch.device, num_examples: int = 16) -> Dict:
    """
    Get predictions and sample images from test set for visualization.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device for inference
        num_examples: Number of example images to retrieve
    
    Returns:
        Dictionary with predictions, labels, images, and confusion matrix data
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    example_images = []
    example_predictions = []
    example_labels = []
    examples_collected = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Collect example images
            if examples_collected < num_examples:
                batch_size = inputs.size(0)
                remaining = num_examples - examples_collected
                num_to_take = min(batch_size, remaining)
                
                for i in range(num_to_take):
                    example_images.append(inputs[i].cpu())
                    example_predictions.append(predicted[i].item())
                    example_labels.append(labels[i].item())
                
                examples_collected += num_to_take
    
    return {
        "all_predictions": np.array(all_predictions),
        "all_labels": np.array(all_labels),
        "example_images": example_images,
        "example_predictions": example_predictions,
        "example_labels": example_labels,
    }
