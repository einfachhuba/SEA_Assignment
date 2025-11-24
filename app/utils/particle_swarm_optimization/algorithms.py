import numpy as np
from .functions import evaluate_subset
from .config import DEFAULT_SELECTION_THRESHOLD

class Particle:
    def __init__(self, num_features):
        self.position = np.random.uniform(0, 1, num_features)
        self.velocity = np.random.uniform(-0.1, 0.1, num_features)
        self.pbest_position = self.position.copy()
        self.pbest_fitness = -float('inf')
        self.pbest_accuracy = 0.0
        self.fitness = -float('inf')
        self.accuracy = 0.0

class PSO_FeatureSelection:
    def __init__(
        self,
        num_particles,
        num_iterations,
        w,
        c1,
        c2,
        alpha,
        X_train,
        X_test,
        y_train,
        y_test,
        model_params=None,
        selection_threshold=DEFAULT_SELECTION_THRESHOLD,
        c_global=1.0,
        neighbor_count=5,
    ):
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.c_global = c_global
        self.alpha = alpha
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.num_features = X_train.shape[1]
        
        self.particles = [Particle(self.num_features) for _ in range(num_particles)]
        self.gbest_position = np.zeros(self.num_features)
        self.gbest_fitness = -float('inf')
        self.gbest_accuracy = 0.0
        self.history = []
        self.model_params = model_params or {}
        self.selection_threshold = selection_threshold
        allowable = max(0, self.num_particles - 1)
        self.neighbor_count = max(0, min(neighbor_count, allowable))

    def _calculate_fitness(self, position):
        selected_indices = np.where(position > self.selection_threshold)[0]
        if len(selected_indices) == 0:
            return -1.0, 0.0 # Penalize empty selection heavily
            
        accuracy = evaluate_subset(
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            selected_indices,
            self.model_params,
        )
        
        # Assignment formula: alpha * acc + (1 - alpha) * (1 - n/N)
        selection_ratio = len(selected_indices) / self.num_features
        diversity_term = (1 - self.alpha) * (1 - selection_ratio)
        fitness = (self.alpha * accuracy) + diversity_term
        
        return fitness, accuracy

    def _nearest_neighbor_best(self, idx, positions):
        if self.neighbor_count <= 0:
            return self.gbest_position

        target = positions[idx]
        distances = np.linalg.norm(positions - target, axis=1)
        neighbor_idx = np.argsort(distances)[1 : self.neighbor_count + 1]
        if len(neighbor_idx) == 0:
            return self.gbest_position
        best_idx = max(neighbor_idx, key=lambda i: self.particles[i].pbest_fitness)
        return self.particles[best_idx].pbest_position

    def run(self, progress_callback=None):
        # Initialization
        for p in self.particles:
            fitness, accuracy = self._calculate_fitness(p.position)
            p.fitness = fitness
            p.accuracy = accuracy
            p.pbest_fitness = fitness
            p.pbest_accuracy = accuracy
            p.pbest_position = p.position.copy()
            
            if fitness > self.gbest_fitness:
                self.gbest_fitness = fitness
                self.gbest_accuracy = accuracy
                self.gbest_position = p.position.copy()
        
        # Loop
        for it in range(self.num_iterations):
            positions = np.array([p.position for p in self.particles])
            for idx, p in enumerate(self.particles):
                # Update velocity
                r1 = np.random.rand(self.num_features)
                r2 = np.random.rand(self.num_features)
                r3 = np.random.rand(self.num_features)

                neighbor_best = self._nearest_neighbor_best(idx, positions)
                
                p.velocity = (
                    self.w * p.velocity
                    + self.c1 * r1 * (p.pbest_position - p.position)
                    + self.c2 * r2 * (neighbor_best - p.position)
                    + self.c_global * r3 * (self.gbest_position - p.position)
                )
                
                # Clamp velocity to avoid explosion
                p.velocity = np.clip(p.velocity, -0.5, 0.5)
                
                # Update position
                p.position = p.position + p.velocity
                p.position = np.clip(p.position, 0, 1) # Keep in [0, 1]
                
                # Evaluate
                fitness, accuracy = self._calculate_fitness(p.position)
                p.fitness = fitness
                p.accuracy = accuracy
                
                # Update Personal Best
                if fitness > p.pbest_fitness:
                    p.pbest_fitness = fitness
                    p.pbest_accuracy = accuracy
                    p.pbest_position = p.position.copy()
                    
                    # Update Global Best
                    if fitness > self.gbest_fitness:
                        self.gbest_fitness = fitness
                        self.gbest_accuracy = accuracy
                        self.gbest_position = p.position.copy()
            
            # Record history
            self.history.append({
                'iteration': it,
                'fitness': self.gbest_fitness,
                'accuracy': self.gbest_accuracy,
                'num_features': len(np.where(self.gbest_position > self.selection_threshold)[0])
            })
            
            if progress_callback:
                progress_callback(it + 1, self.gbest_fitness, self.gbest_accuracy)
                
        return {
            "best_fitness": self.gbest_fitness,
            "best_accuracy": self.gbest_accuracy,
            "best_position": self.gbest_position,
            "best_features": np.where(self.gbest_position > self.selection_threshold)[0],
            "history": self.history
        }
