from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from .functions import route_length, init_route_nearest_neighbor


@dataclass
class ACOHistory:
	best_distances: List[float]
	iteration_best_distances: List[float]
	diversity: List[float]
	pheromone_snapshots: List[tuple[int, np.ndarray]]  # (iteration, pheromone_matrix)


class AntColonyOptimizationTSP:
	"""
	Ant Colony Optimization for solving the Traveling Salesperson Problem (TSP).

	The algorithm simulates a colony of ants that construct solutions probabilistically
	based on pheromone trails and heuristic information (distance). Shorter routes
	deposit more pheromone, reinforcing good solutions over time.

	Contract:
	- Input: distance matrix (NxN) and algorithm parameters
	- Output: dictionary with best route, best distance, and history
	- Error modes: if N<2, raises ValueError
	"""

	def __init__(
		self,
		dist: np.ndarray,
		*,
		num_ants: int = 20,
		num_iterations: int = 100,
		alpha: float = 1.0,
		beta: float = 2.0,
		evaporation_rate: float = 0.5,
		pheromone_constant: float = 100.0,
		elitist_weight: float = 0.0,
		initialization_strategy: str = "random",
		seed: Optional[int] = None,
	) -> None:
		"""
		Initialize Ant Colony Optimization solver.

		Args:
			dist: NxN distance matrix
			num_ants: Number of ants in the colony
			num_iterations: Number of iterations to run
			alpha: Pheromone importance ($\\alpha$)
			beta: Heuristic (distance) importance ($\\beta$)
			evaporation_rate: Pheromone evaporation rate ($\\rho$) in [0,1]
			pheromone_constant: Pheromone deposit constant (Q)
			elitist_weight: Additional pheromone weight for best solution
			initialization_strategy: "random" or "nearest_neighbor" for initial pheromone
			seed: Random seed for reproducibility
		"""
		if dist.shape[0] < 2:
			raise ValueError("Need at least two cities for ACO")
		
		self.dist = dist
		self.n = dist.shape[0]
		self.num_ants = int(num_ants)
		self.num_iterations = int(num_iterations)
		self.alpha = float(alpha)
		self.beta = float(beta)
		self.evaporation_rate = float(evaporation_rate)
		self.pheromone_constant = float(pheromone_constant)
		self.elitist_weight = float(elitist_weight)
		self.initialization_strategy = initialization_strategy
		self.rng = random.Random(seed)
		
		# Initialize pheromone matrix
		self.pheromone = self._initialize_pheromone()
		
		# Heuristic information: theta = 1/distance (inversely proportional to distance)
		# Avoid division by zero for diagonal (self-loops)
		self.heuristic = np.zeros_like(dist)
		for i in range(self.n):
			for j in range(self.n):
				if i != j and dist[i, j] > 0:
					self.heuristic[i, j] = 1.0 / dist[i, j]
	
	def _initialize_pheromone(self) -> np.ndarray:
		"""Initialize pheromone trails."""
		if self.initialization_strategy == "nearest_neighbor":
			# Use nearest neighbor heuristic to get initial tour length
			nn_route = init_route_nearest_neighbor(self.n, self.dist, self.rng)
			nn_length = route_length(nn_route, self.dist)
			# Initialize with t_0 = 1 / (n * L_nn)
			initial_pheromone = 1.0 / (self.n * nn_length) if nn_length > 0 else 0.01
		else:
			# Random initialization
			initial_pheromone = 0.01
		
		return np.full((self.n, self.n), initial_pheromone, dtype=float)
	
	def _construct_solution(self, start_city: int) -> List[int]:
		"""
		Construct a solution (tour) for one ant starting from start_city.
		
		Uses probabilistic decision rule based on pheromone and heuristic information.
		"""
		route = [start_city]
		unvisited = set(range(self.n))
		unvisited.remove(start_city)
		
		current = start_city
		
		while unvisited:
			# Calculate probabilities for each unvisited city
			probs = []
			cities = []
			
			for city in unvisited:
				tau = self.pheromone[current, city]
				eta = self.heuristic[current, city]
				prob = (tau ** self.alpha) * (eta ** self.beta)
				probs.append(prob)
				cities.append(city)
			
			# Normalize probabilities
			total = sum(probs)
			if total > 0:
				probs = [p / total for p in probs]
			else:
				# Fallback to uniform if all probabilities are zero
				probs = [1.0 / len(cities)] * len(cities)
			
			# Select next city probabilistically
			next_city = self.rng.choices(cities, weights=probs, k=1)[0]
			
			route.append(next_city)
			unvisited.remove(next_city)
			current = next_city
		
		return route
	
	def _update_pheromones(self, solutions: List[tuple[List[int], float]], best_route: List[int], best_length: float):
		"""
		Update pheromone trails based on solutions found by ants.
		
		Args:
			solutions: List of (route, length) tuples for all ants
			best_route: Best route found so far (for elitist variant)
			best_length: Length of best route
		"""
		# Evaporation
		self.pheromone *= (1.0 - self.evaporation_rate)
		
		# Add pheromone from each ant
		for route, length in solutions:
			if length > 0:
				deposit = self.pheromone_constant / length
				for i in range(len(route)):
					city_from = route[i]
					city_to = route[(i + 1) % len(route)]
					self.pheromone[city_from, city_to] += deposit
					self.pheromone[city_to, city_from] += deposit  # Symmetric for TSP
		
		# Elitist ant: add extra pheromone to best solution
		if self.elitist_weight > 0 and best_length > 0:
			elitist_deposit = self.elitist_weight * self.pheromone_constant / best_length
			for i in range(len(best_route)):
				city_from = best_route[i]
				city_to = best_route[(i + 1) % len(best_route)]
				self.pheromone[city_from, city_to] += elitist_deposit
				self.pheromone[city_to, city_from] += elitist_deposit
	
	def _calculate_diversity(self, solutions: List[tuple[List[int], float]]) -> float:
		"""Calculate diversity of solutions (standard deviation of lengths)."""
		lengths = [length for _, length in solutions]
		if len(lengths) > 1:
			return float(np.std(lengths))
		return 0.0
	
	def run(
		self,
		progress_callback: Optional[Callable[[int, float, float], None]] = None,
		viz_callback: Optional[Callable[[int, List[int], float], None]] = None,
		viz_interval: int = 10
	) -> Dict:
		"""
		Run the Ant Colony Optimization algorithm.

		Args:
			progress_callback: Optional callback(iteration, best_len, iter_best_len)
			viz_callback: Optional callback(iteration, best_route, best_len) for visualization
			viz_interval: Visualize every N iterations

		Returns:
			Dictionary with:
				- best_route: Best route found
				- best_distance: Length of best route
				- iterations: Number of iterations run
				- history: ACOHistory with convergence data
		"""
		# Initialize best solution
		best_route = None
		best_length = float('inf')
		
		# History tracking
		history_best: List[float] = []
		history_iter_best: List[float] = []
		history_diversity: List[float] = []
		pheromone_snapshots: List[tuple[int, np.ndarray]] = []
		
		# Determine which iterations to capture pheromone snapshots
		snapshot_iterations = set([0, self.num_iterations // 4, self.num_iterations // 2, 
								   3 * self.num_iterations // 4, self.num_iterations - 1])
		
		# Capture initial pheromone state
		pheromone_snapshots.append((0, self.pheromone.copy()))
		
		# Initial visualization
		if viz_callback is not None:
			# Create a random initial solution to show
			initial_route = list(range(self.n))
			self.rng.shuffle(initial_route)
			initial_length = route_length(initial_route, self.dist)
			viz_callback(0, initial_route, initial_length)
		
		for iteration in range(self.num_iterations):
			# Construct solutions for all ants
			solutions = []
			
			for ant_id in range(self.num_ants):
				# Each ant starts from a random city
				start_city = self.rng.randint(0, self.n - 1)
				route = self._construct_solution(start_city)
				length = route_length(route, self.dist)
				solutions.append((route, length))
				
				# Update best solution
				if length < best_length:
					best_length = length
					best_route = route.copy()
			
			# Update pheromones
			self._update_pheromones(solutions, best_route, best_length)
			
			# Track history
			iteration_best_length = min(length for _, length in solutions)
			diversity = self._calculate_diversity(solutions)
			
			history_best.append(best_length)
			history_iter_best.append(iteration_best_length)
			history_diversity.append(diversity)
			
			# Capture pheromone snapshots at key iterations
			if iteration in snapshot_iterations:
				pheromone_snapshots.append((iteration + 1, self.pheromone.copy()))
			
			# Callbacks
			if progress_callback is not None:
				progress_callback(iteration + 1, best_length, iteration_best_length)
			
			if viz_callback is not None and (iteration + 1) % viz_interval == 0:
				viz_callback(iteration + 1, best_route, best_length)
		
		# Final visualization
		if viz_callback is not None:
			viz_callback(self.num_iterations, best_route, best_length)
		
		return {
			"best_route": best_route if best_route is not None else list(range(self.n)),
			"best_distance": best_length,
			"iterations": self.num_iterations,
			"history": ACOHistory(
				best_distances=history_best,
				iteration_best_distances=history_iter_best,
				diversity=history_diversity,
				pheromone_snapshots=pheromone_snapshots,
			),
		}
