from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from .functions import (
	build_initial_route,
	propose_neighbor,
	route_length,
	rng_from_seed,
	is_valid_route,
)


@dataclass
class SAHistory:
	best_distances: List[float]
	temperatures: List[float]
	accepted_ratio: List[float]


class SimulatedAnnealingTSP:
	"""
	Simulated Annealing for a TSP-like problem over a set of cities.

	Contract:
	- Input: distance matrix (NxN), selected cities list (labels optional), and configuration
	- Output: dictionary with best route, best distance, iterations, final temperature, and history
	- Error modes: if N<2, raises ValueError
	"""

	def __init__(
		self,
		dist: np.ndarray,
		*,
		initial_temperature: float,
		minimum_temperature: float,
		max_iterations: int,
		iterations_per_temperature: int = 50,
		cooling_method: str = "exponential",  # keys: exponential, linear, inverse, logarithmic, inverse_linear
		alpha: float = 0.995,
		beta: float = 1e-3,
		eta: float = 0.5,
		neighbor_strategy: str = "two_opt",
		initialization_strategy: str = "nearest_neighbor",
		seed: Optional[int] = None,
	) -> None:
		if dist.shape[0] < 2:
			raise ValueError("Need at least two cities for SA")
		self.dist = dist
		self.n = dist.shape[0]
		self.T0 = float(initial_temperature)
		self.Tmin = float(minimum_temperature)
		self.max_iter = int(max_iterations)
		self.iters_per_T = int(iterations_per_temperature)
		self.cooling_method = cooling_method
		self.alpha = float(alpha)
		self.beta = float(beta)
		self.eta = float(eta)
		self.neighbor_strategy = neighbor_strategy
		self.initialization_strategy = initialization_strategy
		self.rng = rng_from_seed(seed)

	# -----------------
	# Cooling schedules
	# -----------------
	def _cool(self, T: float, k: int) -> float:
		if self.cooling_method == "exponential":
			return self.alpha * T
		elif self.cooling_method == "linear":
			return max(T - self.eta, self.Tmin)
		elif self.cooling_method == "inverse":
			return T / (1.0 + self.beta * T)
		elif self.cooling_method == "logarithmic":
			# avoid log(0)
			return self.T0 / max(1.0, math.log(k + 2))
		elif self.cooling_method == "inverse_linear":
			return self.T0 / max(1.0, k + 1)
		else:
			return self.alpha * T

	@staticmethod
	def _accept_prob(delta: float, T: float) -> float:
		if delta <= 0:
			return 1.0
		if T <= 0:
			return 0.0
		return math.exp(-delta / T)

	# ----
	# Run
	# ----
	# Run
	# ----
	def run(
		self, 
		progress_callback: Optional[Callable[[int, float, float], None]] = None,
		viz_callback: Optional[Callable[[int, List[int], float], None]] = None,
		viz_interval: int = 100
	) -> Dict:
		# Initialization
		current = build_initial_route(self.initialization_strategy, self.n, self.dist, None)
		current_len = route_length(current, self.dist)
		best = list(current)
		best_len = current_len

		T = self.T0
		history_best: List[float] = []
		history_T: List[float] = []
		history_acc: List[float] = []

		iteration = 0
		k = 0  # cooling step counter

		# Initial visualization
		if viz_callback is not None:
			viz_callback(iteration, best, best_len)

		while iteration < self.max_iter and T > self.Tmin:
			accepted = 0
			proposals = 0
			for _ in range(self.iters_per_T):
				proposals += 1
				candidate = propose_neighbor(current, self.neighbor_strategy, self.rng)
				# Safety: ensure permutation validity (no duplicates / missing)
				if not is_valid_route(candidate, self.n):
					# Skip invalid neighbor (shouldn't happen with our neighbor ops)
					continue
				cand_len = route_length(candidate, self.dist)
				delta = cand_len - current_len
				if self.rng.random() < self._accept_prob(delta, T):
					current, current_len = candidate, cand_len
					accepted += 1
					if current_len < best_len:
						best, best_len = list(current), float(current_len)
						# Visualize when best improves
						if viz_callback is not None:
							viz_callback(iteration, best, best_len)
				iteration += 1
				history_best.append(best_len)
				history_T.append(T)
				
				# Periodic visualization update
				if viz_callback is not None and iteration % viz_interval == 0:
					viz_callback(iteration, best, best_len)
				
				if iteration >= self.max_iter:
					break

			acc_ratio = accepted / max(1, proposals)
			history_acc.append(acc_ratio)
			k += 1
			T = self._cool(T, k)

			if progress_callback is not None:
				progress_callback(iteration, best_len, T)

		return {
			"best_route": best,
			"best_distance": best_len,
			"final_temperature": T,
			"iterations": iteration,
			"history": SAHistory(history_best, history_T, history_acc),
		}

