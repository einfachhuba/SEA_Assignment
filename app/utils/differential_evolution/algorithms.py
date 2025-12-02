from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from .config import RADIAL_STRATEGIES
from .functions import (
	ShapeEvaluation,
	generate_initial_population,
	objective_function,
	smooth_profile,
	vector_to_shape,
)


ProgressCallback = Callable[[int, float, float], None]


@dataclass
class ShapeSnapshot:
	generation: int
	cost: float
	area: float
	shape: object


@dataclass
class DifferentialEvolutionResult:
	best_vector: np.ndarray
	best_shape: object
	evaluation: ShapeEvaluation
	history: List[Dict[str, float]]
	snapshots: List[ShapeSnapshot]


class DifferentialEvolutionSolver:
	def __init__(
		self,
		corridor,
		bounds: Dict[str, float],
		weights: Dict[str, float],
		population_size: int,
		generations: int,
		mutation_factor: float,
		crossover_rate: float,
		radial_points: int,
		rotation_increment: float,
		step_size: float,
		seed: Optional[int] = None,
		strategies: Optional[List[str]] = None,
		smoothing_window: int = 5,
		max_snapshots: int = 6,
	) -> None:
		self.corridor = corridor
		self.bounds = bounds
		self.weights = (
			weights["rotation"],
			weights["placement"],
			weights["reward"],
		)
		self.population_size = population_size
		self.generations = generations
		self.mutation_factor = mutation_factor
		self.crossover_rate = crossover_rate
		self.radial_points = radial_points
		self.rotation_increment = rotation_increment
		self.step_size = step_size
		self.strategies = strategies or list(RADIAL_STRATEGIES)
		self.rng = np.random.default_rng(seed)
		self.smoothing_window = smoothing_window
		self.max_snapshots = max(1, max_snapshots)

	def _evaluate_vector(self, vector: np.ndarray) -> ShapeEvaluation:
		shape = vector_to_shape(
			vector,
			corridor_width=self.bounds["corridor_width"],
			x_limit=self.bounds["x_limit"],
			x_margin=self.bounds.get("x_margin", 0.0),
			y_margin=self.bounds.get("y_margin", 0.0),
		)
		return objective_function(
			self.corridor,
			shape,
			self.weights,
			rotation_increment=self.rotation_increment,
		)

	def run(self, on_progress: Optional[ProgressCallback] = None) -> DifferentialEvolutionResult:
		population = np.array(
			generate_initial_population(
				self.rng,
				size=self.population_size,
				radial_points=self.radial_points,
				bounds=self.bounds,
				strategies=self.strategies,
				smoothing_window=self.smoothing_window,
			)
		)

		evaluations: List[ShapeEvaluation] = []
		shapes = []
		for vec in population:
			shape = vector_to_shape(
				vec,
				corridor_width=self.bounds["corridor_width"],
				x_limit=self.bounds["x_limit"],
				x_margin=self.bounds.get("x_margin", 0.0),
				y_margin=self.bounds.get("y_margin", 0.0),
			)
			eval_result = objective_function(
				self.corridor,
				shape,
				self.weights,
				rotation_increment=self.rotation_increment,
			)
			evaluations.append(eval_result)
			shapes.append(shape)

		best_idx = int(np.argmin([e.cost for e in evaluations]))
		best_eval = evaluations[best_idx]
		best_vector = population[best_idx].copy()
		best_shape = shapes[best_idx]

		history: List[Dict[str, float]] = []
		snapshot_records: List[ShapeSnapshot] = [
			ShapeSnapshot(
				generation=0,
				cost=float(best_eval.cost),
				area=float(best_eval.area),
				shape=best_shape,
			)
		]

		for gen in range(self.generations):
			for i in range(self.population_size):
				trial = self._mutate(i, population)
				trial = self._crossover(population[i], trial)
				trial = smooth_profile(trial, window=self.smoothing_window)
				trial = np.clip(trial, self.bounds["min_radius"], self.bounds["max_radius"])
				trial_eval = self._evaluate_vector(trial)

				if trial_eval.cost <= evaluations[i].cost:
					population[i] = trial
					evaluations[i] = trial_eval
					shapes[i] = vector_to_shape(
						trial,
						corridor_width=self.bounds["corridor_width"],
						x_limit=self.bounds["x_limit"],
						x_margin=self.bounds.get("x_margin", 0.0),
						y_margin=self.bounds.get("y_margin", 0.0),
					)

					if trial_eval.cost < best_eval.cost:
						best_eval = trial_eval
						best_vector = trial.copy()
						best_shape = shapes[i]

			costs = [e.cost for e in evaluations]
			history.append(
				{
					"generation": gen + 1,
					"best_cost": float(best_eval.cost),
					"mean_cost": float(np.mean(costs)),
					"best_area": float(best_eval.area),
				}
			)

			if on_progress:
				on_progress(gen + 1, float(best_eval.cost), float(best_eval.area))

			snapshot_records.append(
				ShapeSnapshot(
					generation=gen + 1,
					cost=float(best_eval.cost),
					area=float(best_eval.area),
					shape=best_shape,
				)
			)

		return DifferentialEvolutionResult(
			best_vector=best_vector,
			best_shape=best_shape,
			evaluation=best_eval,
			history=history,
			snapshots=self._select_snapshots(snapshot_records),
		)

	def _mutate(self, idx: int, population: np.ndarray) -> np.ndarray:
		candidates = [i for i in range(self.population_size) if i != idx]
		a_idx, b_idx, c_idx = self.rng.choice(candidates, 3, replace=False)
		a = population[a_idx]
		b = population[b_idx]
		c = population[c_idx]
		mutant = a + self.mutation_factor * (b - c)
		return np.clip(mutant, self.bounds["min_radius"], self.bounds["max_radius"])

	def _crossover(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
		cross_points = self.rng.random(len(target)) < self.crossover_rate
		if not np.any(cross_points):
			idx = self.rng.integers(0, len(target))
			cross_points[idx] = True
		offspring = np.where(cross_points, mutant, target)
		return offspring

	def _select_snapshots(self, records: List[ShapeSnapshot]) -> List[ShapeSnapshot]:
		if len(records) <= self.max_snapshots:
			return records
		indices = np.linspace(0, len(records) - 1, self.max_snapshots, dtype=int)
		selected = []
		seen = set()
		for idx in indices:
			if idx in seen:
				continue
			seen.add(idx)
			selected.append(records[int(idx)])
		return selected

