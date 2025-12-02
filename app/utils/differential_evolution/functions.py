from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from shapely.affinity import rotate, scale, translate
from shapely.geometry import Polygon, box


@dataclass
class ShapeEvaluation:
	cost: float
	area: float
	placement_penalty: float
	rotation_penalty: float
	noncircularity: float
	feasible: bool
	max_rotation_fraction: float


def construct_corridor(
	corridor_width: float,
	horizontal_length: float,
	vertical_length: float,
	clearance: float = 0.0,
) -> Polygon:
	"""Returns an L-shaped corridor polygon."""

	horizontal_leg = box(-clearance, -clearance, horizontal_length, corridor_width)
	vertical_leg = box(horizontal_length - corridor_width, -clearance, horizontal_length, vertical_length)
	return horizontal_leg.union(vertical_leg)


def hammersley_sofa(disk_radius: float = 0.98, number_points: int = 40) -> Polygon:
	"""Constructs the classic Hammersley sofa polygon (for reference/benchmarking)."""

	rectangle_width = 4 / np.pi
	removed_disk_radius = 2 / np.pi

	theta_left = np.linspace(np.pi / 2, np.pi, number_points)
	theta_right = np.linspace(0, np.pi / 2, number_points)
	theta_bottom = np.linspace(np.pi, 0, number_points)

	x_left = disk_radius * np.cos(theta_left)
	y_left = disk_radius * np.sin(theta_left)

	x_bottom_left = np.linspace(-disk_radius, 0, number_points)
	y_bottom_left = np.zeros(number_points)

	x_bottom_bow = -removed_disk_radius + removed_disk_radius * np.cos(theta_bottom)
	y_bottom_bow = removed_disk_radius * np.sin(theta_bottom)

	x_bottom_right = np.linspace(2 * removed_disk_radius, disk_radius + 2 * removed_disk_radius, number_points)
	y_bottom_right = np.zeros(number_points)

	x_offset = disk_radius + 2 * removed_disk_radius - 1
	x_right = x_offset + disk_radius * np.cos(theta_right)
	y_right = disk_radius * np.sin(theta_right)

	coords: List[Tuple[float, float]] = []
	coords += list(zip(x_left, y_left))
	coords += list(zip(x_bottom_left, y_bottom_left))
	coords += list(zip(x_bottom_bow + rectangle_width, y_bottom_bow))
	coords += list(zip(x_bottom_right, y_bottom_right))
	coords += list(zip(x_right, y_right))

	sofa_polygon = Polygon(coords)
	sofa_polygon = rotate(sofa_polygon, 180, use_radians=False)
	sofa_polygon = translate(sofa_polygon, xoff=+0.98, yoff=0)
	return sofa_polygon


def radial_vector_to_polygon(lengths: Sequence[float]) -> Polygon:
	"""Creates a polygon from radial distances sampled around the origin."""

	lengths = np.asarray(lengths, dtype=float)
	angles = np.linspace(0, 2 * np.pi, len(lengths), endpoint=False)
	x = lengths * np.cos(angles)
	y = lengths * np.sin(angles)
	polygon = Polygon(np.column_stack([x, y]))
	if not polygon.is_valid:
		polygon = polygon.buffer(0)
	return polygon


def _fit_shape_into_corridor(
	shape: Polygon,
	corridor_width: float,
	x_limit: float,
	x_margin: float,
	y_margin: float,
) -> Polygon:
	minx, miny, maxx, maxy = shape.bounds
	height = max(maxy - miny, 1e-3)
	width = max(maxx - minx, 1e-3)
	available_height = max(corridor_width - 2 * y_margin, 1e-3)
	available_width = max(x_limit - 2 * x_margin, 1e-3)
	y_scale = min(available_height / height, 1.0)
	x_scale = min(available_width / width, 1.0)
	scale_factor = min(y_scale, x_scale, 1.0)
	shape = scale(shape, xfact=scale_factor, yfact=scale_factor, origin=(0, 0))
	minx, miny, _, _ = shape.bounds
	shape = translate(shape, xoff=-minx + x_margin, yoff=-miny + y_margin)

	attempts = 0
	while not shape.is_valid and attempts < 4:
		shape = shape.buffer(0)
		attempts += 1
	return shape


def vector_to_shape(
	vector: Sequence[float],
	corridor_width: float,
	x_limit: float,
	x_margin: float,
	y_margin: float,
) -> Polygon:
	"""Converts the vector into a polygon aligned with the corridor entry."""

	polygon = radial_vector_to_polygon(vector)
	polygon = _fit_shape_into_corridor(polygon, corridor_width, x_limit, x_margin, y_margin)
	return polygon


def _ellipse_profile(angles: np.ndarray, rng: np.random.Generator) -> np.ndarray:
	a = rng.uniform(0.4, 1.4)
	b = rng.uniform(0.2, a)
	shift = rng.uniform(0, 2 * np.pi)
	numerator = a * b
	denom = np.sqrt((b * np.cos(angles - shift)) ** 2 + (a * np.sin(angles - shift)) ** 2)
	return numerator / np.clip(denom, 1e-6, None)


def _rectangle_profile(angles: np.ndarray, rng: np.random.Generator) -> np.ndarray:
	half_w = rng.uniform(0.3, 1.0)
	half_h = rng.uniform(0.2, 0.8)
	denom = np.maximum(np.abs(np.cos(angles)) / half_w, np.abs(np.sin(angles)) / half_h)
	denom = np.clip(denom, 1e-3, None)
	return 1.0 / denom


def _barbell_profile(angles: np.ndarray, rng: np.random.Generator) -> np.ndarray:
	base = rng.uniform(0.3, 0.8)
	bumps = 0.3 * np.sin(2 * angles) + 0.15 * np.sin(5 * angles + rng.uniform(0, 2 * np.pi))
	return np.clip(base + bumps, 0.15, None)


def smooth_profile(values: np.ndarray, window: int = 5) -> np.ndarray:
	"""Applies a circular moving average to soften sharp radial changes."""

	if window < 2:
		return values
	if window % 2 == 0:
		window += 1

	pad = window // 2
	extended = np.concatenate([values[-pad:], values, values[:pad]])
	kernel = np.ones(window) / window
	smoothed = np.convolve(extended, kernel, mode="valid")
	return smoothed[: len(values)]


def generate_candidate(
	rng: np.random.Generator,
	radial_points: int,
	bounds: dict,
	strategy: str,
	smoothing_window: int,
) -> np.ndarray:
	angles = np.linspace(0, 2 * np.pi, radial_points, endpoint=False)

	if strategy == "ellipse":
		profile = _ellipse_profile(angles, rng)
	elif strategy == "rectangle":
		profile = _rectangle_profile(angles, rng)
	elif strategy == "barbell":
		profile = _barbell_profile(angles, rng)
	else:
		profile = rng.uniform(0.15, 1.2, radial_points)

	noise = rng.normal(0, 0.05, radial_points)
	candidate = profile + noise
	candidate = smooth_profile(candidate, window=smoothing_window)
	return np.clip(candidate, bounds["min_radius"], bounds["max_radius"])


def generate_initial_population(
	rng: np.random.Generator,
	size: int,
	radial_points: int,
	bounds: dict,
	strategies: Iterable[str],
	smoothing_window: int,
) -> List[np.ndarray]:
	population = []
	strategies = list(strategies)
	for idx in range(size):
		strategy = strategies[idx % len(strategies)]
		population.append(
			generate_candidate(
				rng,
				radial_points,
				bounds,
				strategy,
				smoothing_window=smoothing_window,
			)
		)
	return population


def move_and_rotate_smooth(
	corridor: Polygon,
	polygon: Polygon,
	step_size: float = 0.05,
	rotation_increment: float = 1.0,
) -> Tuple[bool, float, List[Polygon]]:
	path: List[Polygon] = []
	current = polygon
	total_rotation = 0.0
	finished_rotation = False

	if not corridor.covers(current):
		return False, 0.0, path

	while True:
		if not finished_rotation:
			moved = translate(current, xoff=step_size, yoff=0.0)
			if moved.within(corridor):
				current = moved
				path.append(current)
			else:
				pivot = (current.bounds[2], current.bounds[3])
				rotated = rotate(current, rotation_increment, origin=pivot, use_radians=False)
				total_rotation += rotation_increment

				_, miny, maxx, maxy = rotated.bounds
				corr_minx, corr_miny, corr_maxx, corr_maxy = corridor.bounds
				shift_x = 0.0
				shift_y = 0.0
				if maxx > corr_maxx:
					shift_x = corr_maxx - maxx
				if miny < corr_miny:
					shift_y = corr_miny - miny
				if maxy > corr_maxy:
					shift_y = corr_maxy - maxy
				rotated = translate(rotated, xoff=shift_x, yoff=shift_y)

				if not rotated.within(corridor):
					return False, min(total_rotation / 90.0, 1.0), path

				current = rotated
				path.append(current)
				if total_rotation >= 90.0:
					finished_rotation = True
		else:
			moved = translate(current, xoff=0.0, yoff=step_size)
			if moved.within(corridor):
				current = moved
				path.append(current)
			else:
				break

	return True, min(total_rotation / 90.0, 1.0), path


def check_feasibility(
	corridor: Polygon,
	polygon: Polygon,
	rotation_increment: float = 3.0,
) -> Tuple[bool, float]:
	_, _, corr_maxx, _ = corridor.bounds
	_, _, maxx_p, _ = polygon.bounds
	shift_x = corr_maxx - maxx_p
	current = translate(polygon, xoff=shift_x)

	if not current.within(corridor):
		return False, 0.0

	total_rotation = 0.0
	while total_rotation < 90.0:
		moved = translate(current, xoff=0.2, yoff=0.0)
		if moved.within(corridor):
			current = moved
			continue

		pivot = (moved.bounds[2], moved.bounds[3])
		rotated = rotate(moved, rotation_increment, origin=pivot, use_radians=False)
		total_rotation += rotation_increment

		_, miny, maxx, maxy = rotated.bounds
		corr_minx, corr_miny, corr_maxx, corr_maxy = corridor.bounds
		shift_x = 0.0
		shift_y = 0.0
		if maxx > corr_maxx:
			shift_x = corr_maxx - maxx
		if miny < corr_miny:
			shift_y = corr_miny - miny
		if maxy > corr_maxy:
			shift_y = corr_maxy - maxy
		rotated = translate(rotated, xoff=shift_x, yoff=shift_y)

		if not rotated.within(corridor):
			return False, min(total_rotation / 90.0, 1.0)

		current = rotated

	return True, 1.0


def objective_function(
	corridor: Polygon,
	shape: Polygon,
	weights: Tuple[float, float, float],
	rotation_increment: float,
) -> ShapeEvaluation:
	area = shape.area

	if corridor.covers(shape):
		placement_penalty = 0.0
	else:
		outside = shape.difference(corridor)
		placement_penalty = outside.area if not outside.is_empty else area

	feasible, rotation_fraction = check_feasibility(corridor, shape, rotation_increment=rotation_increment)
	rotation_penalty = 0.0 if feasible else (1.0 - rotation_fraction)

	coords = np.array(shape.exterior.coords)
	centroid = np.array([shape.centroid.x, shape.centroid.y])
	radial_distances = np.linalg.norm(coords - centroid, axis=1)
	noncircularity = float(np.std(radial_distances))

	cost = (
		weights[0] * rotation_penalty
		+ weights[1] * placement_penalty
		- weights[2] * (area + noncircularity)
	)

	return ShapeEvaluation(
		cost=cost,
		area=area,
		placement_penalty=placement_penalty,
		rotation_penalty=rotation_penalty,
		noncircularity=noncircularity,
		feasible=feasible,
		max_rotation_fraction=rotation_fraction,
	)


def compute_path_preview(
	corridor: Polygon,
	shape: Polygon,
	step_size: float,
	rotation_increment: float,
	max_frames: int = 100,
) -> Tuple[bool, float, List[Polygon]]:
	feasible, fraction, path = move_and_rotate_smooth(
		corridor,
		shape,
		step_size=step_size,
		rotation_increment=rotation_increment,
	)
	if len(path) > max_frames:
		stride = max(1, len(path) // max_frames)
		path = path[::stride]
	return feasible, fraction, path

