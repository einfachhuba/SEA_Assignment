"""Configuration constants for the moving sofa differential evolution demo."""

from __future__ import annotations

DEFAULTS = {
	"population_size": 28,
	"generations": 80,
	"mutation_factor": 0.75,
	"crossover_rate": 0.85,
	"radial_points": 36,
	"rotation_increment": 3.0,
	"step_size": 0.05,
	"seed": 42,
	"smoothing_window": 5,
}

CORRIDOR_SETTINGS = {
	"corridor_width": 1.01,
	"horizontal_length": 6.25,
	"vertical_length": 6.25,
	"clearance": 0.01,
}

SHAPE_BOUNDS = {
	"min_radius": 0.12,
	"max_radius": 3.2,
	"x_limit": 4.5,
	"x_margin": 0.05,
	"y_margin": 0.05,
}

WEIGHT_DEFAULTS = {
	"rotation": 1.2,
	"placement": 1.0,
	"reward": 0.8,
}

UI_HELP = {
	"population": "Larger populations explore more shape variants but slow down the search.",
	"generations": "Number of evolutionary steps. Each generation evaluates the full population.",
	"mutation": "Controls how far new candidates stray from parents (Differential Weight F).",
	"crossover": "Probability of taking mutant genes over the incumbent vector.",
	"radial": "Number of radial spokes that describe the shape silhouette.",
	"rotation": "Smaller increments approximate the corridor turn more accurately but take longer.",
	"smoothing": "Moving-average window applied to radial vectors; larger values enforce smoother shapes.",
}

RADIAL_STRATEGIES = ("random", "ellipse", "rectangle", "barbell")

