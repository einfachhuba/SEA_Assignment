from __future__ import annotations

DEFAULTS = {
	"initial_temperature": 200.0,   # T0
	"minimum_temperature": 1e-3,    # stop when temperature falls below this
	"max_iterations": 8000,         # hard cap on iterations
	"iterations_per_temperature": 50,  # inner loop per temperature level
	"cooling_strategy": "Exponential (geometric)",
	"alpha": 0.995,                 # used by exponential cooling T <- alpha * T
	"beta": 1e-3,                   # used by inverse cooling
	"eta": 0.5,                     # used by linear cooling (T <- max(T-eta, Tmin))
	"neighbor_strategy": "Inversion",
	"initialization_strategy": "Nearest neighbor",
	"seed": 55,
}

# Cooling strategies shown in the lecture slides
COOLING_STRATEGIES = {
	"Exponential (geometric)": {
		"key": "exponential",
		"description": "T_{k+1} = alpha * T_k (smooth, widely used)",
	},
	"Linear": {
		"key": "linear",
		"description": "T_{k+1} = max(T_k - eta, Tmin) (unphysical but simple)",
	},
	"Inverse": {
		"key": "inverse",
		"description": "T_{k+1} = T_k / (1 + beta * T_k) (can cool faster)",
	},
	"Logarithmic": {
		"key": "logarithmic",
		"description": "T_{k} = T0 / ln(k+1) (very slow theoretical schedule)",
	},
	"Inverse linear": {
		"key": "inverse_linear",
		"description": "T_k = T0 / k (harmonic-like cooling)",
	},
}

NEIGHBOR_STRATEGIES = {
	"Inversion": {
		"key": "inversion",
		"description": "Reverse a random segment (inversion mutation); strong for TSP-like problems",
	},
	"Swap": {
		"key": "swap",
		"description": "Swap two cities; simple and fast",
	},
	"Insertion": {
		"key": "insertion",
		"description": "Remove a city and insert at another position",
	},
}

INITIALIZATION_STRATEGIES = {
	"Random": {
		"key": "random",
		"description": "Random permutation of selected cities",
	},
	"Nearest neighbor": {
		"key": "nearest_neighbor",
		"description": "Greedy tour starting from a random city",
	},
}

UI_HELP = {
	"temperature": "Higher T accepts worse moves more often (exploration). Lower T prefers better moves (exploitation).",
	"alpha": "Geometric decay factor (0.9 - 0.999). Closer to 1 slows cooling and often improves quality.",
	"beta": "Inverse cooling shape parameter (~1e-3). Larger = steeper cooling.",
	"eta": "Linear cooling step size. Larger values cool faster.",
	"neighbor": "Defines how we perturb the current tour to generate candidates.",
}

