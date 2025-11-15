# Default algorithm parameters
DEFAULTS = {
	"num_ants": 20,
	"num_iterations": 100,
	"alpha": 1.0,  # Pheromone importance
	"beta": 2.0,   # Heuristic importance (distance)
	"evaporation_rate": 0.5,
	"pheromone_constant": 100.0,
	"elitist_weight": 0.0,  # Weight for best solution pheromone boost (0 = no elitism)
	"initialization_strategy": "random",
}

# Parameter ranges for validation
RANGES = {
	"num_ants": (1, 200),
	"num_iterations": (10, 1000),
	"alpha": (0.0, 5.0),
	"beta": (0.0, 10.0),
	"evaporation_rate": (0.0, 1.0),
	"pheromone_constant": (1.0, 1000.0),
	"elitist_weight": (0.0, 10.0),
}

# Initialization strategies
INITIALIZATION_STRATEGIES = {
	"random": {
		"description": "Each ant starts from a random city"
	},
	"nearest_neighbor": {
		"description": "Initialize pheromone using nearest neighbor heuristic"
	},
}

# UI help texts
UI_HELP = {
	"num_ants": "Number of ants in the colony. More ants explore more solutions per iteration but increase computation time.",
	"num_iterations": "Number of iterations (cycles) to run. Each iteration, all ants construct solutions and pheromones are updated.",
	"alpha": "Pheromone influence weight ($\\alpha$). Higher values make ants follow pheromone trails more strongly.",
	"beta": "Heuristic influence weight ($\\beta$). Higher values make ants prefer shorter edges (greedy behavior).",
	"evaporation_rate": "Pheromone evaporation rate ($\\rho$). Controls how quickly old pheromone trails fade. Range: 0 (no evaporation) to 1 (complete evaporation).",
	"pheromone_constant": "Pheromone deposit constant (Q). Scales the amount of pheromone deposited by ants. Higher values strengthen good solutions.",
	"elitist_weight": "Elitist ant weight (e). Extra pheromone boost for the best solution found. 0 = no elitism, higher values = stronger reinforcement of best solution.",
	"initialization": "How to initialize pheromone trails at the start.",
}

# Visualization settings
VIZ_CONFIG = {
	"update_interval": 10,  # Visualize every N iterations
	"show_pheromones": False,  # Option to overlay pheromone levels
}
