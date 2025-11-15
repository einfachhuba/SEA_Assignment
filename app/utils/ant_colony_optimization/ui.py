from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import streamlit as st

from utils.general.ui import spacer

from .config import DEFAULTS, INITIALIZATION_STRATEGIES, UI_HELP
from .functions import (
	load_city_coordinates, 
	plot_convergence, 
	plot_diversity, 
	plot_route_map,
	plot_pheromone_heatmap,
	plot_pheromone_evolution,
)


def display_city_selection() -> Dict:
	"""Display city selection UI and return selected cities and coordinates."""
	st.subheader("Select cities")
	data_path = Path("app/data/ant_colony_optimization/routes_carinthia/city_coordinates.json")
	if not data_path.exists():
		st.error("City data not found at app/data/ant_colony_optimization/routes_carinthia/city_coordinates.json")
		st.stop()

	coords = load_city_coordinates(data_path)
	all_cities = sorted(coords.keys())
	
	# Default: select some well-known Carinthian cities
	default_cities = [
		"Klagenfurt am Wörthersee", "Villach", "Wolfsberg", 
		"St. Veit an der Glan", "Spittal an der Drau", "Feldkirchen in Kärnten",
		"Völkermarkt", "Hermagor-Pressegger See"
	]
	default_pick = [c for c in default_cities if c in all_cities]

	sel = st.multiselect(
		"Select cities to visit",
		options=all_cities,
		default=default_pick
	)
	
	use_road = st.checkbox(
		"Use road distances (if available)", 
		value=True, 
		help="Prefer real route distances from the dataset over straight-line distances."
	)
	
	if len(sel) < 2:
		st.info("Select at least 2 cities to get a meaningful tour.")
	
	return {"selected_cities": sel, "coords": coords, "use_road_distances": use_road}


def display_parameter_controls() -> Dict:
	"""Display parameter controls and return selected parameters."""
	st.subheader("Algorithm Configuration")
	
	st.markdown("""
	Configure the Ant Colony Optimization parameters below. The algorithm simulates ants
	that probabilistically construct solutions based on pheromone trails and distance heuristics.
	""")
	
	# Colony settings
	st.markdown("##### Colony Settings")
	col1, col2 = st.columns(2)
	with col1:
		num_ants = st.slider(
			"Number of ants",
			min_value=5,
			max_value=60,
			value=int(DEFAULTS["num_ants"]),
			help=UI_HELP["num_ants"]
		)
	with col2:
		num_iterations = st.slider(
			"Number of iterations",
			min_value=10,
			max_value=500,
			value=int(DEFAULTS["num_iterations"]),
			help=UI_HELP["num_iterations"]
		)
	
	spacer(2)
	
	# Algorithm parameters
	st.markdown("##### Algorithm Parameters")
	col1, col2, col3 = st.columns(3)
	with col1:
		alpha = st.slider(
			"$\\alpha$ - Pheromone weight",
			min_value=0.0,
			max_value=5.0,
			step=0.1,
			value=float(DEFAULTS["alpha"]),
			help=UI_HELP["alpha"]
		)
	with col2:
		beta = st.slider(
			"$\\beta$ - Heuristic weight",
			min_value=0.0,
			max_value=5.0,
			step=0.1,
			value=float(DEFAULTS["beta"]),
			help=UI_HELP["beta"]
		)
	with col3:
		evaporation_rate = st.slider(
			"Evaporation rate $\\rho$",
			min_value=0.0,
			max_value=1.0,
			step=0.05,
			value=float(DEFAULTS["evaporation_rate"]),
			help=UI_HELP["evaporation_rate"]
		)
	
	spacer(2)
	
	# Pheromone settings
	st.markdown("##### Pheromone Settings")
	col1, col2 = st.columns(2)
	with col1:
		pheromone_constant = st.number_input(
			"Pheromone constant (Q)",
			min_value=1.0,
			max_value=1000.0,
			value=float(DEFAULTS["pheromone_constant"]),
			help=UI_HELP["pheromone_constant"]
		)
	with col2:
		elitist_weight = st.slider(
			"Elitist weight (e)",
			min_value=0.0,
			max_value=10.0,
			step=0.5,
			value=float(DEFAULTS["elitist_weight"]),
			help=UI_HELP["elitist_weight"]
		)
	
	spacer(2)
	
	# Strategy settings
	st.markdown("##### Initialization Strategy")
	init_name = st.selectbox(
		"Initialization strategy",
		list(INITIALIZATION_STRATEGIES.keys()),
		index=list(INITIALIZATION_STRATEGIES.keys()).index(DEFAULTS["initialization_strategy"]),
		help=UI_HELP["initialization"]
	)
	st.caption(INITIALIZATION_STRATEGIES[init_name]["description"])
	
	return {
		"num_ants": num_ants,
		"num_iterations": num_iterations,
		"alpha": alpha,
		"beta": beta,
		"evaporation_rate": evaporation_rate,
		"pheromone_constant": pheromone_constant,
		"elitist_weight": elitist_weight,
		"initialization_strategy": init_name,
	}


def display_results(
	best_distance: float,
	cities: List[str],
	coords: Dict,
	route: List[int],
	history,
	use_road_paths: bool = True,
):
	"""Display algorithm results including route map and convergence plots."""
	st.header("Results")
	
	# Summary metrics
	col1, col2, col3 = st.columns(3)
	with col1:
		st.metric("Best Distance", f"{best_distance:.2f} km")
	with col2:
		st.metric("Number of Cities", len(cities))
	with col3:
		improvement = ((history.iteration_best_distances[0] - best_distance) / history.iteration_best_distances[0] * 100) if history.iteration_best_distances[0] > 0 else 0
		st.metric("Improvement", f"{improvement:.1f}%")
	
	spacer(4)
	
	# Best route visualization
	st.subheader("Best Route Found")
	ordered_cities = [cities[i] for i in route]
	st.write(f"**Route:** {' -> '.join(ordered_cities)} -> {ordered_cities[0]}")
	
	fig_map = plot_route_map(cities, coords, route, use_road_paths=use_road_paths)
	st.plotly_chart(fig_map, use_container_width=True)
	
	spacer(4)
	
	# Convergence plots
	st.subheader("Convergence Analysis")
	
	col1, col2 = st.columns(2)
	
	with col1:
		st.markdown("**Best Distance Over Iterations**")
		fig_conv = plot_convergence(history.best_distances)
		st.plotly_chart(fig_conv, use_container_width=True)
		st.caption("Shows how the best solution found improves over iterations.")
	
	with col2:
		st.markdown("**Solution Diversity**")
		fig_div = plot_diversity(history.diversity)
		st.plotly_chart(fig_div, use_container_width=True)
		st.caption("Standard deviation of solution distances per iteration. Higher diversity means ants are exploring different solutions.")
	
	spacer(8)
	
	# Pheromone visualization
	st.subheader("Pheromone Trail Analysis")
	st.markdown("""
	Visualize how pheromone trails evolve during the optimization process. 
	Stronger pheromones (brighter colors) indicate paths that ants found more successful.
	""")
	
	if history.pheromone_snapshots and len(history.pheromone_snapshots) > 0:
		# Pheromone evolution over time
		st.markdown("**Pheromone Evolution on Best Route**")
		fig_phero_evo = plot_pheromone_evolution(cities, history.pheromone_snapshots, route)
		st.plotly_chart(fig_phero_evo, use_container_width=True)
		st.caption("Shows how pheromone levels increase on the edges of the best route as the algorithm converges.")
		
		spacer(4)
		
		# Interactive pheromone heatmap selector
		st.markdown("**Pheromone Intensity Heatmap**")
		
		# Create iteration selector
		iterations_available = [iter_num for iter_num, _ in history.pheromone_snapshots]
		selected_iteration = st.select_slider(
			"Select iteration to view pheromone matrix",
			options=iterations_available,
			value=iterations_available[-1] if iterations_available else 0,
			help="Drag to see how pheromone trails evolve from initial uniform distribution to concentrated trails"
		)
		
		# Find the corresponding pheromone matrix
		pheromone_matrix = None
		for iter_num, matrix in history.pheromone_snapshots:
			if iter_num == selected_iteration:
				pheromone_matrix = matrix
				break
		
		if pheromone_matrix is not None:
			fig_heatmap = plot_pheromone_heatmap(cities, pheromone_matrix, selected_iteration, route)
			st.plotly_chart(fig_heatmap, use_container_width=True)
			st.caption("Red boxes highlight edges in the best route. Brighter colors = stronger pheromone trails. Notice how pheromone concentrates on the optimal path over time.")
		else:
			st.warning("Pheromone matrix not available for selected iteration.")
	else:
		st.info("Pheromone snapshots not available. This data is collected during algorithm execution.")
