from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st

from utils.general.ui import spacer

from .config import (
	DEFAULTS,
	COOLING_STRATEGIES,
	INITIALIZATION_STRATEGIES,
	NEIGHBOR_STRATEGIES,
	UI_HELP,
)
from .functions import (
	load_city_coordinates,
	plot_convergence,
	plot_route_map,
)

def display_city_selection() -> Dict:
	st.subheader("Select cities")
	data_path = Path("app/data/simulated_annealing/routes/city_coordinates.json")
	if not data_path.exists():
		st.error("City data not found at app/data/simulated_annealing/routes/city_coordinates.json")
		st.stop()

	coords = load_city_coordinates(data_path)
	all_cities = sorted(coords.keys())
	default_pick = [c for c in ["Wien", "Graz", "Linz", "Salzburg", "Innsbruck", "Klagenfurt", "Villach", "Wels", "Sankt Pölten", "Bregenz"] if c in all_cities]

	sel = st.multiselect(
		"Select cities to visit",
		options=all_cities,
		default=default_pick
	)
	use_road = st.checkbox("Use road distances (if available)", value=True, help="Prefer real route distances from the dataset over straight-line distances.")
	if len(sel) < 6:
		st.info("Select at least 6 cities to get a meaningful tour.")
	return {"selected_cities": sel, "coords": coords, "use_road_distances": use_road}


def display_parameter_controls() -> Dict:
	st.subheader("Algorithm Configuration")
	
	st.markdown("""
	Configure the simulated annealing parameters below. The algorithm explores the search space by accepting 
	worse solutions with a probability that decreases as the temperature cools down.
	""")
	
	# Temperature settings
	st.markdown("##### Temperature Settings")
	col1, col2 = st.columns(2)
	with col1:
		initial_temperature = st.number_input(
			"Initial temperature (T_0)", 
			min_value=1.0, 
			max_value=10000.0, 
			value=float(DEFAULTS["initial_temperature"]), 
			help=UI_HELP["temperature"]
		)
	with col2:
		minimum_temperature = st.number_input(
			"Minimum temperature (T_min)", 
			min_value=1e-6, 
			max_value=10.0, 
			value=float(DEFAULTS["minimum_temperature"]),
			format="%.6f",
			help="Algorithm stops when temperature falls below this threshold."
		)
	
	spacer(2)
	
	# Iteration settings
	st.markdown("##### Iteration Settings")
	col1, col2 = st.columns(2)
	with col1:
		max_iterations = st.slider(
			"Maximum iterations", 
			min_value=500, 
			max_value=50000, 
			value=int(DEFAULTS["max_iterations"]),
			help="Hard cap on the total number of iterations regardless of temperature."
		)
	with col2:
		iterations_per_temperature = st.slider(
			"Iterations per temperature level", 
			min_value=10, 
			max_value=500, 
			value=int(DEFAULTS["iterations_per_temperature"]),
			help="Number of candidate solutions to try at each temperature before cooling."
		)

	spacer(2)

	# Strategy settings
	st.markdown("##### Strategy Selection")
	col1, col2 = st.columns(2)
	with col1:
		cooling_name = st.selectbox(
			"Cooling strategy", 
			list(COOLING_STRATEGIES.keys()), 
			index=list(COOLING_STRATEGIES.keys()).index(DEFAULTS["cooling_strategy"]),
			help="How the temperature decreases over time - affects exploration vs exploitation balance."
		)
		st.caption(COOLING_STRATEGIES[cooling_name]["description"])
		
		neighbor_name = st.selectbox(
			"Neighbor generation strategy", 
			list(NEIGHBOR_STRATEGIES.keys()), 
			index=list(NEIGHBOR_STRATEGIES.keys()).index(DEFAULTS["neighbor_strategy"]),
			help=UI_HELP["neighbor"]
		)
		st.caption(NEIGHBOR_STRATEGIES[neighbor_name]["description"])
	
	with col2:
		init_name = st.selectbox(
			"Initialization strategy", 
			list(INITIALIZATION_STRATEGIES.keys()), 
			index=list(INITIALIZATION_STRATEGIES.keys()).index(DEFAULTS["initialization_strategy"]),
			help="How to generate the starting tour."
		)
		st.caption(INITIALIZATION_STRATEGIES[init_name]["description"])
	
	spacer(2)
	
	# Cooling parameters
	st.markdown("##### Cooling Parameters")
	st.caption("These parameters fine-tune the selected cooling strategy.")
	col1, col2, col3 = st.columns(3)
	with col1:
		alpha = st.slider(
			"Alpha - Exponential", 
			min_value=0.90, 
			max_value=0.999, 
			step=0.001, 
			value=float(DEFAULTS["alpha"]), 
			help=UI_HELP["alpha"]
		)
	with col2:
		beta = st.number_input(
			"Beta - Inverse", 
			min_value=1e-6, 
			max_value=1e-1, 
			value=float(DEFAULTS["beta"]), 
			format="%.6f",
			help=UI_HELP["beta"]
		)
	with col3:
		eta = st.number_input(
			"Eta - Linear", 
			min_value=0.001, 
			max_value=5.0, 
			value=float(DEFAULTS["eta"]), 
			help=UI_HELP["eta"]
		)

	return {
		"initial_temperature": initial_temperature,
		"minimum_temperature": minimum_temperature,
		"max_iterations": max_iterations,
		"iterations_per_temperature": iterations_per_temperature,
		"cooling_strategy": COOLING_STRATEGIES[cooling_name]["key"],
		"neighbor_strategy": NEIGHBOR_STRATEGIES[neighbor_name]["key"],
		"initialization_strategy": INITIALIZATION_STRATEGIES[init_name]["key"],
		"alpha": alpha,
		"beta": beta,
		"eta": eta,
	}


def display_results(
	best_distance: float,
	cities: List[str],
	coords: Dict[str, Tuple[float, float]],
	route: List[int],
	history,
	use_road_paths: bool = True,
) -> None:
	st.subheader("Results")
	# First row: plots side-by-side
	col1, col2 = st.columns([1, 1])
	with col1:
		fig_map = plot_route_map(cities, coords, route, use_road_paths=use_road_paths)
		st.plotly_chart(fig_map, use_container_width=True)
	with col2:
		fig_conv = plot_convergence(history.best_distances, history.temperatures)
		st.plotly_chart(fig_conv, use_container_width=True)

	spacer(1)

	# Second row: left = route list, right = best distance metric
	col1, col2 = st.columns([1, 1])
	with col1:
		st.markdown("##### Best route order")
		if route:
			ordered = [cities[i] for i in route]
			ordered_cycle = ordered + [ordered[0]]  # close the loop
			with st.expander("Shown as numbered list"):
				st.markdown("\n".join([f"{idx+1}. {name}" for idx, name in enumerate(ordered_cycle)]))
		else:
			st.info("No route available to display.")
	with col2:
		st.metric("Best route distance", f"{best_distance:.1f} km")

