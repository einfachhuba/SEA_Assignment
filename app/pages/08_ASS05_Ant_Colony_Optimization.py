import streamlit as st
from typing import List
from utils.general.pdf_viewer import display_pdf_with_controls
from utils.general.ui import spacer
from utils.ant_colony_optimization.ui import (
	display_city_selection,
	display_parameter_controls,
	display_results,
)
from utils.ant_colony_optimization.algorithms import AntColonyOptimizationTSP
from utils.ant_colony_optimization.functions import build_distance_matrix, build_distance_matrix_with_routes


st.set_page_config(page_title="AT05: Ant Colony Optimization")

tab1, tab2 = st.tabs(["📄 Assignment Paper", "🔧 Ant Colony Optimization Implementation"])

with tab1:
	st.title("Assignment 05: Ant Colony Optimization")
	pdf_path = "Assignment_Sheets/05/SEA_Exercise05_Ant_Colony_Optimization.pdf"
	try:
		display_pdf_with_controls(pdf_path)
	except Exception:
		st.info("Assignment PDF not found. Proceed to the implementation tab.")

with tab2:
	st.title("Assignment 05: Ant Colony Optimization Implementation")
	
	st.header("Introduction")
	st.markdown("""
	**Ant Colony Optimization (ACO)** is a nature-inspired metaheuristic based on the foraging behavior 
	of ants. Real ants deposit pheromone trails while searching for food, and other ants are more likely 
	to follow stronger pheromone paths. Over time, shorter paths accumulate more pheromone, which leads the 
	colony to converge on optimal routes. We apply ACO to solve the **Traveling Salesperson Problem (TSP)** - finding the shortest route 
	that visits all cities in Carinthia exactly once and returns to the starting point.
	""")

	spacer(6)

	cont = st.container(border=True)
	with cont:
		col1, col2 = st.columns(2)
		with col1:
			st.badge("**Key Features:**", color="blue")
			st.markdown("""
			- **Probabilistic construction**: Each ant builds a tour based on pheromone and distance
			- **Pheromone trails**: Shorter routes receive more pheromone
			- **Collective intelligence**: Colony converges on good solutions through strong pheromone trails
			- **Exploration vs exploitation**: Balanced by $\\alpha$ (pheromone) and $\\beta$ (heuristic) parameters
			- **Elitist variant**: Optional extra reinforcement for the best solution found
			""")

		with col2:
			st.badge("**Problem Characteristics:**", color="blue")
			st.markdown("""
			- **Search space**: Permutations of city sequences - *O((n-1)!/2)* possible tours
			- **Dataset**: Carinthian cities with real road distances from OpenRouteService
			- **Distance metric**: Road distances (km) with fallback to Haversine
			- **Visualization**: File-name matching of routes to cities on a geographic map
			""")

	spacer(24)

	st.header("Methods")
	
	st.markdown("""
	### Ant Colony Optimization Algorithm
	
	The ACO algorithm for TSP follows these steps:
	
	1. **Initialization**: 
	   - Initialize pheromone matrix $\\tau$ (all edges start with small equal pheromone)
	   - Calculate heuristic information $\\eta$ = 1/distance (visibility) because shorter edges are preferred
	
	2. **Solution Construction** (for each ant):
	   - Start from a random city (initial for all ants)
	   - While unvisited cities remain:
	     - Choose next city probabilistically: $P_{ij} = \\frac{\\tau_{ij}^\\alpha \\cdot \\eta_{ij}^\\beta}{\\sum_{k \\in \\text{unvisited}} \\tau_{ik}^\\alpha \\cdot \\eta_{ik}^\\beta}$
	     - Move to chosen city and mark as visited
	   - Return to starting city to complete tour
	
	3. **Pheromone Update**:
	   - **Evaporation**: $\\tau_{ij} \\leftarrow (1-\\rho) \\cdot \\tau_{ij}$ for all edges ($\\rho$ = evaporation rate)
	   - **Deposit**: Each ant k deposits pheromone: $\\Delta \\tau_{ij}^k = Q / L_k$ on edges along its tour
	   - **Elitist boost**: Best solution gets extra pheromone: $\\tau_{ij} \\leftarrow \\tau_{ij} + e \\cdot Q / L_{\\text{best}}$
	
	4. **Termination**: Repeat steps 2-3 until max iterations reached.
	
	### Key Parameters
	
	- **Pheromone importance $\\alpha$**: higher values make ants follow pheromone trails more strongly
	- **Heuristic importance $\\beta$**: higher values make ants prefer shorter edges (greedy behavior)
	- **Evaporation rate $\\rho$**: controls how quickly old pheromone trails fade (0 = no evaporation, 1 = complete)
	- **Pheromone constant Q**: scales the amount of pheromone deposited
	- **Elitist weight e**: extra pheromone reinforcement for the best solution (0 = no elitism)
	
	### Balance of Exploration and Exploitation
	
	- **Exploration**: Random probabilistic choices allow discovery of new solutions
	- **Exploitation**: Strong pheromone trails guide ants toward known good solutions
	- The $\\alpha$/$\\beta$ ratio controls this balance: high $\\beta$ -> greedy (exploit), high $\\alpha$ -> follow pheromones (exploit past successes)

	""")

	spacer(24)

	st.header("Configuration & Execution")
	
	st.markdown("""
	Select cities from Carinthia, configure ACO parameters, and run the algorithm below. 
	Results will show the best tour found, visualized on a map with actual road routes and 
	convergence and diversity plots.
	""")

	spacer(12)

	# City and data selection
	city_data = display_city_selection()
	selected_cities = city_data["selected_cities"]
	coords = city_data["coords"]
	use_road_distances = city_data.get("use_road_distances", True)

	spacer(8)

	# Parameters
	params = display_parameter_controls()

	spacer(12)

	if st.button("Run Ant Colony Optimization", type="primary", use_container_width=True, disabled=len(selected_cities) < 2):
		# Generate unique run ID to avoid key conflicts across multiple runs
		import time
		run_id = int(time.time() * 1000)
		
		# Build distance matrix
		if use_road_distances:
			dist = build_distance_matrix_with_routes(selected_cities, coords)
		else:
			dist = build_distance_matrix(selected_cities, coords)

		# Initialize solver
		aco = AntColonyOptimizationTSP(
			dist,
			num_ants=params["num_ants"],
			num_iterations=params["num_iterations"],
			alpha=params["alpha"],
			beta=params["beta"],
			evaporation_rate=params["evaporation_rate"],
			pheromone_constant=params["pheromone_constant"],
			elitist_weight=params["elitist_weight"],
			initialization_strategy=params["initialization_strategy"],
		)

		# Create visualization placeholders
		progress_bar = st.progress(0.0)
		status_text = st.empty()
		viz_placeholder = st.empty()
		
		# Import plot function here
		from utils.ant_colony_optimization.functions import plot_route_map

		def on_progress(iteration: int, best_len: float, iter_best_len: float):
			progress_bar.progress(min(1.0, iteration / params["num_iterations"]))
			status_text.text(f"iter={iteration}/{params['num_iterations']} | best={best_len:.1f} km | iter_best={iter_best_len:.1f} km")

		# Counter to ensure each visualization has a unique key
		viz_counter = [0]
		
		def on_viz_update(iteration: int, route: List[int], distance: float):
			viz_counter[0] += 1
			with viz_placeholder.container():
				st.caption(f"Current best route at iteration {iteration}: {distance:.1f} km")
				fig = plot_route_map(selected_cities, coords, route, use_road_paths=use_road_distances)
				st.plotly_chart(fig, use_container_width=True, key=f"route_viz_{run_id}_{viz_counter[0]}")

		with st.spinner("Running ant colony optimization..."):
			results = aco.run(
				progress_callback=on_progress,
				viz_callback=on_viz_update,
				viz_interval=max(5, params["num_iterations"] // 20)  # Update ~20 times during run
			)

		progress_bar.empty()
		status_text.empty()
		viz_placeholder.empty()

		spacer(12)
		
		# Store results in session state
		st.session_state['aco_results'] = results
		st.session_state['aco_cities'] = selected_cities
		st.session_state['aco_coords'] = coords
		st.session_state['aco_params'] = params
		st.session_state['aco_use_road_distances'] = use_road_distances
	
	spacer(24)
	
	# Display results if they exist
	if 'aco_results' in st.session_state:
		results = st.session_state['aco_results']
		selected_cities = st.session_state['aco_cities']
		coords = st.session_state['aco_coords']
		params = st.session_state.get('aco_params', {})
		use_road_distances = st.session_state.get('aco_use_road_distances', True)
		
		display_results(
			best_distance=results["best_distance"],
			cities=selected_cities,
			coords=coords,
			route=results["best_route"],
			history=results["history"],
			use_road_paths=use_road_distances,
		)
		
		spacer(24)
		
		st.header("Discussion")
		
		st.markdown("""
		### Algorithm Behavior and Performance
		
		**Pheromone Dynamics:**
		- Early iterations: Pheromone trails are relatively uniform, leading to diverse exploration
		- Middle iterations: Stronger trails emerge on good paths, balancing exploration and exploitation
		- Late iterations: Pheromone concentrates on the best routes, refining the solution
		- Evaporation prevents premature convergence by gradually forgetting weak early choices
		
		**Parameter Influence:**
		- **High $\\alpha$ (pheromone weight)**: Ants follow pheromone trails strongly -> faster convergence but risk of premature convergence
		- **High $\\beta$ (heuristic weight)**: Ants prefer short edges greedily -> more exploitation, good for quick solutions
		- **High $\\rho$ (evaporation)**: Faster forgetting -> more exploration but slower convergence
		- **Elitist weight > 0**: Accelerates convergence by reinforcing the best solution found
		
		**Pheromone Trail Dynamics:**
		- The pheromone heatmap visualization shows how trails concentrate on good paths over time
		- Early iterations: Relatively uniform pheromone distribution (exploration phase)
		- Middle iterations: Pheromone begins concentrating on promising edges
		- Final iterations: Strong pheromone trails on optimal/near-optimal routes (exploitation phase)
		- The evolution plot shows how pheromone levels on best-route edges grow
		
		**Comparison to Other Algorithms:**
		- More structured exploration than random search or simple hill climbing
		- Similar metaheuristic approach to Simulated Annealing but with population-based search
		- Can be slower than Simulated Annealing but offers better diversity maintenance
		
		""")
