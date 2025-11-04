import streamlit as st
from typing import List
from utils.general.pdf_viewer import display_pdf_with_controls
from utils.general.ui import spacer
from utils.simulated_annealing.ui import (
	display_city_selection,
	display_parameter_controls,
	display_results,
)
from utils.simulated_annealing.algorithms import SimulatedAnnealingTSP
from utils.simulated_annealing.functions import build_distance_matrix, build_distance_matrix_with_routes


st.set_page_config(page_title="AT04: Simulated Annealing")

tab1, tab2 = st.tabs(["📄 Assignment Paper", "🔧 Simulated Annealing Implementation"])

with tab1:
	st.title("Assignment 04: Simulated Annealing")
	pdf_path = "Assignment_Sheets/04/SEA_Exercise04_Simulated_Annealing.pdf"
	try:
		display_pdf_with_controls(pdf_path)
	except Exception:
		st.info("Assignment PDF not found. Proceed to the implementation tab.")

with tab2:
	st.title("Assignment 04: Simulated Annealing Implementation")
	
	st.header("Introduction")
	st.markdown("""
	**Simulated Annealing** is a probabilistic optimization technique inspired by the physical process of 
	annealing in metallurgy. SA starts with a high "temperature" that allows exploration 
	of the solution space by accepting worse solutions, then gradually "cools down" to focus on exploitation 
	and refinement. We implemented SA to the **Traveling Salesperson Problem (TSP)** - finding the shortest route 
	visiting a set of cities exactly once and returning to the start.
	""")

	spacer(6)

	cont = st.container(border=True)
	with cont:
		col1, col2 = st.columns(2)
		with col1:
			st.badge("**Key Features:**", color="blue")
			st.markdown("""
			- **Probabilistic acceptance**: Accepts worse solutions with probability $P = exp(-\Delta E/T)$
			- **Temperature control**: Exploration -> Exploitation transition via cooling schedules
			- **Neighbor generation**: Multiple strategies (inversion, swap, insertion)
			- **Flexible initialization**: Random or greedy (nearest neighbor) starting tours
			- **Real road distances**: Uses actual route data when available
			""")

		with col2:
			st.badge("**Problem Characteristics:**", color="blue")
			st.markdown("""
			- **Discrete search space**: Permutations of city sequences
			- **Combinatorial explosion**: *O((n-1)!/2)* possible tours
			- **Distance metric**: Road distances (km) from OpenRouteService data
			- **Visualization**: Geographic map with actual road paths
			- **Quality metric**: Total tour distance in kilometers
			""")

	spacer(24)

	st.header("Methods")
	
	st.markdown("""
	### Simulated Annealing Algorithm
	
	The algorithm follows the classic Metropolis-Hastings acceptance criterion:
	
	1. **Initialization**: Generate an initial tour (random or nearest-neighbor greedy)
	2. **Main Loop**: While $T > T_{min}$ and iterations remain:
	   - Generate a neighbor solution (alter current tour)
	   - Calculate energy difference: $\Delta E$ = distance(new) - distance(current)
	   - Accept new solution if:
	     - $\Delta E \\leq 0$ (better solution) -> always accept
	     - $\Delta E > 0$ (worse solution) -> accept with probability $P = exp(-\Delta E/T)$
	   - Cool down: $T = \\alpha \cdot T$ according to cooling schedule
	3. **Termination**: Stop when temperature reaches $T_{min}$ or max iterations exceeded
	
	### Cooling Schedules
	
	Different cooling strategies control the exploration-exploitation trade-off:

	- **Exponential (Geometric)**: $T_{k+1} = \\alpha \cdot T_k$ where $\\alpha$ in [0.90, 0.999]
	  - Most common and reliable; slow cooling with $\\alpha$ close to 1 often yields better solutions

	- **Linear**: $T_{k+1} = \max( T_k - \\eta, T_{min})$
	  - Simple but "unphysical"; can cool too quickly

	- **Inverse**: $T_{k+1} = T_k / (1 + \\beta \cdot T_k)$
	  - Can achieve faster cooling than exponential depending on $\\beta$

	- **Logarithmic**: $T_k = T_0 / \ln(k+2)$
	  - Very slow theoretical schedule; impractical for finite iterations

	- **Inverse Linear**: $T_k = T_0 / k$
	  - Harmonic-like cooling; temperature decreases as $1/k$

	### Neighbor Generation Strategies
	
	How we alter the current tour to explore nearby solutions:
	
	- **Inversion**: Reverse a random segment of the tour (inversion mutation)
	  - Most effective for TSP; breaks crossing edges and creates smoother paths
	  
	- **Swap**: Exchange positions of two random cities
	  - Simple and fast but less effective for TSP than inversion
	  
	- **Insertion**: Remove a city and reinsert it at a different position
	  - Good for local refinement
	
	### Distance Calculation
	
	**Road distances** are extracted from OpenRouteService GeoJSON route files when available:
	- Each route file contains actual driving paths with distance in meters
	- Fallback to great-circle (Haversine) distance for missing city pairs
	- Toggle available to use straight-line distances for comparison
	""")

	spacer(24)

	st.header("Configuration & Execution")
	
	st.markdown("""
	Select cities, configure parameters, and run the algorithm below. Results will show the best tour found, 
	visualized on a map with actual road routes, along with convergence plots.
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

	spacer(12)

	if st.button("Run Simulated Annealing", type="primary", use_container_width=True, disabled=len(selected_cities) < 6):
		# Generate unique run ID to avoid key conflicts across multiple runs
		import time
		run_id = int(time.time() * 1000)
		
		# Build distance matrix (prefer road distances if selected)
		if use_road_distances:
			dist = build_distance_matrix_with_routes(selected_cities, coords)
		else:
			dist = build_distance_matrix(selected_cities, coords)

		# Initialize solver
		sa = SimulatedAnnealingTSP(
			dist,
			initial_temperature=params["initial_temperature"],
			minimum_temperature=params["minimum_temperature"],
			max_iterations=params["max_iterations"],
			iterations_per_temperature=params["iterations_per_temperature"],
			cooling_method=params["cooling_strategy"],
			alpha=params["alpha"],
			beta=params["beta"],
			eta=params["eta"],
			neighbor_strategy=params["neighbor_strategy"],
			initialization_strategy=params["initialization_strategy"],
		)

		# Create visualization placeholders
		progress_bar = st.progress(0.0)
		status_text = st.empty()
		viz_placeholder = st.empty()
		
		# Import plot function here to avoid circular imports
		from utils.simulated_annealing.functions import plot_route_map

		def on_progress(iteration: int, best_len: float, T: float):
			# simple progress approximation
			denom = max(1, params["max_iterations"]) 
			progress_bar.progress(min(1.0, iteration / denom))
			status_text.text(f"iter={iteration} | best={best_len:.1f} km | T={T:.4f}")

		# Counter to ensure each visualization has a unique key
		viz_counter = [0]
		
		def on_viz_update(iteration: int, route: List[int], distance: float):
			# Update live route visualization with unique key per run
			viz_counter[0] += 1
			with viz_placeholder.container():
				st.caption(f"Current best route at iteration {iteration}: {distance:.1f} km")
				fig = plot_route_map(selected_cities, coords, route, use_road_paths=use_road_distances)
				st.plotly_chart(fig, use_container_width=True, key=f"route_viz_{run_id}_{viz_counter[0]}")
				# Force Streamlit to update the display
				import time
				time.sleep(0.3)  # Small delay to make evolution visible

		with st.spinner("Running simulated annealing..."):
			results = sa.run(
				progress_callback=on_progress,
				viz_callback=on_viz_update,
				viz_interval=max(50, params["max_iterations"] // 25)  # Update ~25 times during run
			)

		progress_bar.empty()
		status_text.empty()
		viz_placeholder.empty()

		spacer(12)
		
		# Store results in session state
		st.session_state['sa_results'] = results
		st.session_state['sa_cities'] = selected_cities
		st.session_state['sa_coords'] = coords
		st.session_state['sa_params'] = params
		st.session_state['sa_use_road_distances'] = use_road_distances
	
	spacer(24)
	
	# Display results if they exist
	if 'sa_results' in st.session_state:
		results = st.session_state['sa_results']
		selected_cities = st.session_state['sa_cities']
		coords = st.session_state['sa_coords']
		params = st.session_state.get('sa_params', {})
		use_road_distances = st.session_state.get('sa_use_road_distances', True)
		
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
		
		Simulated Annealing demonstrates several characteristic behaviors when solving TSP instances:
		
		**Temperature and Acceptance Rate:**
		- At high temperatures (early iterations), the algorithm accepts around 40 - 60% of worse solutions
		- This broad exploration helps escape local optima and discover promising regions
		- As temperature decreases, acceptance becomes more selective, focusing on refinement
		- The final "frozen" state (T -> 0) behaves like pure hill climbing
		
		**Cooling Schedule Impact:**
		- **Exponential cooling** ( $\\alpha$ = 0.995 - 0.999) provides the best quality-time trade-off for most problems
		- Slower cooling ( $\\alpha$ closer to 1) generally finds better solutions but takes longer
		- **Linear cooling** can be too aggressive, sometimes missing good solutions
		- **Logarithmic cooling** is theoretically guaranteed to find the global optimum but impractically slow
		
		**Neighbor Strategy Effectiveness:**
		- **Inversion** is highly effective for TSP because it directly removes edge crossings (when tour paths intersect) which, always indicate suboptimal solutions
		- **Swap** and **Insertion** are less effective but computationally cheaper
		- Combining strategies (adaptive neighbor selection) can improve robustness

		**Advantages of SA:**
		- Simple to implement with few parameters
		- Theoretical convergence guarantees (with slow enough cooling)
		- Works well on small to medium TSP instances
		- No population maintenance overhead (single solution trajectory)
		
		**Limitations:**
		- Slower than specialized TSP algorithms (e.g., Lin-Kernighan, Concorde)
		- Parameter tuning ( $T_0$, cooling schedule) significantly affects performance
		- No diversity mechanism - can get stuck in local optima despite probabilistic acceptance
		- Computational cost grows with problem size
		
		### Parameter Tuning Guidelines
		
		**Initial Temperature ($T_0$):**
		- Rule of thumb: Set so around 80 - 90 % of worse moves are initially accepted
		- For TSP: $T_0$ ≈ average distance between cities x 2-5
		- Too high: wastes iterations on random search
		- Too low: premature convergence

		**Cooling Rate ($\\alpha$ for exponential):**
		- Fast: $\\alpha$ = 0.90 - 0.95 (quick, lower quality)
		- Medium: $\\alpha$ = 0.95 - 0.98 (balanced)
		- Slow: $\\alpha$ = 0.98 - 0.999 (high quality, slower)

		**Beta ($\\beta$ for inverse cooling):**
		- Typical: $\\beta$ in $[10^{-4}, 10^{-2}]$ (default often $10^{-3}$)
		- Larger $\\beta$ cools faster (temperature drops more per step)
		- Rule of thumb: choose $\\beta$ so $T$ reaches $T_{min}$ near the end of the run

		**Eta ($\\eta$ for linear cooling):**
		- Step size per level: $T_{k+1} = \\max(T_k - \\eta, T_{min})$
		- Larger $\\eta$ = more aggressive cooling
		- Smaller $\\eta$ = gentler, higher-quality search

		**Iterations per Temperature:**
		- Should explore a reasonable fraction of the neighborhood
		- Typical: 20 - 100 iterations per temperature level
		- Larger problems need more iterations per level
		""")
		
		spacer(24)
		
		st.header("Conclusion")
		cont = st.container(border=True)
		with cont:
			st.badge("Key Takeaways:", color="blue")
			st.markdown("""
            - **Temperature control is crucial**: The cooling schedule determines solution quality vs computation time
            - **Neighbor strategy matters**: Inversion significantly outperforms simpler alternations for TSP
            - **Smart initialization helps**: Nearest-neighbor starting tours improve both speed and final quality
            """)
	
	else:
		st.info("Configure parameters above and click 'Run Simulated Annealing' to see results and analysis.")

