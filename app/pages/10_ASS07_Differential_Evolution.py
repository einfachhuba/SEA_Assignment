from __future__ import annotations

import streamlit as st

from utils.general.pdf_viewer import display_pdf_with_controls
from utils.general.ui import spacer
from utils.differential_evolution.algorithms import DifferentialEvolutionResult, DifferentialEvolutionSolver
from utils.differential_evolution.config import CORRIDOR_SETTINGS, SHAPE_BOUNDS
from utils.differential_evolution.functions import compute_path_preview, construct_corridor, hammersley_sofa
from utils.differential_evolution.ui import display_parameter_controls, display_results, display_weight_controls


st.set_page_config(page_title="ASS07: Differential Evolution")

tab1, tab2 = st.tabs(["📄 Assignment Paper", "🔧 Differential Evolution"])

with tab1:
	st.title("Assignment 07: Differential Evolution & the Moving Sofa")
	pdf_path = "Assignment_Sheets/07/SEA_Exercise07_Differential_Evolution.pdf"
	try:
		display_pdf_with_controls(pdf_path)
	except Exception:
		st.info("Assignment PDF not found. Proceed to the implementation tab.")

with tab2:
	st.title("Assignment 07: Moving Sofa Optimization")
	st.header("Introduction")
	st.markdown(
		"""
		We tackle the **Moving Sofa problem** with **Differential Evolution (DE)**. The corridor looks like an L with
		one-unit-wide legs. Our goal is to evolve a 2D shape that can slide around the corner while maximizing its area.
		The helper code handles the corridor geometry, collision checks, and
		feasibility penalties.
		"""
	)

	st.markdown(
		"""
		Differential Evolution keeps a whole crowd of candidate sofas and lets them trade geometry tweaks so the good ideas spread.
		Each new generation is basically the old one plus some noisy mashups, and we keep whatever actually fits and grows the area.
		"""
	)

	spacer(4)

	st.header("Problem Setup")
	st.markdown(
		"""
		* Shapes are described by radial spokes sampled around the origin so every dimension matches a fixed angle.
		* Initial populations mix circles, rectangles, ellipses, and barbell-like silhouettes for shape diversity.
		* The objective rewards the covered area and penalises invalid placements plus insufficient rotation.
		* We benchmark the evolved shapes against the classical **Hammersley sofa** (area 2.2074) for sanity checks.
		"""
	)

	spacer(2)

	st.header("Method Overview")
	st.markdown(
		"""
		1. **Population initialization:** Create a pool of radial profiles by sampling from the allowed bounds and seeding it with reference shapes such as circles or ellipses.
		2. **Mutation:** For every target sofa, pick three other sofas and form a mutant vector `base + F * (diff1 - diff2)` that injects geometry changes.
		3. **Crossover:** Mix mutant and target spokes according to the crossover rate so only a subset of angles inherits the new radii.
		4. **Evaluation:** Slide each trial sofa through the L-corridor, compute area, and apply the feasibility/rotation penalties to obtain the cost.
		5. **Selection:** Keep whichever candidate (old or new) yields the better penalized cost; repeat until the generation maximum is reached.
		
		The loop maintains a diverse frontier because every new shape is tied to multiple parents while costs/penalties steer the search toward feasible, high-area sofas.
		"""
	)

	spacer(2)

	params = display_parameter_controls()
	spacer(1)
	weight_params = display_weight_controls()
	spacer(2)

	st.subheader("Reference Shape")
	ref_area = hammersley_sofa().area
	st.write(f"Hammersley sofa area ≈ {ref_area:.4f} a.u.")
	corridor = construct_corridor(**CORRIDOR_SETTINGS)
	shape_bounds = {
		**SHAPE_BOUNDS,
		"corridor_width": CORRIDOR_SETTINGS["corridor_width"],
	}

	if st.button("Run Differential Evolution", type="primary", use_container_width=True):
		progress = st.progress(0.0)
		status = st.empty()

		def on_progress(gen: int, best_cost: float, best_area: float) -> None:
			ratio = min(gen / params["generations"], 1.0)
			progress.progress(ratio)
			status.text(f"Generation {gen}: best cost {best_cost:.4f}, area {best_area:.4f}")

		solver = DifferentialEvolutionSolver(
			corridor=corridor,
			bounds=shape_bounds,
			weights=weight_params,
			population_size=params["population_size"],
			generations=params["generations"],
			mutation_factor=params["mutation_factor"],
			crossover_rate=params["crossover_rate"],
			radial_points=params["radial_points"],
			rotation_increment=params["rotation_increment"],
			step_size=params["step_size"],
			seed=params["seed"],
				smoothing_window=params["smoothing_window"],
		)

		with st.spinner("Evolving candidate shapes..."):
			result: DifferentialEvolutionResult = solver.run(on_progress=on_progress)

		feasible, _, path = compute_path_preview(
			corridor,
			result.best_shape,
			step_size=params["step_size"],
			rotation_increment=params["rotation_increment"],
			max_frames=80,
		)

		progress.empty()
		status.empty()

		st.session_state["de_result"] = result
		st.session_state["de_path"] = path
		st.session_state["de_corridor"] = corridor

	spacer(4)

	if "de_result" in st.session_state:
		display_results(
			st.session_state["de_result"],
			corridor=st.session_state.get("de_corridor", corridor),
			path=st.session_state.get("de_path", []),
		)

		spacer(4)
		st.header("Discussion")
		st.markdown(
			"""
			### Algorithm Behavior

			**Pros**

			- Naturally maintains population diversity via mutation/crossover.
			- Handles non-differentiable feasibility checks because it only needs cost comparisons.
			- Vectorized mutation/crossover operations let us evaluate many sofas per generation efficiently.


			**Cons**

			- Sensitive to population initialization; weak diversity collapses to circles quickly.
			- Objective evaluations are costly because every candidate must be slid through the full corridor simulation.
			- No guarantee of feasibility; strong penalties are needed to guide the population.

			### Parameter Influence
			- **Population size / generations:** Larger values explore more shapes but increase runtime quadratically with `radial_points`.
			- **Mutation factor (F):** High values encourage bold geometry changes; low values fine-tune existing silhouettes.
			- **Crossover rate:** Controls how much of the mutant vector is copied. Near 1.0 speeds up exploration but can destabilize good structures.
			- **Radial spokes:** More spokes allow detailed outlines but also enlarge the search space; a rough grid is often sufficient early on.
			- **Rotation increment / step size:** Smaller increments provide accurate feasibility tests yet slow simulations; larger steps may misclassify near-feasible sofas.
			- **Objective weights:** Raising the rotation/placement penalties pushes validity; amplifying the reward term prioritizes area even if some candidates fail.
			"""
		)
	else:
		st.info("Configure the parameters and run the solver to see results.")

