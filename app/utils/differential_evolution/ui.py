from __future__ import annotations

import math
from typing import Dict, List, Optional

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from utils.general.ui import spacer

from .config import DEFAULTS, UI_HELP, WEIGHT_DEFAULTS


def display_parameter_controls() -> Dict[str, float]:
	with st.expander("Differential Evolution Configuration", expanded=True):
		col1, col2 = st.columns(2)
		with col1:
			population_size = st.slider(
				"Population size",
				min_value=12,
				max_value=80,
				value=int(DEFAULTS["population_size"]),
				step=2,
				help=UI_HELP["population"],
			)
			generations = st.slider(
				"Generations",
				min_value=20,
				max_value=200,
				value=int(DEFAULTS["generations"]),
				step=5,
				help=UI_HELP["generations"],
			)
			radial_points = st.slider(
				"Radial spokes",
				min_value=12,
				max_value=72,
				value=int(DEFAULTS["radial_points"]),
				step=4,
				help=UI_HELP["radial"],
			)
			seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=int(DEFAULTS["seed"]))
			smoothing_window = st.slider(
				"Smoothing window",
				min_value=1,
				max_value=19,
				step=2,
				value=int(DEFAULTS["smoothing_window"]),
				help=UI_HELP["smoothing"],
			)
		with col2:
			mutation_factor = st.slider(
				"Mutation factor (F)",
				min_value=0.3,
				max_value=1.2,
				value=float(DEFAULTS["mutation_factor"]),
				step=0.05,
				help=UI_HELP["mutation"],
			)
			crossover_rate = st.slider(
				"Crossover probability",
				min_value=0.1,
				max_value=1.0,
				value=float(DEFAULTS["crossover_rate"]),
				step=0.05,
				help=UI_HELP["crossover"],
			)
			rotation_increment = st.slider(
				"Rotation increment (degrees)",
				min_value=1.0,
				max_value=10.0,
				value=float(DEFAULTS["rotation_increment"]),
				step=0.5,
				help=UI_HELP["rotation"],
			)
			step_size = st.slider(
				"Translation step size",
				min_value=0.02,
				max_value=0.2,
				value=float(DEFAULTS["step_size"]),
				step=0.01,
				help="Increment used when sliding the shape through the corridor.",
			)

	return {
		"population_size": population_size,
		"generations": generations,
		"mutation_factor": mutation_factor,
		"crossover_rate": crossover_rate,
		"radial_points": radial_points,
		"rotation_increment": rotation_increment,
		"step_size": step_size,
		"seed": int(seed),
		"smoothing_window": int(smoothing_window),
	}


def display_weight_controls() -> Dict[str, float]:
	with st.expander("Objective Weights", expanded=False):
		rotation = st.slider(
			"Penalty weight: rotation",
			min_value=0.2,
			max_value=3.0,
			value=float(WEIGHT_DEFAULTS["rotation"]),
			step=0.1,
		)
		placement = st.slider(
			"Penalty weight: placement",
			min_value=0.2,
			max_value=3.0,
			value=float(WEIGHT_DEFAULTS["placement"]),
			step=0.1,
		)
		reward = st.slider(
			"Reward weight: area + ruggedness",
			min_value=0.2,
			max_value=3.0,
			value=float(WEIGHT_DEFAULTS["reward"]),
			step=0.1,
		)

	return {"rotation": rotation, "placement": placement, "reward": reward}


def _add_polygon_trace(
	fig: go.Figure,
	poly,
	name: str,
	color: str,
	opacity: float = 0.35,
	fill_color: str | None = None,
	line_width: float = 2.0,
	row: int | None = None,
	col: int | None = None,
) -> None:
	if poly.is_empty:
		return
	x, y = poly.exterior.xy
	fig.add_trace(
		go.Scatter(
			x=list(x),
			y=list(y),
			fill="toself",
			fillcolor=fill_color or color,
			name=name,
			mode="lines",
			line=dict(color=color, width=line_width),
			opacity=opacity,
		),
		row=row,
		col=col,
	)


def _build_shape_grid(snapshots, corridor):
	if not snapshots:
		return None
	cols = min(3, len(snapshots))
	rows = math.ceil(len(snapshots) / cols)
	titles = [f"Gen {snap.generation} | cost {snap.cost:.2f}" for snap in snapshots]
	fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles)
	for idx, snap in enumerate(snapshots):
		row = idx // cols + 1
		col = idx % cols + 1
		_add_polygon_trace(
			fig,
			corridor,
			"Corridor",
			"#bbbbbb",
			opacity=0.25,
			fill_color="#efefef",
			line_width=1.5,
			row=row,
			col=col,
		)
		_add_polygon_trace(fig, snap.shape, "Shape", "#4c78a8", opacity=0.55, row=row, col=col)
		axis_idx = idx + 1
		fig.update_xaxes(scaleanchor=f"y{axis_idx}", row=row, col=col, visible=False)
		fig.update_yaxes(visible=False, row=row, col=col)
	fig.update_layout(showlegend=False, height=280 * rows, margin=dict(t=60, l=20, r=20, b=20))
	return fig


def display_results(result, corridor, path: Optional[List] = None) -> None:
	evaluation = result.evaluation

	col1, col2, col3 = st.columns(3)
	col1.metric("Area", f"{evaluation.area:.4f}")
	col2.metric("Feasible turn", "Yes" if evaluation.feasible else "No", delta=f"{evaluation.max_rotation_fraction*90:.1f}°")
	col3.metric("Cost", f"{evaluation.cost:.4f}")

	spacer(1)

	st.markdown("#### Shape inside the corridor")
	fig = go.Figure()
	_add_polygon_trace(
		fig,
		corridor,
		"Corridor",
		"#333333",
		opacity=0.35,
		fill_color="#dddddd",
		line_width=3.0,
	)
	_add_polygon_trace(fig, result.best_shape, "Best shape", "#4c78a8", opacity=0.45)

	if path:
		stride = max(1, len(path) // 6)
		for idx, poly in enumerate(path[::stride]):
			_add_polygon_trace(fig, poly, f"Path step {idx}", "#f58518", opacity=0.2)

	fig.update_layout(
		showlegend=True,
		xaxis=dict(scaleanchor="y", scaleratio=1),
		height=500,
	)
	st.plotly_chart(fig, use_container_width=True)

	spacer(1)

	st.markdown("#### Convergence history")
	if result.history:
		generations = [entry["generation"] for entry in result.history]
		best_cost = [entry["best_cost"] for entry in result.history]
		mean_cost = [entry["mean_cost"] for entry in result.history]
		best_areas = [entry["best_area"] for entry in result.history]

		fig_hist = go.Figure()
		fig_hist.add_trace(go.Scatter(x=generations, y=best_cost, name="Best cost", mode="lines"))
		fig_hist.add_trace(go.Scatter(x=generations, y=mean_cost, name="Mean cost", mode="lines", line=dict(dash="dash")))
		fig_hist.add_trace(
			go.Scatter(x=generations, y=best_areas, name="Best area", mode="lines", yaxis="y2", line=dict(color="#72b7b2"))
		)
		fig_hist.update_layout(
			xaxis_title="Generation",
			yaxis_title="Cost",
			yaxis2=dict(title="Area", overlaying="y", side="right"),
			height=400,
		)
		st.plotly_chart(fig_hist, use_container_width=True)
	else:
		st.info("Run the solver to generate history data.")

	snapshots = getattr(result, "snapshots", [])
	if snapshots:
		spacer(1)
		st.markdown("#### Shape evolution")
		fig_grid = _build_shape_grid(snapshots, corridor)
		if fig_grid:
			st.plotly_chart(fig_grid, use_container_width=True)

