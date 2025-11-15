from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import plotly.graph_objects as go


# ------------------------
# Data loading and metrics
# ------------------------

def load_city_coordinates(json_path: str | Path) -> Dict[str, Tuple[float, float]]:
	"""
	Load city coordinates from JSON. The JSON is expected to map city name -> [lon, lat].
	Returns a dict mapping city -> (lat, lon) in the conventional order for plotting.
	"""
	p = Path(json_path)
	with p.open("r", encoding="utf-8") as f:
		data = json.load(f)
	# Input is [lon, lat]; return (lat, lon)
	coords = {city: (float(latlon[1]), float(latlon[0])) for city, latlon in data.items()}
	return coords


def load_route_geojson(city1: str, city2: str, routes_dir: str | Path = "app/data/ant_colony_optimization/routes_carinthia") -> Optional[List[Tuple[float, float]]]:
	"""
	Load the actual route coordinates between two cities from GeoJSON files.
	Tries both city1_city2 and city2_city1 filenames.
	Returns list of (lat, lon) tuples for the route path, or None if not found.
	"""
	routes_path = Path(routes_dir)
	
	# Normalize city names (replace spaces with underscores)
	c1 = city1.replace(" ", "_")
	c2 = city2.replace(" ", "_")
	
	# Try both directions
	filename1 = routes_path / f"route_{c1}_{c2}.json"
	filename2 = routes_path / f"route_{c2}_{c1}.json"
	
	route_file = None
	reverse = False
	
	if filename1.exists():
		route_file = filename1
	elif filename2.exists():
		route_file = filename2
		reverse = True
	else:
		return None
	
	try:
		with route_file.open("r", encoding="utf-8") as f:
			data = json.load(f)
		
		# Extract coordinates from GeoJSON
		if "features" in data and len(data["features"]) > 0:
			geometry = data["features"][0]["geometry"]
			if geometry["type"] == "LineString":
				coords = geometry["coordinates"]
				# Convert [lon, lat] to (lat, lon) and reverse if needed
				route_coords = [(lat, lon) for lon, lat in coords]
				if reverse:
					route_coords = list(reversed(route_coords))
				return route_coords
	except Exception:
		return None
	
	return None


def load_route_distance(city1: str, city2: str, coords_lookup: Dict[str, Tuple[float, float]], routes_dir: str | Path = "app/data/ant_colony_optimization/routes_carinthia") -> Optional[float]:
	"""
	Return the road distance in kilometers between two cities using the route GeoJSON
	if available. Falls back to summing haversine over the geometry when segments
	are missing. Returns None if no route file exists.
	"""
	routes_path = Path(routes_dir)
	c1 = city1.replace(" ", "_")
	c2 = city2.replace(" ", "_")

	filename1 = routes_path / f"route_{c1}_{c2}.json"
	filename2 = routes_path / f"route_{c2}_{c1}.json"

	reverse = False
	if filename1.exists():
		route_file = filename1
	elif filename2.exists():
		route_file = filename2
		reverse = True
	else:
		return None

	try:
		with route_file.open("r", encoding="utf-8") as f:
			data = json.load(f)
		# Prefer properties.segments distance if present (meters)
		props = None
		if isinstance(data, dict) and "features" in data and data["features"]:
			feat0 = data["features"][0]
			props = feat0.get("properties")
			geom = feat0.get("geometry")
			if props and "segments" in props and props["segments"]:
				meters = 0.0
				for seg in props["segments"]:
					meters += float(seg.get("distance", 0.0))
				return meters / 1000.0
			# Fallback: sum haversine along geometry
			if geom and geom.get("type") == "LineString":
				coords = geom.get("coordinates", [])
				if reverse:
					coords = list(reversed(coords))
				km = 0.0
				for i in range(len(coords) - 1):
					lon1, lat1 = coords[i]
					lon2, lat2 = coords[i + 1]
					km += haversine_km(lat1, lon1, lat2, lon2)
				return km
	except Exception:
		return None

	return None


def build_distance_matrix_with_routes(cities: List[str], coords: Dict[str, Tuple[float, float]], routes_dir: str | Path = "app/data/ant_colony_optimization/routes_carinthia") -> np.ndarray:
	"""
	Build an NxN distance matrix preferring actual road distances from the routes
	files when available, and falling back to great-circle (haversine) distances.
	"""
	n = len(cities)
	dist = np.zeros((n, n), dtype=float)
	cache: Dict[Tuple[str, str], float] = {}
	for i in range(n):
		ci = cities[i]
		lat1, lon1 = coords[ci]
		for j in range(i + 1, n):
			cj = cities[j]
			# Try cache both directions
			val = cache.get((ci, cj)) or cache.get((cj, ci))
			if val is None:
				road_km = load_route_distance(ci, cj, coords, routes_dir)
				if road_km is None:
					# fallback to haversine
					lat2, lon2 = coords[cj]
					val = haversine_km(lat1, lon1, lat2, lon2)
				else:
					val = road_km
				cache[(ci, cj)] = val
			dist[i, j] = dist[j, i] = float(val)
	return dist


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
	"""Great-circle distance between two points on Earth in kilometers."""
	R = 6371.0
	phi1, phi2 = math.radians(lat1), math.radians(lat2)
	dphi = math.radians(lat2 - lat1)
	dlambda = math.radians(lon2 - lon1)
	a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
	c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
	return R * c


def build_distance_matrix(cities: List[str], coords: Dict[str, Tuple[float, float]]) -> np.ndarray:
	"""Build an NxN distance matrix using haversine distances."""
	n = len(cities)
	dist = np.zeros((n, n), dtype=float)
	for i in range(n):
		lat1, lon1 = coords[cities[i]]
		for j in range(i + 1, n):
			lat2, lon2 = coords[cities[j]]
			d = haversine_km(lat1, lon1, lat2, lon2)
			dist[i, j] = dist[j, i] = d
	return dist


def route_length(route: List[int], dist: np.ndarray, closed: bool = True) -> float:
	"""Compute total path length for visiting nodes in the given order.
	If closed=True, includes the edge from last to first (Hamiltonian cycle)."""
	total = 0.0
	for i in range(len(route) - 1):
		total += dist[route[i], route[i + 1]]
	if closed and len(route) > 1:
		total += dist[route[-1], route[0]]
	return total


# ------------------------
# Initialization
# ------------------------

def init_route_nearest_neighbor(n: int, dist: np.ndarray, rng: random.Random) -> List[int]:
	"""Initialize route using nearest neighbor heuristic."""
	start = rng.randrange(n)
	unvisited = set(range(n))
	unvisited.remove(start)
	route = [start]
	current = start
	while unvisited:
		next_city = min(unvisited, key=lambda j: dist[current, j])
		route.append(next_city)
		unvisited.remove(next_city)
		current = next_city
	return route


# -----------------------
# Plotly map visualization
# -----------------------

def plot_route_map(
	cities: List[str],
	coords: Dict[str, Tuple[float, float]],
	route: List[int],
	use_road_paths: bool = True,
	routes_dir: str | Path = "app/data/ant_colony_optimization/routes_carinthia"
) -> go.Figure:
	"""Create a map-like visualization using Plotly Scattergeo.
	Accepts a route as a list of city indices and draws actual road routes between cities
	when use_road_paths is True; otherwise draws straight-line segments.
	"""
	fig = go.Figure()
	
	# Draw routes between consecutive cities
	route_with_return = route + [route[0]]  # Close the loop
	for i in range(len(route_with_return) - 1):
		city_from = cities[route_with_return[i]]
		city_to = cities[route_with_return[i + 1]]
		
		route_coords = None
		# Try to load actual route data only if enabled
		if use_road_paths:
			route_coords = load_route_geojson(city_from, city_to, routes_dir)
		
		if route_coords:
			# Use actual route
			lats = [lat for lat, lon in route_coords]
			lons = [lon for lat, lon in route_coords]
		else:
			# Fallback to straight line if route not found
			lat_from, lon_from = coords[city_from]
			lat_to, lon_to = coords[city_to]
			lats = [lat_from, lat_to]
			lons = [lon_from, lon_to]
		
		# Add route segment
		fig.add_trace(
			go.Scattergeo(
				lat=lats,
				lon=lons,
				mode="lines",
				line=dict(width=2, color="#1f77b4"),
				showlegend=(i == 0),  # Only show legend for first segment
				name="Route",
				hoverinfo="skip",
			)
		)
	
	# City markers
	fig.add_trace(
		go.Scattergeo(
			lat=[coords[c][0] for c in cities],
			lon=[coords[c][1] for c in cities],
			text=cities,
			mode="markers+text",
			textposition="top center",
			textfont=dict(size=10, color="#2c3e50"),
			marker=dict(size=7, color="#ff7f0e"),
			name="Cities",
		)
	)
	
	# Center on Carinthia region
	fig.update_layout(
		geo=dict(
			scope="europe",
			showcountries=True,
			countrycolor="rgba(0,0,0,0.2)",
			showland=True,
			landcolor="#f7f7f7",
			center=dict(lat=46.7, lon=13.8),
			projection_type="mercator",
			lataxis_range=[46.4, 47.2],
			lonaxis_range=[12.5, 15.0],
		),
		margin=dict(l=10, r=10, t=30, b=10),
		height=520,
	)
	return fig


def plot_convergence(history_best: List[float]) -> go.Figure:
	"""Plot convergence of best distance over iterations."""
	fig = go.Figure()
	fig.add_trace(go.Scatter(y=history_best, mode="lines", name="Best distance", line=dict(color="#1f77b4")))
	fig.update_layout(
		xaxis_title="Iteration",
		yaxis=dict(title="Distance (km)"),
		height=350,
		margin=dict(l=10, r=10, t=30, b=10),
		legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
	)
	return fig


def plot_diversity(history_diversity: List[float]) -> go.Figure:
	"""Plot solution diversity over iterations."""
	fig = go.Figure()
	fig.add_trace(go.Scatter(y=history_diversity, mode="lines", name="Solution diversity", line=dict(color="#2ca02c")))
	fig.update_layout(
		xaxis_title="Iteration",
		yaxis=dict(title="Avg. distance std. dev. (km)"),
		height=350,
		margin=dict(l=10, r=10, t=30, b=10),
		legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
	)
	return fig


def plot_pheromone_heatmap(
	cities: List[str], 
	pheromone_matrix: np.ndarray,
	iteration: int,
	best_route: Optional[List[int]] = None
) -> go.Figure:
	"""
	Create a heatmap visualization of pheromone intensity between cities.
	
	Args:
		cities: List of city names
		pheromone_matrix: NxN pheromone matrix
		iteration: Current iteration number
		best_route: Optional best route to highlight
		
	Returns:
		Plotly figure with pheromone heatmap
	"""
	n = len(cities)
	
	# Create a copy to avoid modifying original
	pheromone_display = pheromone_matrix.copy()
	
	# Normalize for better visualization (log scale to handle wide range)
	# Add small epsilon to avoid log(0)
	pheromone_display = np.log10(pheromone_display + 1e-10)
	
	# Create heatmap
	fig = go.Figure(data=go.Heatmap(
		z=pheromone_display,
		x=cities,
		y=cities,
		colorscale='Viridis',
		text=[[f"{pheromone_matrix[i,j]:.4f}" for j in range(n)] for i in range(n)],
		hovertemplate='%{y} -> %{x}<br>Pheromone: %{text}<extra></extra>',
		colorbar=dict(title="$log_{10}$(Pheromone)")
	))
	
	# Highlight best route edges if provided
	if best_route is not None and len(best_route) > 1:
		route_edges = []
		for i in range(len(best_route)):
			from_idx = best_route[i]
			to_idx = best_route[(i + 1) % len(best_route)]
			route_edges.append((from_idx, to_idx))
		
		# Add rectangles to highlight best route edges
		shapes = []
		for from_idx, to_idx in route_edges:
			shapes.append(dict(
				type="rect",
				x0=to_idx - 0.5, x1=to_idx + 0.5,
				y0=from_idx - 0.5, y1=from_idx + 0.5,
				line=dict(color="red", width=2),
				fillcolor="rgba(0,0,0,0)"
			))
			# Also highlight the symmetric entry
			shapes.append(dict(
				type="rect",
				x0=from_idx - 0.5, x1=from_idx + 0.5,
				y0=to_idx - 0.5, y1=to_idx + 0.5,
				line=dict(color="red", width=2),
				fillcolor="rgba(0,0,0,0)"
			))
		
		fig.update_layout(shapes=shapes)
	
	fig.update_layout(
		title=f"Pheromone Trail Intensity at Iteration {iteration}",
		xaxis=dict(title="To City", tickangle=-45),
		yaxis=dict(title="From City"),
		height=600,
		margin=dict(l=100, r=50, t=80, b=100),
	)
	
	return fig


def plot_pheromone_evolution(
	cities: List[str],
	pheromone_snapshots: List[tuple[int, np.ndarray]],
	best_route: List[int]
) -> go.Figure:
	"""
	Create a visualization showing how pheromone evolves on best route edges over time.
	
	Args:
		cities: List of city names
		pheromone_snapshots: List of (iteration, pheromone_matrix) tuples
		best_route: Best route found
		
	Returns:
		Plotly figure showing pheromone evolution
	"""
	if not pheromone_snapshots or not best_route:
		return go.Figure()
	
	# Extract edges from best route
	route_edges = []
	edge_labels = []
	for i in range(len(best_route)):
		from_idx = best_route[i]
		to_idx = best_route[(i + 1) % len(best_route)]
		route_edges.append((from_idx, to_idx))
		edge_labels.append(f"{cities[from_idx][:3]}-{cities[to_idx][:3]}")
	
	# Sample a subset of edges if there are too many
	if len(route_edges) > 10:
		# Sample evenly
		step = len(route_edges) // 10
		route_edges = route_edges[::step]
		edge_labels = edge_labels[::step]
	
	fig = go.Figure()
	
	# Plot pheromone level for each edge over iterations
	for idx, (edge, label) in enumerate(zip(route_edges, edge_labels)):
		from_idx, to_idx = edge
		pheromone_levels = []
		iterations = []
		
		for iter_num, pheromone_matrix in pheromone_snapshots:
			iterations.append(iter_num)
			pheromone_levels.append(pheromone_matrix[from_idx, to_idx])
		
		fig.add_trace(go.Scatter(
			x=iterations,
			y=pheromone_levels,
			mode='lines+markers',
			name=label,
			line=dict(width=2),
			marker=dict(size=6)
		))
	
	fig.update_layout(
		title="Pheromone Evolution on Best Route Edges",
		xaxis_title="Iteration",
		yaxis_title="Pheromone Level",
		height=450,
		margin=dict(l=10, r=10, t=50, b=10),
		legend=dict(
			orientation="v",
			yanchor="top",
			y=1,
			xanchor="left",
			x=1.02,
			font=dict(size=9)
		),
		hovermode='x unified'
	)
	
	return fig
