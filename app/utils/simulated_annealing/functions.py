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


def load_route_geojson(city1: str, city2: str, routes_dir: str | Path = "app/data/simulated_annealing/routes") -> Optional[List[Tuple[float, float]]]:
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


def load_route_distance(city1: str, city2: str, coords_lookup: Dict[str, Tuple[float, float]], routes_dir: str | Path = "app/data/simulated_annealing/routes") -> Optional[float]:
	"""
	Return the road distance in kilometers between two cities using the route GeoJSON
	if available. Falls back to summing haversine over the geometry when segments
	are missing. Returns None if no route file exists.
	"""

	# Try to load file and parse segments distances
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


def build_distance_matrix_with_routes(cities: List[str], coords: Dict[str, Tuple[float, float]]) -> np.ndarray:
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
				road_km = load_route_distance(ci, cj, coords)
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
# Initialization strategies
# ------------------------

def init_route_random(n: int, rng: random.Random) -> List[int]:
	"""Initialize a random route."""
	route = list(range(n))
	rng.shuffle(route)
	return route


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


def build_initial_route(strategy_key: str, n: int, dist: np.ndarray, seed: int | None) -> List[int]:
	"""Build initial route based on the selected strategy."""
	rng = random.Random(seed)
	if strategy_key == "random":
		return init_route_random(n, rng)
	elif strategy_key == "nearest_neighbor":
		return init_route_nearest_neighbor(n, dist, rng)
	else:
		return init_route_random(n, rng)


# -----------------------
# Neighbor perturbations
# -----------------------

def neighbor_swap(route: List[int], rng: random.Random) -> List[int]:
	"""Swap two cities in the route."""
	i, j = rng.sample(range(len(route)), 2)
	if i > j:
		i, j = j, i
	new_route = route.copy()
	new_route[i], new_route[j] = new_route[j], new_route[i]
	return new_route


def neighbor_inversion(route: List[int], rng: random.Random) -> List[int]:
	"""Perform inversion mutation by reversing a segment of the route."""
	i, j = sorted(rng.sample(range(len(route)), 2))
	return route[:i] + list(reversed(route[i:j])) + route[j:]


def neighbor_insertion(route: List[int], rng: random.Random) -> List[int]:
    """Remove a city and insert it at another position."""
    i, j = rng.sample(range(len(route)), 2)
    new = route.copy()
    city = new.pop(i)
    new.insert(j, city)
    return new


def propose_neighbor(route: List[int], strategy_key: str, rng: random.Random) -> List[int]:
	"""Propose a neighbor route based on the selected strategy."""
	if strategy_key == "swap":
		return neighbor_swap(route, rng)
	elif strategy_key == "inversion":
		return neighbor_inversion(route, rng)
	elif strategy_key == "insertion":
		return neighbor_insertion(route, rng)
	else:
		return neighbor_inversion(route, rng)


# -----------------------
# Plotly map visualization
# -----------------------

def plot_route_map(
	cities: List[str],
	coords: Dict[str, Tuple[float, float]],
	route: List[int],
	use_road_paths: bool = True,
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
			route_coords = load_route_geojson(city_from, city_to)
		
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
	fig.update_layout(
		geo=dict(
			scope="europe",
			showcountries=True,
			countrycolor="rgba(0,0,0,0.2)",
			showland=True,
			landcolor="#f7f7f7",
			center=dict(lat=47.5, lon=14.2),
			projection_type="mercator",
			lataxis_range=[46.2, 49.2],
			lonaxis_range=[8.5, 17.5],
		),
		margin=dict(l=10, r=10, t=30, b=10),
		height=520,
	)
	return fig


def plot_convergence(history_best: List[float], history_temp: List[float]) -> go.Figure:
	"""Plot convergence of best distance and temperature over iterations."""
	fig = go.Figure()
	fig.add_trace(go.Scatter(y=history_best, mode="lines", name="Best distance"))
	fig.add_trace(go.Scatter(y=history_temp, mode="lines", name="Temperature", yaxis="y2"))
	fig.update_layout(
		xaxis_title="Iteration",
		yaxis=dict(title="Distance (km)"),
		yaxis2=dict(title="Temperature", overlaying="y", side="right", showgrid=False),
		height=350,
		margin=dict(l=10, r=10, t=30, b=10),
		legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
	)
	return fig


def rng_from_seed(seed: int | None) -> random.Random:
	"""Create a random.Random instance from the given seed."""
	return random.Random(seed)


# -----------------------
# Validation helpers
# -----------------------

def is_valid_route(route: List[int], n: int) -> bool:
	"""Return True if the route is a permutation of 0..n-1 with no duplicates."""
	if len(route) != n:
		return False
	# Fast check: each index appears exactly once
	return set(route) == set(range(n))

