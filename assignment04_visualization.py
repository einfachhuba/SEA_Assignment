"""
Visualization script for assignment 04.

Run from the project root:
    python -m sea_demo.codes.assignment04_visualization

Do NOT run directly as:
    python sea_demo/codes/assignment04_visualization.py
"""

import json
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import LineString
import os
import matplotlib.cm as cm
from sea_demo.config import settings


def fetch_route(city_from: str, city_to: str):
    """
    inputs:
        - city_from: starting city name
        - city_to: target city name
    output:
        - route_gdf: geopandas frame containing the route between the given cities
    """
    file_name_alt1 = f"{settings.routes_path}route_{city_from}_{city_to}.json"
    file_name_alt2 = f"{settings.routes_path}route_{city_to}_{city_from}.json"

    # only one direction of each city pair's route stored; check which exists
    if os.path.exists(file_name_alt1):
        route_file = file_name_alt1
    elif os.path.exists(file_name_alt2):
        route_file = file_name_alt2
    else:
        raise FileNotFoundError(f"No route file found for {city_from} ↔ {city_to}")

    with open(route_file, "r") as f:
        route_data = json.load(f)

    # fetch the routes coordinates and store in a geopandas dataframe
    coords = route_data["features"][0]["geometry"]["coordinates"]
    route_line = LineString(coords)
    route_gdf = gpd.GeoDataFrame(geometry=[route_line], crs="EPSG:4326")
    return route_gdf



def plot_route(route: list[str]):
    """
    plot a route using geolocation data from geopandas.
    inputs: 
    - route: list of strings (city names) displaying the sequence of travel. possible
        city names are: "Vienna", "St._Pölten", "Linz", "Salzburg", "Graz", 
            "Eisenstadt", "Klagenfurt", "Lienz", Innsbruck", "Bregenz"
        example: ["Graz", "Vienna", "Linz"]
    output: figure showing the travel route on a geomap
    """

    with open(settings.summary_path, "r", encoding="utf-8") as f:
        routes = json.load(f)

    with open(settings.coordinates_path, "r", encoding="utf-8") as f:
        coords = json.load(f)
   
    world = gpd.read_file(settings.gpd_url)
    austria = world[world["NAME"] == "Austria"]

    _, ax = plt.subplots(figsize=(8, 6))
    austria.plot(ax=ax, color="beige", edgecolor="gray")
    cmap = plt.get_cmap("viridis", len(route)-1)

    # plot route lines
    for i in range(len(route)-1):
        color = cmap(i)
        current_route = fetch_route(route[i], route[i+1])
        current_route.plot(ax=ax, color=color, linewidth=2)

    # plot cities
    for city, (lon, lat) in coords.items():
        ax.scatter(lon, lat, color="black", s=50)
        ax.text(lon + 0.1, lat + 0.1, city, fontsize=8)

    ax.set_title("TSP Route Across Austrian Cities", fontsize=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.show()


if __name__ == "__main__":
    example_route = ["Graz", "Klagenfurt", "Villach", "Spittal_an_der_Drau", "Salzburg",
        "Graz"]
    plot_route(example_route)