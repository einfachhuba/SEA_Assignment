import numpy as np
import plotly.graph_objects as go
import plotly.express as px


def coffee_fitness_4d(roast: float, blend: float, grind: float, brew_time: float) -> float:
    """
    Fictional 4D coffee quality fitness function
    (From SEA-demo repository by hannahwimmer)
    ---------------------------------------------
    Parameters:
        roast (float):  [0, 20]
        blend (float):  [0, 100]
        grind (float):  [0, 10]
        brew_time (float): [0.0, 5.0] (minutes)

    Returns:
        float: quality score in [0, 100]

    The function is intentionally multimodal, with many local optima and
    a clear global optimum region around ideal values.
    """

    # --- normalize inputs to [0, 1] ---
    R = np.clip(roast / 20.0, 0, 1)
    B = np.clip(blend / 100.0, 0, 1)
    G = np.clip(grind / 10.0, 0, 1)
    T = np.clip(brew_time / 5.0, 0, 1)

    # --- sinusoidal landscape for local optima ---
    base_pattern = (
        np.sin(6 * np.pi * R) * np.cos(4 * np.pi * B) +
        np.sin(5 * np.pi * G) * np.cos(3 * np.pi * T) +
        0.5 * np.sin(2 * np.pi * (R + B + G + T))
    )

    # --- smooth "global optimum" Gaussian region ---
    # Ideal combination: medium roast, balanced blend, mid grind, moderate brew
    ideal = np.exp(
        -((R - 0.6)**2 / 0.015)
        -((B - 0.5)**2 / 0.02)
        -((G - 0.5)**2 / 0.02)
        -((T - 0.55)**2 / 0.015)
    )

    # --- cross-interaction term to couple dimensions (non-separable landscape) ---
    interactions = 0.2 * np.sin(3 * np.pi * R * B) + 0.15 * np.cos(4 * np.pi * G * T)

    # --- combine components ---
    score = 0.6 * ideal + 0.3 * base_pattern + interactions

    # --- add a small asymmetry (e.g., bitterness penalty) ---
    bitterness = 0.6 * R + 0.4 * T
    if bitterness > 0.7:
        score -= 0.2 * (bitterness - 0.7) ** 2

    # --- scale and clip to [0, 100] ---
    quality = np.clip(50 + 50 * score, 0, 100)

    return float(quality)


def fitness_from_chromosome(chromosome: np.ndarray) -> float:
    """
    Wrapper function to evaluate fitness from a GA chromosome.
    
    Parameters:
        chromosome (np.ndarray): [roast, blend, grind, brew_time] encoded as floats
                                roast: [0, 20]
                                blend: [0, 100] 
                                grind: [0, 10]
                                brew_time: [0.0, 5.0]
    
    Returns:
        float: quality score in [0, 100]
    """
    roast, blend, grind, brew_time = chromosome
    
    # Ensure values are within bounds
    roast = np.clip(roast, 0, 20)
    blend = np.clip(blend, 0, 100)
    grind = np.clip(grind, 0, 10)
    brew_time = np.clip(brew_time, 0.0, 5.0)
    
    return coffee_fitness_4d(roast, blend, grind, brew_time)


def plot_fitness_grid_interactive(fitness_function, fixed_dims, fixed_values, grid_points=100):
    """
    Create an interactive 2D contour plot using Plotly.
    
    Args:
        fitness_function: callable, e.g., fitness_function(roast, blend, grind, brew_time)
        fixed_dims: list of two strings, e.g., ['roast', 'blend'] (dims to keep fixed)
        fixed_values: list of two floats, values for the fixed dimensions
        grid_points: int, resolution of the grid
    
    Returns:
        plotly.graph_objects.Figure: Interactive 2D contour plot
    """
    # all dimension names
    all_dims = ['roast', 'blend', 'grind', 'brew_time']
    variable_dims = [d for d in all_dims if d not in fixed_dims]
    limits = [20, 100, 10, 5.0]
    variable_lims = [l for (l, d) in zip(limits, all_dims) if d not in fixed_dims]

    # create grid
    X_vals = np.linspace(0, variable_lims[0], grid_points)
    Y_vals = np.linspace(0, variable_lims[1], grid_points)
    X, Y = np.meshgrid(X_vals, Y_vals)
    Z = np.zeros_like(X)

    # compute fitness
    for i in range(grid_points):
        for j in range(grid_points):
            args = dict(zip(fixed_dims, fixed_values))
            args[variable_dims[0]] = X[i,j]
            args[variable_dims[1]] = Y[i,j]
            Z[i,j] = fitness_function(**args)

    # Create interactive contour plot
    fig = go.Figure()
    
    # Add contour plot
    fig.add_trace(go.Contour(
        x=X_vals,
        y=Y_vals,
        z=Z,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title='Quality Score'),
        contours=dict(
            showlabels=True,
            labelfont=dict(size=10, color='white'),
            start=Z.min(),
            end=Z.max(),
            size=(Z.max() - Z.min()) / 20
        ),
        hovertemplate=f'<b>{variable_dims[0].capitalize()}</b>: %{{x:.2f}}<br>' +
                     f'<b>{variable_dims[1].capitalize()}</b>: %{{y:.2f}}<br>' +
                     f'<b>Fitness</b>: %{{z:.2f}}<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title=f'Interactive Fitness Landscape<br>(Fixed {fixed_dims[0].capitalize()}={fixed_values[0]:.1f}, {fixed_dims[1].capitalize()}={fixed_values[1]:.1f})',
        xaxis_title=variable_dims[0].capitalize(),
        yaxis_title=variable_dims[1].capitalize(),
        width=600,
        height=500,
        margin=dict(l=0, r=0, b=0, t=60)
    )
    
    return fig


def plot_fitness_evolution_interactive(fitness_history, diversity_history=None, title="Fitness Evolution"):
    """
    Create an interactive fitness evolution plot using Plotly.
    
    Args:
        fitness_history: List of best fitness values per generation
        diversity_history: Optional list of diversity values per generation
        title: Plot title
    
    Returns:
        plotly.graph_objects.Figure: Interactive fitness evolution plot
    """
    generations = list(range(len(fitness_history)))
    
    fig = go.Figure()
    
    # Add fitness trace
    fig.add_trace(go.Scatter(
        x=generations,
        y=fitness_history,
        mode='lines+markers',
        name='Best Fitness',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=6),
        hovertemplate='<b>Generation</b>: %{x}<br>' +
                     '<b>Best Fitness</b>: %{y:.2f}<extra></extra>'
    ))
    
    # Add diversity on secondary y-axis if provided
    if diversity_history is not None:
        fig.add_trace(go.Scatter(
            x=generations,
            y=diversity_history,
            mode='lines',
            name='Population Diversity',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            yaxis='y2',
            hovertemplate='<b>Generation</b>: %{x}<br>' +
                         '<b>Diversity</b>: %{y:.3f}<extra></extra>'
        ))
        
        # Update layout for dual y-axis
        fig.update_layout(
            yaxis=dict(
                title='Best Fitness',
                side='left',
                color='#1f77b4'
            ),
            yaxis2=dict(
                title='Population Diversity',
                side='right',
                overlaying='y',
                color='#ff7f0e'
            )
        )
    else:
        fig.update_layout(yaxis_title='Best Fitness')
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Generation',
        width=800,
        height=400,
        hovermode='x unified',
        margin=dict(l=0, r=0, b=0, t=40),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig


def plot_population_evolution_2d_interactive(population_history, fitness_function, bounds, 
                                           param_x=0, param_y=1, generations_to_show=None,
                                           show_landscape=True):
    """
    Create an interactive 2D population evolution plot using Plotly.
    
    Args:
        population_history: List of population arrays for each generation
        fitness_function: Function to evaluate fitness
        bounds: Parameter bounds
        param_x: Index of parameter for x-axis
        param_y: Index of parameter for y-axis
        generations_to_show: List of specific generations to show
        show_landscape: Whether to show background fitness landscape
    
    Returns:
        plotly.graph_objects.Figure: Interactive 2D population evolution plot
    """
    param_names = ['roast', 'blend', 'grind', 'brew_time']
    
    if generations_to_show is None:
        total_gens = len(population_history)
        if total_gens <= 10:
            generations_to_show = list(range(total_gens))
        else:
            generations_to_show = list(range(0, total_gens, max(1, total_gens // 10)))
            if generations_to_show[-1] != total_gens - 1:
                generations_to_show.append(total_gens - 1)
    
    fig = go.Figure()
    
    # Add background fitness landscape
    if show_landscape:
        grid_size = 50
        x_range = np.linspace(bounds[param_x][0], bounds[param_x][1], grid_size)
        y_range = np.linspace(bounds[param_y][0], bounds[param_y][1], grid_size)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.zeros_like(X)
        
        # Use middle values for other parameters
        fixed_values = []
        for i in range(len(bounds)):
            if i not in [param_x, param_y]:
                fixed_values.append((bounds[i][0] + bounds[i][1]) / 2)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                chromosome = [0] * len(bounds)
                chromosome[param_x] = X[i, j]
                chromosome[param_y] = Y[i, j]
                
                fixed_idx = 0
                for k in range(len(bounds)):
                    if k not in [param_x, param_y]:
                        chromosome[k] = fixed_values[fixed_idx]
                        fixed_idx += 1
                
                Z[i, j] = fitness_function(np.array(chromosome))
        
        # Add contour background
        fig.add_trace(go.Contour(
            x=x_range,
            y=y_range,
            z=Z,
            colorscale='Viridis',
            opacity=0.4,
            showscale=False,
            contours=dict(showlabels=False),
            hoverinfo='skip',
            name='Fitness Landscape'
        ))
    
    # Color scale for generations
    colors = px.colors.sample_colorscale('plasma', [i/len(generations_to_show) for i in range(len(generations_to_show))])
    
    # Add population evolution traces
    for i, gen in enumerate(generations_to_show):
        population = population_history[gen]
        fitness_values = [fitness_function(ind) for ind in population]
        
        x_coords = population[:, param_x]
        y_coords = population[:, param_y]
        
        # Determine marker properties
        if gen == 0:
            name = f'Generation {gen} (Initial)'
            marker_symbol = 'circle'
            marker_size = 10
        elif gen == len(population_history) - 1:
            name = f'Generation {gen} (Final)'
            marker_symbol = 'star'
            marker_size = 15
        else:
            name = f'Generation {gen}'
            marker_symbol = 'circle'
            marker_size = 8
        
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers',
            name=name,
            marker=dict(
                size=marker_size,
                color=fitness_values,
                colorscale='Viridis',
                symbol=marker_symbol,
                opacity=0.8,
                line=dict(width=1, color='black'),
                colorbar=dict(title='Fitness') if i == 0 and not show_landscape else None,
                showscale=(i == 0 and not show_landscape)
            ),
            hovertemplate=f'<b>Generation</b>: {gen}<br>' +
                         f'<b>{param_names[param_x].capitalize()}</b>: %{{x:.2f}}<br>' +
                         f'<b>{param_names[param_y].capitalize()}</b>: %{{y:.2f}}<br>' +
                         f'<b>Fitness</b>: %{{marker.color:.2f}}<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title=f'Interactive Population Evolution: {param_names[param_x].capitalize()} vs {param_names[param_y].capitalize()}',
        xaxis_title=param_names[param_x].capitalize(),
        yaxis_title=param_names[param_y].capitalize(),
        width=800,
        height=600,
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    return fig




def plot_fitness_3d_interactive(fitness_function, fixed_dim, fixed_value, grid_points=30):
    """
    Create an interactive 3D surface plot using Plotly.
    
    Args:
        fitness_function: callable, e.g., fitness_function(roast, blend, grind, brew_time)
        fixed_dim: string, e.g., 'brew_time' (dimension to keep fixed)
        fixed_value: float, value for the fixed dimension
        grid_points: int, resolution of the grid
    
    Returns:
        plotly.graph_objects.Figure: Interactive 3D plot
    """
    # all dimension names and limits
    all_dims = ['roast', 'blend', 'grind', 'brew_time']
    limits = [20, 100, 10, 5.0]
    
    variable_dims = [d for d in all_dims if d != fixed_dim]
    variable_lims = [l for (l, d) in zip(limits, all_dims) if d != fixed_dim]
    
    # Create 3D grid for the first two variable dimensions
    X_vals = np.linspace(0, variable_lims[0], grid_points)
    Y_vals = np.linspace(0, variable_lims[1], grid_points)
    X, Y = np.meshgrid(X_vals, Y_vals)
    
    # Create surfaces for different values of the third variable
    Z_third_vals = np.linspace(0, variable_lims[2], 6)  # 6 different levels
    
    fig = go.Figure()
    
    # Color scale for different surfaces
    colors = px.colors.qualitative.Plotly
    
    for i, third_val in enumerate(Z_third_vals):
        Z = np.zeros_like(X)
        
        # Compute fitness for this slice
        for row in range(grid_points):
            for col in range(grid_points):
                args = {}
                args[fixed_dim] = fixed_value
                args[variable_dims[0]] = X[row, col]
                args[variable_dims[1]] = Y[row, col]
                args[variable_dims[2]] = third_val
                Z[row, col] = fitness_function(**args)
        
        # Add surface to plot
        fig.add_trace(go.Surface(
            x=X,
            y=Y,
            z=Z,
            name=f'{variable_dims[2]}={third_val:.1f}',
            colorscale='Viridis',
            opacity=0.7,
            showscale=(i == 0),  # Only show colorbar for first surface
            hovertemplate=f'<b>{variable_dims[0].capitalize()}</b>: %{{x:.1f}}<br>' +
                         f'<b>{variable_dims[1].capitalize()}</b>: %{{y:.1f}}<br>' +
                         f'<b>{variable_dims[2].capitalize()}</b>: {third_val:.1f}<br>' +
                         f'<b>Fitness</b>: %{{z:.2f}}<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title=f'Interactive 3D Fitness Landscape<br>(Fixed {fixed_dim.capitalize()} = {fixed_value})',
        scene=dict(
            xaxis_title=variable_dims[0].capitalize(),
            yaxis_title=variable_dims[1].capitalize(),
            zaxis_title='Fitness',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        width=800,
        height=600,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    return fig


def plot_population_evolution_3d_interactive(population_history, fitness_function, bounds, 
                                           param_x=0, param_y=1, param_z=2, 
                                           generations_to_show=None):
    """
    Create an interactive 3D scatter plot showing population evolution using Plotly.
    
    Args:
        population_history: List of population arrays for each generation
        fitness_function: Function to evaluate fitness
        bounds: Parameter bounds
        param_x, param_y, param_z: Indices of parameters for x, y, z axes
        generations_to_show: List of specific generations to show
    
    Returns:
        plotly.graph_objects.Figure: Interactive 3D scatter plot
    """
    param_names = ['roast', 'blend', 'grind', 'brew_time']
    
    if generations_to_show is None:
        total_gens = len(population_history)
        if total_gens <= 10:
            generations_to_show = list(range(total_gens))
        else:
            generations_to_show = list(range(0, total_gens, max(1, total_gens // 10)))
            if generations_to_show[-1] != total_gens - 1:
                generations_to_show.append(total_gens - 1)
    
    fig = go.Figure()
    
    # Color scale for generations
    colors = px.colors.sample_colorscale('plasma', [i/len(generations_to_show) for i in range(len(generations_to_show))])
    
    for i, gen in enumerate(generations_to_show):
        population = population_history[gen]
        
        # Calculate fitness for each individual
        fitness_values = [fitness_function(ind) for ind in population]
        
        x_coords = population[:, param_x]
        y_coords = population[:, param_y]
        z_coords = population[:, param_z]
        
        # Determine marker properties
        if gen == 0:
            name = f'Generation {gen} (Initial)'
            marker_symbol = 'circle'
            marker_size = 6
        elif gen == len(population_history) - 1:
            name = f'Generation {gen} (Final)'
            marker_symbol = 'diamond'
            marker_size = 10
        else:
            name = f'Generation {gen}'
            marker_symbol = 'circle'
            marker_size = 5
        
        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers',
            name=name,
            marker=dict(
                size=marker_size,
                color=fitness_values,
                colorscale='Viridis',
                opacity=0.8,
                symbol=marker_symbol,
                colorbar=dict(title='Fitness') if i == 0 else None,
                showscale=(i == 0),
                line=dict(width=1, color='black')
            ),
            hovertemplate=f'<b>Generation</b>: {gen}<br>' +
                         f'<b>{param_names[param_x].capitalize()}</b>: %{{x:.2f}}<br>' +
                         f'<b>{param_names[param_y].capitalize()}</b>: %{{y:.2f}}<br>' +
                         f'<b>{param_names[param_z].capitalize()}</b>: %{{z:.2f}}<br>' +
                         f'<b>Fitness</b>: %{{marker.color:.2f}}<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title=f'Interactive Population Evolution: {param_names[param_x]} vs {param_names[param_y]} vs {param_names[param_z]}',
        scene=dict(
            xaxis_title=param_names[param_x].capitalize(),
            yaxis_title=param_names[param_y].capitalize(),
            zaxis_title=param_names[param_z].capitalize(),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        width=800,
        height=600,
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig









