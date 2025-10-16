import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from utils.hill_climb.functions import quadratic, sinusoidal, ackley, rosenbrock, rastrigin
from utils.hill_climb.algorithms import simple_hill_climbing, adaptive_hill_climbing

st.set_page_config(page_title="AT01: Hill Climbing")
st.title("Assignment 01: Hill Climbing Algorithms")

# ---------

st.markdown("# Introduction")
st.markdown("In this assignment, we implement and compare hill climbing algorithms to solve optimization problems.")

st.markdown("""
**Hill Climbing** is a local search optimization algorithm that iteratively moves to better 
neighboring solutions. The algorithm continues until it reaches a local optimum where no 
neighboring solution offers improvement.

We implement two variants:

1. **Simple Hill Climbing (SHC)**: Accepts the first improving neighbor found, making it 
   fast and efficient for simple landscapes.

2. **Adaptive Hill Climbing (AHC)**: Dynamically adjusts the search step size based on 
   success or failure, allowing it to explore broadly with large steps and refine locally 
   with small steps. This makes it more robust across different problem types.
""")

cont = st.container(border=True)
with cont:
    st.badge("Simple Hill Climbing", color="blue")
    st.markdown("""
    **Strengths**
    - Minimal evaluations per step
    - Fast execution
    - Easy to implement and understand
    - Works well on smooth, convex landscapes

    **Weaknesses**
    - Fixed step size may be suboptimal
    - Easily trapped in local optima
    - No mechanism to escape poor regions
    - Sensitive to initial step size choice
    """)
    
    st.badge("Adaptive Hill Climbing", color="green")
    st.markdown("""
    **Strengths**
    - Dynamically adjusts step size for better convergence
    - Can escape shallow local optima through larger steps
    - Self-tuning mechanism adapts to landscape
    - Better exploration-exploitation balance

    **Weaknesses**
    - More iterations needed for fine-tuning
    - Additional complexity in step size management
    - May oscillate around optimal solution
    - Still susceptible to deep local optima
    """)

# ---------

st.markdown("# Methods")

st.markdown("""
For implementing Hill Climbing algorithms, we need to define several key parameters:

**Common Parameters:**
- **`fitness_function`**: The objective function to maximize/minimize
- **`bounds`**: The lower and upper limits of the search space
- **`step_size`**: How far to move when exploring neighbors
- **`max_iterations`**: Maximum number of steps before stopping
- **`initial_point`**: Starting position (optional, random if not specified)
""")

cont = st.container(border=True)
with cont:
    st.badge("Simple Hill Climbing Algorithm:", color="blue")
    st.markdown("""
    1. Start at an initial point (random or specified)
    2. Evaluate current position's fitness
    3. Generate neighboring solutions (e.g., left and right by step_size)
    4. Check each neighbor in **fixed order**
    5. **Move to the FIRST neighbor that improves fitness**
    6. Repeat until no improving neighbor is found (local optimum)
    7. Return the best solution found
    
    **Characteristics:** Simple, fast per iteration, but easily trapped in local optima.
    """)
    
    st.badge("Adaptive Hill Climbing Algorithm:", color="green")
    st.markdown("""
    1. Start at an initial point with an initial step size
    2. Evaluate current position's fitness
    3. Generate neighboring solutions using **current adaptive step size**
    4. Evaluate all neighbors and find the best one
    5. **If improvement found:**
       - Move to better neighbor
       - **INCREASE step size** (accelerate exploration)
    6. **If no improvement:**
       - Stay at current position
       - **DECREASE step size** (refine local search)
    7. Repeat until step size becomes very small (converged)
    8. Return the best solution found
    
    **Key Difference:** Adaptive HC dynamically adjusts the step size based on search success,
    allowing it to explore quickly with large steps and refine locally with small steps.
    This can escape shallow local optima and converge more precisely.
    """)

# ---------

st.markdown("# Results")

col1, col2 = st.columns(spec=[0.3, 0.7])

with col1:
    # Set up a control panel
    st.subheader("Controls")
    
    # Dimension selector and plot type selector side by side
    col_dim, col_plot = st.columns([1, 1])
    with col_dim:
        dimensions = st.radio("Problem Dimensions", [1, 2], index=0)
    with col_plot:
        if dimensions == 2:
            plot_type = st.radio("Plot Type", ["Depth Plot", "3D Interactive Plot"], index=0)
        else:
            plot_type = None

    # Function selection based on dimensions
    if dimensions == 1:
        function_options = ["Quadratic", "Sinusoidal", "Ackley", "Rosenbrock", "Rastrigin"]
        st.caption("All functions available in 1D")
    else:
        function_options = ["Ackley", "Rosenbrock", "Rastrigin"]
        st.caption("Only complex benchmark functions available in 2D")
    
    function_choice = st.selectbox("Objective Function", function_options)
    
    if dimensions == 1:
        neighborhood = "N/A"
    else:
        neighborhood = st.selectbox("Neighbourhood Shape", ["square", "cross"], index=0)
    
    # Set recommended bounds based on function
    if function_choice == "Rosenbrock":
        default_lower = -2.0
        default_upper = 2.0
    elif function_choice == "Rastrigin":
        default_lower = -5.0
        default_upper = 5.0
    elif function_choice == "Ackley":
        default_lower = -32.0
        default_upper = 32.0
    else:
        default_lower = -5.0
        default_upper = 5.0
    
    comparison_mode = st.checkbox("Compare Algorithms", value=False)
    
    if not comparison_mode:
        algorithm_type = st.selectbox("Algorithm", 
            ["Simple Hill Climbing", "Adaptive HC"])
    else:
        # In comparison mode, allow selecting two algorithms
        st.markdown("**Select algorithms to compare:**")
        algo1 = st.selectbox("Algorithm 1", 
            ["Simple Hill Climbing", "Adaptive HC"],
            index=0)
        algo2 = st.selectbox("Algorithm 2", 
            ["Simple Hill Climbing", "Adaptive HC"],
            index=1)
        algorithm_type = "Comparison"
    
    
    # Optimization mode selector
    optimize_mode = st.radio("Optimization Goal", 
        ["Minimize (find lowest value)", "Maximize (find highest value)"],
        index=0)
    minimize = optimize_mode.startswith("Minimize")
    
    iterations = st.slider("Max Iterations", 1, 100, step=1, value=50)
    step_size = st.slider("Step Size", 0.1, 5.0, value=0.5, step=0.1)
    lower_bound = st.number_input("Lower Bound", -50.0, 0.0, default_lower, step=0.5)
    upper_bound = st.number_input("Upper Bound", 0.0, 50.0, default_upper, step=0.5)
    
    use_random_start = st.checkbox("Random Start", value=True)
    if not use_random_start:
        if dimensions == 1:
            initial_point = st.number_input("Initial Point", lower_bound, upper_bound, 
                value=(lower_bound + upper_bound) / 2)
        else:  # 2D
            st.markdown("**Initial Point:**")
            col_x, col_y = st.columns(2)
            with col_x:
                x_init = st.number_input("X coordinate", lower_bound, upper_bound, 
                    value=(lower_bound + upper_bound) / 2, key="x_init")
            with col_y:
                y_init = st.number_input("Y coordinate", lower_bound, upper_bound, 
                    value=(lower_bound + upper_bound) / 2, key="y_init")
            initial_point = [x_init, y_init]
    else:
        initial_point = None

    # Map function choice
    match function_choice:
        case "Quadratic":
            fitness_function = quadratic
        case "Sinusoidal":
            fitness_function = sinusoidal
        case "Ackley":
            fitness_function = ackley
        case "Rosenbrock":
            fitness_function = rosenbrock
        case "Rastrigin":
            fitness_function = rastrigin
        case _:
            fitness_function = quadratic

    # Run the selected algorithm
    start_point = None if use_random_start else initial_point
    
    # Helper function to get algorithm function by name
    def get_algorithm_func(algo_name):
        if algo_name == "Simple Hill Climbing":
            return simple_hill_climbing
        else:
            return adaptive_hill_climbing
    
    if comparison_mode:
        # Run two selected algorithms with the SAME starting point for fair comparison
        if use_random_start:
            if dimensions == 1:
                start_point = np.random.uniform(lower_bound, upper_bound)
            else:  # 2D
                start_point = np.array([
                    np.random.uniform(lower_bound, upper_bound),
                    np.random.uniform(lower_bound, upper_bound)
                ])
        
        # Get algorithm functions
        algo_func1 = get_algorithm_func(algo1)
        algo_func2 = get_algorithm_func(algo2)
        
        # Run Algorithm 1
        best_algo1, fitness_algo1, history_algo1 = algo_func1(
            fitness_function,
            bounds=(lower_bound, upper_bound),
            step_size=step_size,
            max_iterations=iterations,
            initial_point=start_point,
            minimize=minimize,
            dimensions=dimensions,
            neighborhood=neighborhood,
        )
        
        # Run Algorithm 2 with SAME starting point
        best_algo2, fitness_algo2, history_algo2 = algo_func2(
            fitness_function, 
            bounds=(lower_bound, upper_bound),
            step_size=step_size,
            max_iterations=iterations,
            initial_point=start_point,
            minimize=minimize,
            dimensions=dimensions,
            neighborhood=neighborhood,
        )
        
        # Store both results for comparison
        results_comparison = {
            'algo1': (best_algo1, fitness_algo1, history_algo1, algo1),
            'algo2': (best_algo2, fitness_algo2, history_algo2, algo2),
            'start_point': start_point
        }
        
    else:
        # Single algorithm mode
        algo_func = get_algorithm_func(algorithm_type)
        best_candidate, best_fitness, history = algo_func(
            fitness_function, 
            bounds=(lower_bound, upper_bound),
            step_size=step_size,
            max_iterations=iterations,
            initial_point=start_point,
            minimize=minimize,
            dimensions=dimensions,
            neighborhood=neighborhood,
        )
    
    # In single mode, results will be shown below plots in col2 (not in this column)

with col2:
    # plot_type is now set in col1 next to dimensions selector
    # Extract comparison results variables for plotting
    if comparison_mode:
        best_algo1, fitness_algo1, history_algo1, name_algo1 = results_comparison['algo1']
        best_algo2, fitness_algo2, history_algo2, name_algo2 = results_comparison['algo2']
        # Plot everything
        st.subheader(f"Algorithm Comparison on {function_choice}")
        if dimensions == 1:
            # 1D Visualization: show both algorithms' search paths and convergence
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            candidate_values = np.linspace(lower_bound, upper_bound, 400)
            fitness_values = [fitness_function(x) for x in candidate_values]
            candidate_history_algo1, fitness_history_algo1 = zip(*history_algo1) if history_algo1 else ([], [])
            candidate_history_algo2, fitness_history_algo2 = zip(*history_algo2) if history_algo2 else ([], [])
            # Plot fitness curve and both search paths
            ax1.plot(candidate_values, fitness_values, 'k-', linewidth=2, alpha=0.5, label='Objective Function')
            if len(candidate_history_algo1) > 0:
                ax1.plot(candidate_history_algo1, [fitness_function(x) for x in candidate_history_algo1],
                         'ro-', markersize=4, linewidth=2, label=name_algo1)
            if len(candidate_history_algo2) > 0:
                ax1.plot(candidate_history_algo2, [fitness_function(x) for x in candidate_history_algo2],
                         'go-', markersize=4, linewidth=2, label=name_algo2)
            ax1.set_xlabel('x')
            ax1.set_ylabel('f(x)')
            ax1.set_title('Search Paths')
            ax1.legend()
            # Plot convergence for both algorithms
            ax2.plot(range(len(fitness_history_algo1)), fitness_history_algo1, 'r-o', label=f'{name_algo1} Fitness')
            ax2.plot(range(len(fitness_history_algo2)), fitness_history_algo2, 'g-s', label=f'{name_algo2} Fitness')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Fitness')
            ax2.set_title('Convergence')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        elif dimensions == 2:
            x = np.linspace(lower_bound, upper_bound, 100)
            y = np.linspace(lower_bound, upper_bound, 100)
            X, Y = np.meshgrid(x, y)
            Z = np.array([[fitness_function([xi, yi]) for xi, yi in zip(x_row, y_row)] 
                         for x_row, y_row in zip(X, Y)])
            candidate_history_algo1, fitness_history_algo1 = zip(*history_algo1) if history_algo1 else ([], [])
            candidate_history_algo2, fitness_history_algo2 = zip(*history_algo2) if history_algo2 else ([], [])
            if plot_type == "Depth Plot":
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                contour1 = ax1.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.8)
                ax1.contour(X, Y, Z, levels=10, colors='black', alpha=0.2, linewidths=0.5)
                if len(candidate_history_algo1) > 0:
                    path1 = np.array(candidate_history_algo1)
                    ax1.plot(path1[:,0], path1[:,1], 'ro-', markersize=4, linewidth=2, label=name_algo1)
                ax1.set_xlabel('x')
                ax1.set_ylabel('y')
                ax1.set_title(f"{name_algo1} Search Path")
                ax1.legend()
                contour2 = ax2.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.8)
                ax2.contour(X, Y, Z, levels=10, colors='black', alpha=0.2, linewidths=0.5)
                if len(candidate_history_algo2) > 0:
                    path2 = np.array(candidate_history_algo2)
                    ax2.plot(path2[:,0], path2[:,1], 'go-', markersize=4, linewidth=2, label=name_algo2)
                ax2.set_xlabel('x')
                ax2.set_ylabel('y')
                ax2.set_title(f"{name_algo2} Search Path")
                ax2.legend()
                plt.tight_layout()
                st.pyplot(fig)
            elif plot_type == "3D Interactive Plot":
                fig3d = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.8)])
                if len(candidate_history_algo1) > 0:
                    path1 = np.array(candidate_history_algo1)
                    fig3d.add_trace(go.Scatter3d(x=path1[:,0], y=path1[:,1], z=[fitness_function(p) for p in path1],
                        mode='lines+markers', name=name_algo1, line=dict(color='blue', width=4), marker=dict(size=4)))
                if len(candidate_history_algo2) > 0:
                    path2 = np.array(candidate_history_algo2)
                    fig3d.add_trace(go.Scatter3d(x=path2[:,0], y=path2[:,1], z=[fitness_function(p) for p in path2],
                        mode='lines+markers', name=name_algo2, line=dict(color='green', width=4), marker=dict(size=4)))
                fig3d.update_layout(title="3D Interactive Plot", scene=dict(
                    xaxis_title='x', yaxis_title='y', zaxis_title='f(x, y)'))
                st.plotly_chart(fig3d, use_container_width=True)
            if len(fitness_history_algo1) > 0 or len(fitness_history_algo2) > 0:
                fig2, ax2 = plt.subplots(figsize=(10, 4))
                if len(fitness_history_algo1) > 0:
                    ax2.plot(range(len(fitness_history_algo1)), fitness_history_algo1, 
                        marker='o', color='blue', linewidth=2, label=name_algo1, alpha=0.7)
                if len(fitness_history_algo2) > 0:
                    ax2.plot(range(len(fitness_history_algo2)), fitness_history_algo2, 
                        marker='s', color='red', linewidth=2, label=name_algo2, alpha=0.7)
                ax2.set_xlabel("Iteration", fontsize=12)
                ax2.set_ylabel("Fitness", fontsize=12)
                ax2.set_title("Convergence Comparison", fontsize=14, fontweight='bold')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                st.pyplot(fig2)

        # Show Comparison Results full-width below the plots
        goal_text = "Minimum" if minimize else "Maximum"
        st.markdown("---")
        st.subheader(f"Comparison Results ({goal_text} Search)")
        best_algo1, fitness_algo1, history_algo1, name_algo1 = results_comparison['algo1']
        best_algo2, fitness_algo2, history_algo2, name_algo2 = results_comparison['algo2']
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**{name_algo1}**")
            st.metric("Iterations", len(history_algo1))
            st.metric("Evaluations", len(history_algo1) * (4 if dimensions == 2 else 2))
            st.metric(f"Best {goal_text}", f"{fitness_algo1:.4f}")
            if dimensions == 1:
                st.metric("Best Position", f"{best_algo1:.4f}")
            else:
                st.metric("Best Position", f"({best_algo1[0]:.4f}, {best_algo1[1]:.4f})")
        with col_b:
            st.markdown(f"**{name_algo2}**")
            st.metric("Iterations", len(history_algo2))
            st.metric("Evaluations", len(history_algo2) * (4 if dimensions == 2 else 2))
            st.metric(f"Best {goal_text}", f"{fitness_algo2:.4f}")
            if dimensions == 1:
                st.metric("Best Position", f"{best_algo2:.4f}")
            else:
                st.metric("Best Position", f"({best_algo2[0]:.4f}, {best_algo2[1]:.4f})")
        # Highlight differences
        if dimensions == 1:
            position_diff = abs(best_algo1 - best_algo2)
            if position_diff < 0.001:
                st.success(f"Both algorithms found the **same local optimum** at x ≈ {best_algo1:.4f}")
            else:
                st.warning(f"Algorithms found **different local optima**!\n\n"
                          f"- {name_algo1}: x = {best_algo1:.4f} (f = {fitness_algo1:.4f})\n"
                          f"- {name_algo2}: x = {best_algo2:.4f} (f = {fitness_algo2:.4f})")
        else:  # 2D
            position_diff = np.linalg.norm(best_algo1 - best_algo2)
            if position_diff < 0.001:
                st.success(f"Both algorithms found the **same local optimum** at ({best_algo1[0]:.4f}, {best_algo1[1]:.4f})")
            else:
                st.warning(f"Algorithms found **different local optima**!\n\n"
                          f"- {name_algo1}: ({best_algo1[0]:.4f}, {best_algo1[1]:.4f}) → f = {fitness_algo1:.4f}\n"
                          f"- {name_algo2}: ({best_algo2[0]:.4f}, {best_algo2[1]:.4f}) → f = {fitness_algo2:.4f}")
    else:
        st.subheader(f"{algorithm_type} on {function_choice}")
        if dimensions == 1:
            # 1D plotting: show fitness curve, search path, and convergence
            candidate_values = np.linspace(lower_bound, upper_bound, 400)
            fitness_values = [fitness_function(x) for x in candidate_values]
            candidate_history, fitness_history = zip(*history) if history else ([], [])
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            # Plot fitness curve and search path
            ax1.plot(candidate_values, fitness_values, 'k-', linewidth=2, alpha=0.5, label='Objective Function')
            if len(candidate_history) > 0:
                ax1.plot(candidate_history, [fitness_function(x) for x in candidate_history],
                         'ro-', markersize=4, linewidth=2, label='Search Path')
            ax1.set_xlabel('x')
            ax1.set_ylabel('f(x)')
            ax1.set_title('Search Path')
            ax1.legend()
            # Plot convergence
            ax2.plot(range(len(fitness_history)), fitness_history, 'b-o', label='Fitness')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Fitness')
            ax2.set_title('Convergence')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        if dimensions == 2:
            x = np.linspace(lower_bound, upper_bound, 100)
            y = np.linspace(lower_bound, upper_bound, 100)
            X, Y = np.meshgrid(x, y)
            Z = np.array([[fitness_function([xi, yi]) for xi, yi in zip(x_row, y_row)] 
                         for x_row, y_row in zip(X, Y)])
            candidate_history, fitness_history = zip(*history) if history else ([], [])
            if plot_type == "Depth Plot":
                fig, ax = plt.subplots(figsize=(8, 6))
                contour = ax.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.8)
                ax.contour(X, Y, Z, levels=10, colors='black', alpha=0.2, linewidths=0.5)
                if len(candidate_history) > 0:
                    path = np.array(candidate_history)
                    ax.plot(path[:,0], path[:,1], 'ro-', markersize=4, linewidth=2, label='Search Path')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_title(f"{algorithm_type} Search Path")
                ax.legend()
                st.pyplot(fig)
            elif plot_type == "3D Interactive Plot":
                fig3d = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.8)])
                if len(candidate_history) > 0:
                    path = np.array(candidate_history)
                    fig3d.add_trace(go.Scatter3d(x=path[:,0], y=path[:,1], z=[fitness_function(p) for p in path],
                        mode='lines+markers', name='Search Path', line=dict(color='red', width=4), marker=dict(size=4)))
                fig3d.update_layout(title="3D Interactive Plot", scene=dict(
                    xaxis_title='x', yaxis_title='y', zaxis_title='f(x, y)'))
                st.plotly_chart(fig3d, use_container_width=True)

            # Always show convergence plot below main plot for 2D
            if len(fitness_history) > 0:
                fig2, ax2 = plt.subplots(figsize=(10, 3))
                ax2.plot(range(len(fitness_history)), fitness_history, 
                    marker='o', color='blue', linewidth=2)
                ax2.set_xlabel("Iteration", fontsize=12)
                ax2.set_ylabel("Fitness", fontsize=12)
                ax2.set_title("Fitness vs Iteration (Convergence)", fontsize=14, fontweight='bold')
                ax2.grid(True, alpha=0.3)
                st.pyplot(fig2)
        # Results full-width below plots (single mode)
        goal_text = "Minimum" if minimize else "Maximum"
        st.markdown("---")
        st.subheader(f"Results ({goal_text} Search)")
        cols = st.columns(3)
        with cols[0]:
            st.metric(f"Best {goal_text}", f"{best_fitness:.4f}")
        with cols[1]:
            if dimensions == 1:
                st.metric("Best Position", f"{best_candidate:.4f}")
            else:
                st.metric("Best Position", f"({best_candidate[0]:.4f}, {best_candidate[1]:.4f})")
        with cols[2]:
            st.metric("Iterations", len(history))
            st.caption(f"Evaluations: {len(history) * (4 if dimensions == 2 else 2)}")

# ---------

st.markdown("# Discussion")

cont = st.container(border=True)
with cont:
    st.badge("Key Conclusions", color="blue")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Simple Hill Climbing**")
        st.markdown("""
        - **Fast decisions** - accepts first improvement
        - **Fixed step size** throughout search
        - **Easily trapped** in local optima
        - Best for quick exploration with limited budget
        """)
    
    with col2:
        st.markdown("**Adaptive Hill Climbing**")
        st.markdown("""
        - **Dynamic step adjustment** based on progress
        - **Large steps** for exploration and escape
        - **Small steps** for precise convergence
        - Best for higher quality solutions
        """)
    
    st.markdown("---")
    
    st.markdown("**Algorithm Performance by Function:**")
    st.markdown("""
    - **Quadratic**: Both find single global optimum efficiently
    - **Sinusoidal**: Both converge to one of multiple optima
    - **Ackley/Rastrigin**: Adaptive HC shows advantages on multimodal landscapes
    - **Rosenbrock**: Narrow valley requires careful step sizing
    """)
    
    st.markdown("---")
    
    st.markdown("**Main Challenge: Local Optimum Problem**")
    st.markdown("""
    - Getting trapped at local peaks (not global optimum)
    - Strong dependence on initial starting point
    - Difficulty escaping once trapped (especially Simple HC)
    - Plateaus where fitness doesn't change
    
    **Adaptive HC partially addresses this** by dynamically adjusting step size to escape shallow traps.
    """)
