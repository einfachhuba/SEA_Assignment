import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
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

col1, col2 = st.columns(spec=[0.25 ,0.75])

with col1:
    # Set up a control panel
    st.subheader("Controls")
    
    # Dimension selector
    dimensions = st.radio("Problem Dimensions", 
        [1, 2], 
        index=0)
    
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
    
    # Display metrics
    st.markdown("---")
    
    if comparison_mode:
        goal_text = "Minimum" if minimize else "Maximum"
        st.subheader(f"Comparison Results ({goal_text} Search)")
        
        # Extract data for both algorithms
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
                st.success(f"✅ Both algorithms found the **same local optimum** at x ≈ {best_algo1:.4f}")
            else:
                st.warning(f"Algorithms found **different local optima**!\n\n"
                          f"- {name_algo1}: x = {best_algo1:.4f} (f = {fitness_algo1:.4f})\n"
                          f"- {name_algo2}: x = {best_algo2:.4f} (f = {fitness_algo2:.4f})")
        else:  # 2D
            position_diff = np.linalg.norm(best_algo1 - best_algo2)
            if position_diff < 0.001:
                st.success(f"✅ Both algorithms found the **same local optimum** at ({best_algo1[0]:.4f}, {best_algo1[1]:.4f})")
            else:
                st.warning(f"Algorithms found **different local optima**!\n\n"
                          f"- {name_algo1}: ({best_algo1[0]:.4f}, {best_algo1[1]:.4f}) → f = {fitness_algo1:.4f}\n"
                          f"- {name_algo2}: ({best_algo2[0]:.4f}, {best_algo2[1]:.4f}) → f = {fitness_algo2:.4f}")
        
    else:
        # Single algorithm metrics
        goal_text = "Minimum" if minimize else "Maximum"
        if history:
            candidate_history, fitness_history = zip(*history)
        else:
            candidate_history, fitness_history = [], []
        
        st.metric(f"Best {goal_text}", f"{best_fitness:.4f}")
        if dimensions == 1:
            st.metric("Best Position", f"{best_candidate:.4f}")
        else:
            st.metric("Best Position", f"({best_candidate[0]:.4f}, {best_candidate[1]:.4f})")
        st.metric("Iterations Used", len(history))
        st.metric("Function Evaluations", len(history) * (4 if dimensions == 2 else 2))

with col2:
    # Plot everything
    if comparison_mode:
        st.subheader(f"Algorithm Comparison on {function_choice}")
        
        if dimensions == 1:
            # 1D Visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Plot objective function on both
            candidate_values = np.linspace(lower_bound, upper_bound, 400)
            fitness_values = [fitness_function(x) for x in candidate_values]
            
            # Extract histories
            candidate_history_algo1, fitness_history_algo1 = zip(*history_algo1) if history_algo1 else ([], [])
            candidate_history_algo2, fitness_history_algo2 = zip(*history_algo2) if history_algo2 else ([], [])
            
            # Algorithm 1 plot
            ax1.plot(candidate_values, fitness_values, 'k-', linewidth=2, alpha=0.5, label='Objective Function')
            
            if len(candidate_history_algo1) > 0:
                colors_algo1 = np.linspace(0, 1, len(candidate_history_algo1))
                ax1.scatter(candidate_history_algo1, fitness_history_algo1, 
                    c=colors_algo1, cmap='Blues', s=80, edgecolors='black', linewidth=1,
                    label='Search Path', zorder=5)
                
                # Draw path
                ax1.plot(candidate_history_algo1, fitness_history_algo1, 'b--', alpha=0.3, linewidth=1)
                
                # Highlight start and end
                ax1.scatter(candidate_history_algo1[0], fitness_history_algo1[0], 
                    color="green", marker='o', s=200, label="Start", zorder=10, edgecolors='black', linewidth=2)
                ax1.scatter(best_algo1, fitness_algo1, 
                    color="red", marker='*', s=400, label="Best Found", zorder=10, edgecolors='black', linewidth=2)
            
            ax1.set_xlabel("x", fontsize=11)
            ax1.set_ylabel("f(x)", fontsize=11)
            ax1.set_title(f"{name_algo1}\n({len(history_algo1)} iterations)", fontsize=12, fontweight='bold')
            ax1.legend(loc='best', fontsize=9)
            ax1.grid(True, alpha=0.3)
            
            # Algorithm 2 plot
            ax2.plot(candidate_values, fitness_values, 'k-', linewidth=2, alpha=0.5, label='Objective Function')
            
            if len(candidate_history_algo2) > 0:
                colors_algo2 = np.linspace(0, 1, len(candidate_history_algo2))
                ax2.scatter(candidate_history_algo2, fitness_history_algo2, 
                    c=colors_algo2, cmap='Reds', s=80, edgecolors='black', linewidth=1,
                    label='Search Path', zorder=5)
                
                # Draw path
                ax2.plot(candidate_history_algo2, fitness_history_algo2, 'r--', alpha=0.3, linewidth=1)
                
                # Highlight start and end
                ax2.scatter(candidate_history_algo2[0], fitness_history_algo2[0], 
                    color="green", marker='o', s=200, label="Start", zorder=10, edgecolors='black', linewidth=2)
                ax2.scatter(best_algo2, fitness_algo2, 
                    color="red", marker='*', s=400, label="Best Found", zorder=10, edgecolors='black', linewidth=2)
            
            ax2.set_xlabel("x", fontsize=11)
            ax2.set_ylabel("f(x)", fontsize=11)
            ax2.set_title(f"{name_algo2}\n({len(history_algo2)} iterations)", fontsize=12, fontweight='bold')
            ax2.legend(loc='best', fontsize=9)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Create meshgrid for contour plot
            x = np.linspace(lower_bound, upper_bound, 100)
            y = np.linspace(lower_bound, upper_bound, 100)
            X, Y = np.meshgrid(x, y)
            Z = np.array([[fitness_function([xi, yi]) for xi, yi in zip(x_row, y_row)] 
                         for x_row, y_row in zip(X, Y)])
            
            # Extract 2D paths
            candidate_history_algo1, fitness_history_algo1 = zip(*history_algo1) if history_algo1 else ([], [])
            candidate_history_algo2, fitness_history_algo2 = zip(*history_algo2) if history_algo2 else ([], [])
            
            # Algorithm 1 plot
            contour1 = ax1.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.8)
            ax1.contour(X, Y, Z, levels=10, colors='black', alpha=0.2, linewidths=0.5)
            
            if len(candidate_history_algo1) > 0:
                path1 = np.array(candidate_history_algo1)
                ax1.plot(path1[:, 0], path1[:, 1], 'b-', linewidth=2, alpha=0.7, label='Search Path')
                ax1.scatter(path1[:, 0], path1[:, 1], c=range(len(path1)), cmap='Blues', 
                           s=50, edgecolors='white', linewidth=1, zorder=5)
                ax1.scatter(path1[0, 0], path1[0, 1], color='green', marker='o', s=200, 
                           label='Start', zorder=10, edgecolors='black', linewidth=2)
                ax1.scatter(best_algo1[0], best_algo1[1], color='red', marker='*', s=400, 
                           label='Best Found', zorder=10, edgecolors='black', linewidth=2)
            
            ax1.set_xlabel("x", fontsize=11)
            ax1.set_ylabel("y", fontsize=11)
            ax1.set_title(f"{name_algo1}\n({len(history_algo1)} iterations)", fontsize=12, fontweight='bold')
            ax1.legend(loc='best', fontsize=9)
            plt.colorbar(contour1, ax=ax1, label='f(x, y)')
            
            # Algorithm 2 plot
            contour2 = ax2.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.8)
            ax2.contour(X, Y, Z, levels=10, colors='black', alpha=0.2, linewidths=0.5)
            
            if len(candidate_history_algo2) > 0:
                path2 = np.array(candidate_history_algo2)
                ax2.plot(path2[:, 0], path2[:, 1], 'r-', linewidth=2, alpha=0.7, label='Search Path')
                ax2.scatter(path2[:, 0], path2[:, 1], c=range(len(path2)), cmap='Reds', 
                           s=50, edgecolors='white', linewidth=1, zorder=5)
                ax2.scatter(path2[0, 0], path2[0, 1], color='green', marker='o', s=200, 
                           label='Start', zorder=10, edgecolors='black', linewidth=2)
                ax2.scatter(best_algo2[0], best_algo2[1], color='red', marker='*', s=400, 
                           label='Best Found', zorder=10, edgecolors='black', linewidth=2)
            
            ax2.set_xlabel("x", fontsize=11)
            ax2.set_ylabel("y", fontsize=11)
            ax2.set_title(f"{name_algo2}\n({len(history_algo2)} iterations)", fontsize=12, fontweight='bold')
            ax2.legend(loc='best', fontsize=9)
            plt.colorbar(contour2, ax=ax2, label='f(x, y)')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Convergence comparison (works for both 1D and 2D)
        _, fitness_history_algo1 = zip(*history_algo1) if history_algo1 else ([], [])
        _, fitness_history_algo2 = zip(*history_algo2) if history_algo2 else ([], [])
        
        # Show convergence plot if there's any history
        if len(fitness_history_algo1) > 0 or len(fitness_history_algo2) > 0:
            fig2, ax3 = plt.subplots(figsize=(10, 4))
            
            if len(fitness_history_algo1) > 0:
                ax3.plot(range(len(fitness_history_algo1)), fitness_history_algo1, 
                    marker='o', color='blue', linewidth=2, label=name_algo1, alpha=0.7)
            
            if len(fitness_history_algo2) > 0:
                ax3.plot(range(len(fitness_history_algo2)), fitness_history_algo2, 
                    marker='s', color='red', linewidth=2, label=name_algo2, alpha=0.7)
            
            ax3.set_xlabel("Iteration", fontsize=12)
            ax3.set_ylabel("Fitness", fontsize=12)
            ax3.set_title("Convergence Comparison", fontsize=14, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            st.pyplot(fig2)
    else:
        # Single algorithm plot
        st.subheader(f"{algorithm_type} on {function_choice}")
        
        # Extract history
        if history:
            candidate_history, fitness_history = zip(*history)
        else:
            candidate_history, fitness_history = [], []
        
        if dimensions == 1:
            # 1D Visualization
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Plot the objective function
            candidate_values = np.linspace(lower_bound, upper_bound, 400)
            fitness_values = [fitness_function(x) for x in candidate_values]
            ax.plot(candidate_values, fitness_values, label="Objective Function", 
                color='black', linewidth=2, alpha=0.7)
            
            # Plot the search trajectory
            if len(candidate_history) > 0:
                colors = np.linspace(0, 1, len(candidate_history))
                sc = ax.scatter(candidate_history, fitness_history, 
                    c=colors, cmap='Blues', s=50, edgecolors='black', linewidth=0.5,
                    label='Search Path', zorder=5)
                
                if len(candidate_history) > 1:
                    for i in range(len(candidate_history) - 1):
                        ax.annotate('', 
                            xy=(candidate_history[i+1], fitness_history[i+1]),
                            xytext=(candidate_history[i], fitness_history[i]),
                            arrowprops=dict(arrowstyle='->', color='blue', alpha=0.3, lw=1))
                
                # Highlight start and end
                ax.scatter(candidate_history[0], fitness_history[0], 
                    color="green", marker='o', s=200, label="Start", 
                    zorder=10, edgecolors='black', linewidth=2)
                ax.scatter(best_candidate, best_fitness, 
                    color="red", marker='*', s=400, label="Best Found", 
                    zorder=10, edgecolors='black', linewidth=2)
            
            ax.set_xlabel("x", fontsize=12)
            ax.set_ylabel("f(x)", fontsize=12)
            ax.set_title(f"{algorithm_type} Search Trajectory", fontsize=14, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            if len(candidate_history) > 0:
                cbar = plt.colorbar(sc, ax=ax)
                cbar.set_label("Iteration Progress", fontsize=10)
            
            st.pyplot(fig)
            
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create meshgrid for contour plot
            x = np.linspace(lower_bound, upper_bound, 100)
            y = np.linspace(lower_bound, upper_bound, 100)
            X, Y = np.meshgrid(x, y)
            Z = np.array([[fitness_function([xi, yi]) for xi, yi in zip(x_row, y_row)] 
                         for x_row, y_row in zip(X, Y)])
            
            # Plot contour
            contour = ax.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.8)
            ax.contour(X, Y, Z, levels=10, colors='black', alpha=0.2, linewidths=0.5)
            plt.colorbar(contour, ax=ax, label='f(x, y)')
            
            # Plot search trajectory
            if len(candidate_history) > 0:
                path = np.array(candidate_history)
                ax.plot(path[:, 0], path[:, 1], 'w-', linewidth=3, alpha=0.9, label='Search Path')
                ax.plot(path[:, 0], path[:, 1], 'b-', linewidth=2, alpha=0.7)
                
                colors = np.linspace(0, 1, len(path))
                ax.scatter(path[:, 0], path[:, 1], c=colors, cmap='Blues', 
                          s=50, edgecolors='white', linewidth=1, zorder=5)
                
                # Highlight start and end
                ax.scatter(path[0, 0], path[0, 1], color='green', marker='o', s=200, 
                          label='Start', zorder=10, edgecolors='black', linewidth=2)
                ax.scatter(best_candidate[0], best_candidate[1], color='red', marker='*', s=400, 
                          label='Best Found', zorder=10, edgecolors='black', linewidth=2)
            
            ax.set_xlabel("x", fontsize=12)
            ax.set_ylabel("y", fontsize=12)
            ax.set_title(f"{algorithm_type} Search Trajectory (2D)", fontsize=14, fontweight='bold')
            ax.legend(loc='best')
            
            st.pyplot(fig)
        
        # Convergence plot (works for both 1D and 2D)
        if len(fitness_history) > 0:
            fig2, ax2 = plt.subplots(figsize=(10, 3))
            ax2.plot(range(len(fitness_history)), fitness_history, 
                marker='o', color='blue', linewidth=2)
            ax2.set_xlabel("Iteration", fontsize=12)
            ax2.set_ylabel("Fitness", fontsize=12)
            ax2.set_title("Fitness vs Iteration (Convergence)", fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)

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
