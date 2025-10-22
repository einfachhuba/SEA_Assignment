import streamlit as st
import plotly.graph_objects as go
import time
from utils.genetic_algorithm.algorithms import run_coffee_genetic_algorithm
from utils.genetic_algorithm.functions import (
    coffee_fitness_4d, plot_fitness_3d_interactive, plot_population_evolution_3d_interactive,
    plot_fitness_grid_interactive, plot_fitness_evolution_interactive,
    plot_population_evolution_2d_interactive, fitness_from_chromosome
)
from utils.genetic_algorithm.config import COFFEE_BOUNDS, COFFEE_PARAM_NAMES
from utils.genetic_algorithm.ui import css_blocks, spacer

css_blocks()

st.set_page_config(page_title="AT02: Genetic Algorithms")
st.title("Assignment 02: Genetic Algorithms")

# -------

st.header("Introduction")
st.markdown("""
**Genetic Algorithms (GAs)** are population-based metaheuristic optimization algorithms inspired by the process of natural selection and evolution.
Unlike hill climbing algorithms that work with a single solution, GAs maintain a population of candidate solutions that evolve over multiple generations through selection, crossover, and mutation operations.
GAs are effective for complex, multimodal optimization problems where traditional gradient-based methods may get trapped in local optima.
""")

spacer(6)

cont = st.container(border=True)
with cont:
    col1, col2 = st.columns(2)
    with col1:
        st.badge("**Strengths:**", color="blue")
        st.markdown("""
        - **Global search capability**: Population-based approach helps avoid local optima
        - **No gradient requirements**: Works with any fitness function
        - **Parallelizable**: Population members can be evaluated independently  
        - **Robust**: Performs well on noisy and discontinuous functions
        - **Flexible encoding**: Can handle mixed variable types (continuous/discrete)
        """)

    with col2:
        st.badge("**Weaknesses:**", color="blue")
        st.markdown("""
        - **Computational cost**: Requires many fitness evaluations
        - **Parameter sensitivity**: Performance depends on GA parameters
        - **No convergence guarantee**: May not find global optimum
        - **Premature convergence**: Population may lose diversity too quickly
        - **Problem-specific tuning**: Crossover/mutation operators need adjustment
        """)

spacer(24)

st.header("Methods")

st.markdown("""
For our coffee brewing optimization problem, we implement a genetic algorithm with the following key components:

**Chromosome Encoding:** Each individual represents a coffee brewing recipe as a real-valued vector `[roast, blend, grind, brew_time]` where:
- `roast` in range [0, 20] (encoded as float)
- `blend` in range [0, 100] (encoded as float)  
- `grind` in range [0, 10] (encoded as float)
- `brew_time` in range [0.0, 5.0]

This **real-valued encoding** handles the mixed integer/continuous nature elegantly by representing all parameters as floats and applying appropriate rounding/clipping in the fitness function.
""")

col1, col2 = st.columns(2)

with col1:
    st.badge("**Selection Method:**", color="blue")
    st.markdown("""
    - **Random Selection**: Randomly sample two individuals as parents for crossover until m children are generated
    """)
    
    st.badge("**Crossover Operation:**", color="blue")
    st.markdown("""
    - **Single-Point Crossover**: Split chromosomes at random point, exchange tails
    """)

with col2:
    st.badge("**Mutation Strategy:**", color="blue")
    st.markdown("""
    - **Bit-Flip Mutation**: Replace genes with completely random values within bounds
    """)
    
    st.badge("**Algorithm Features:**", color="blue")
    st.markdown("""
    - **Elitism**: Preserve best individuals across generations
    - **Diversity Tracking**: Monitor population genetic diversity
    - **Termination**: Fixed number of generations
    """)

spacer(24)

st.header("Algorithm Configuration")

col1, col2 = st.columns(2)

with col1:
    st.badge("**Population & Generations**", color="green")
    population_size = st.slider("Population Size", 20, 200, 50, step=10)
    max_generations = st.slider("Max Generations", 20, 200, 100, step=10)
    seed = st.number_input("Random Seed", value=42, step=1)

with col2:
    st.badge("**Mutation Parameters**", color="green")
    mutation_rate = st.slider("Mutation Rate", 0.01, 0.5, 0.1, step=0.01)

spacer(24)

st.header("Results")

if st.button("Run Genetic Algorithm", type="primary"):
    with st.spinner("Running genetic algorithm..."):
        start_time = time.time()
        
        # Run the genetic algorithm with assignment-specified methods
        results = run_coffee_genetic_algorithm(
            population_size=population_size,
            max_generations=max_generations,
            mutation_rate=mutation_rate,
            seed=seed
        )
        
        execution_time = time.time() - start_time
        
        # Store results in session state
        st.session_state.ga_results = results
        st.session_state.ga_execution_time = execution_time

if 'ga_results' in st.session_state:
    results = st.session_state.ga_results
    execution_time = st.session_state.ga_execution_time
    
    # Best solution summary
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Best Solution Found")
        best_params = results['best_params']
        
        st.metric("**Best Fitness**", f"{results['best_fitness']:.2f}")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Roast Level", f"{best_params['roast']:.1f}")
            st.metric("Grind Setting", f"{best_params['grind']:.1f}")
        with col_b:
            st.metric("Blend Ratio", f"{best_params['blend']:.1f}")
            st.metric("Brew Time", f"{best_params['brew_time']:.2f} min")
    
    with col2:
        st.markdown("### Algorithm Statistics")
        st.metric("Execution Time", f"{execution_time:.2f} seconds")
        st.metric("Generations", len(results['fitness_history']))
        st.metric("Total Improvement", f"{results['best_fitness'] - results['fitness_history'][0]:.2f}")
        
    st.markdown("### Evolution Visualizations")
    
    # Interactive fitness evolution plot
    fig_fitness_interactive = plot_fitness_evolution_interactive(
        results['fitness_history'], 
        results['diversity_history'],
        title="Fitness and Diversity Evolution"
    )
    st.plotly_chart(fig_fitness_interactive, use_container_width=True)
    
    # Population evolution visualization
    st.markdown("### Population Evolution in Parameter Space")
    
    # Choose between 2D and 3D population visualization
    pop_viz_type = st.radio("Population Visualization:", ["2D Evolution", "3D Interactive"], horizontal=True)
    
    if pop_viz_type == "2D Evolution":
        # Let user choose which parameters to visualize
        col1, col2, col3 = st.columns(3)
        with col1:
            param_x_idx = st.selectbox("X-axis Parameter", 
                                      options=list(range(4)),
                                      format_func=lambda x: COFFEE_PARAM_NAMES[x].capitalize(),
                                      index=0)
        with col2:
            param_y_idx = st.selectbox("Y-axis Parameter", 
                                      options=list(range(4)),
                                      format_func=lambda x: COFFEE_PARAM_NAMES[x].capitalize(),
                                      index=1)
        with col3:
            show_landscape = st.checkbox("Show Fitness Landscape", value=True)
        
        if param_x_idx != param_y_idx:
            
            fig_pop_2d = plot_population_evolution_2d_interactive(
                results['population_history'],
                fitness_from_chromosome,
                COFFEE_BOUNDS,
                param_x=param_x_idx,
                param_y=param_y_idx,
                show_landscape=show_landscape
            )
            st.plotly_chart(fig_pop_2d, use_container_width=True)
        else:
            st.warning("Please select different parameters for X and Y axes.")
    
    else:
        st.markdown("**Interactive 3D Population Evolution** - See how the population moves through 3D space!")
        
        # Let user choose which parameters to visualize in 3D
        col1, col2, col3 = st.columns(3)
        with col1:
            param_x_3d = st.selectbox("X-axis Parameter ", 
                                     options=list(range(4)),
                                     format_func=lambda x: COFFEE_PARAM_NAMES[x].capitalize(),
                                     index=0, key="x_3d")
        with col2:
            param_y_3d = st.selectbox("Y-axis Parameter ", 
                                     options=list(range(4)),
                                     format_func=lambda x: COFFEE_PARAM_NAMES[x].capitalize(),
                                     index=1, key="y_3d")
        with col3:
            param_z_3d = st.selectbox("Z-axis Parameter ", 
                                     options=list(range(4)),
                                     format_func=lambda x: COFFEE_PARAM_NAMES[x].capitalize(),
                                     index=2, key="z_3d")
        
        if len(set([param_x_3d, param_y_3d, param_z_3d])) == 3:
            fig_pop_3d = plot_population_evolution_3d_interactive(
                results['population_history'],
                fitness_from_chromosome,
                COFFEE_BOUNDS,
                param_x=param_x_3d,
                param_y=param_y_3d,
                param_z=param_z_3d
            )
            st.plotly_chart(fig_pop_3d, use_container_width=True)
            
        else:
            st.warning("Please select three different parameters for the 3D visualization.")
    
    spacer(24)

    st.markdown("### Fitness Landscape Analysis")
    
    viz_type = st.radio("Visualization Type:", ["2D Contour Plot", "3D Surface Plot"], horizontal=True)
    
    if viz_type == "2D Contour Plot":
        # Let user choose which parameters to fix
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            fix_roast = st.checkbox("Fix Roast", value=False)
            roast_value = st.slider("Roast Value", 0.0, 20.0, 
                                   best_params['roast'], step=0.5) if fix_roast else best_params['roast']
        
        with col2:
            fix_blend = st.checkbox("Fix Blend", value=False)
            blend_value = st.slider("Blend Value", 0.0, 100.0, 
                                   best_params['blend'], step=1.0) if fix_blend else best_params['blend']
        
        with col3:
            fix_grind = st.checkbox("Fix Grind", value=True)
            grind_value = st.slider("Grind Value", 0.0, 10.0, 
                                   best_params['grind'], step=0.1) if fix_grind else best_params['grind']
        
        with col4:
            fix_brew_time = st.checkbox("Fix Brew Time", value=True)
            brew_time_value = st.slider("Brew Time Value", 0.0, 5.0, 
                                       best_params['brew_time'], step=0.1) if fix_brew_time else best_params['brew_time']
        
        # Determine which dimensions to plot
        fixed_params = []
        fixed_values = []
        
        if fix_roast:
            fixed_params.append('roast')
            fixed_values.append(roast_value)
        if fix_blend:
            fixed_params.append('blend')
            fixed_values.append(blend_value)
        if fix_grind:
            fixed_params.append('grind')
            fixed_values.append(grind_value)
        if fix_brew_time:
            fixed_params.append('brew_time')
            fixed_values.append(brew_time_value)
        
        if len(fixed_params) == 2:
            fig_landscape_interactive = plot_fitness_grid_interactive(
                coffee_fitness_4d,
                fixed_dims=fixed_params,
                fixed_values=fixed_values,
                grid_points=50
            )
            
            # Add best solution point to the plot
            variable_dims = [d for d in COFFEE_PARAM_NAMES if d not in fixed_params]
            best_x = best_params[variable_dims[0]]
            best_y = best_params[variable_dims[1]]
            
            fig_landscape_interactive.add_trace(go.Scatter(
                x=[best_x],
                y=[best_y],
                mode='markers',
                marker=dict(
                    symbol='star',
                    size=20,
                    color='red',
                    line=dict(width=3, color='white')
                ),
                name=f'Best Solution (Fitness: {results["best_fitness"]:.2f})',
                hovertemplate=f'<b>Best Solution</b><br>' +
                             f'<b>{variable_dims[0].capitalize()}</b>: {best_x:.2f}<br>' +
                             f'<b>{variable_dims[1].capitalize()}</b>: {best_y:.2f}<br>' +
                             f'<b>Fitness</b>: {results["best_fitness"]:.2f}<extra></extra>'
            ))
            
            st.plotly_chart(fig_landscape_interactive, use_container_width=True)
        else:
            st.info("Please select exactly 2 parameters to fix for 2D landscape visualization.")
    
    else:
        st.markdown("**Interactive 3D Surface Visualization** - Rotate, zoom, and explore!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fixed_param_3d = st.selectbox("Parameter to Fix:", 
                                         options=COFFEE_PARAM_NAMES,
                                         index=3)
        
        with col2:
            param_idx = COFFEE_PARAM_NAMES.index(fixed_param_3d)
            min_val, max_val = COFFEE_BOUNDS[param_idx]
            default_val = best_params[fixed_param_3d]
            
            # Set appropriate step size based on parameter precision needed
            if fixed_param_3d == 'brew_time':
                step_val = 0.1
            elif fixed_param_3d in ['grind']:
                step_val = 0.1  # Fine precision for grind
            else:  # roast, blend
                step_val = 0.5  # Medium precision
            
            fixed_value_3d = st.slider(f"{fixed_param_3d.capitalize()} Value", 
                                      min_val, max_val, default_val, 
                                      step=step_val)
        
        # Generate interactive 3D plot
        fig_3d_interactive = plot_fitness_3d_interactive(
            coffee_fitness_4d,
            fixed_dim=fixed_param_3d,
            fixed_value=fixed_value_3d,
            grid_points=25,  # Reduced for performance
            best_point=best_params  # Pass the best solution parameters
        )
        
        st.plotly_chart(fig_3d_interactive, use_container_width=True)

spacer(24)

st.header("Discussion")

st.markdown("""
### Algorithm Performance Analysis

The genetic algorithm demonstrates several characteristics in solving the coffee brewing optimization problem:

**Observed Behaviour:**
- **Population-based exploration**: The GA effectively explores the 4D parameter space using multiple candidate solutions
- **Convergence behavior**: Fitness typically improves rapidly in early generations, then converges more slowly
- **Diversity dynamics**: Population diversity generally decreases over time as individuals converge toward optimal regions

""")

st.markdown("""
### Parameter Impact Analysis

**Population Size:**
- **Small**: Fast convergence, risk of local optima, lower computational cost
- **Large**: Better exploration, global search capability, higher computational cost

**Generation Count:**
- **Few**: May not converge fully, faster execution
- **Many**: Better convergence, less improvement over time, longer runtime

**Mutation Rate:**
- **Low**: Preserves good solutions, limited diversity
- **High**: High exploration, may disrupt convergence

**Key Trade-offs:**
- **Speed vs Quality**: Smaller parameters = faster but potentially lower-quality solutions
- **Exploration vs Exploitation**: Larger parameters favor exploration; smaller favor exploitation

### Limitations and Possible Improvements

**Current Limitations:**
- **Random selection** lacks selective pressure - better individuals have no advantage over worse ones
- **Fixed parameters** throughout evolution - no adaptive parameter control based on convergence state

**Possible Improvements:**
- **Tournament selection** would provide selective pressure while maintaining assignment simplicity
- **Adaptive mutation rates** that decrease over generations to balance exploration and exploitation

""")

spacer(24)

st.header("Comparison with Hill Climbing")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Genetic Algorithm Advantages:**")
    st.markdown("""
    - **Global search**: Population explores multiple regions simultaneously
    - **Robust to local optima**: Crossover can escape local minima
    - **Parallel evaluation**: Multiple solutions evaluated per generation
    """)

with col2:
    st.markdown("**Hill Climbing Advantages:**")
    st.markdown("""
    - **Computational efficiency**: Fewer function evaluations needed
    - **Simple implementation**: Straightforward algorithm logic
    - **Fast local optimization**: Quickly refines promising solutions
    """)

