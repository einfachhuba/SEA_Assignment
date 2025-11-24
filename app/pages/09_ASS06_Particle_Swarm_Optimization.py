import streamlit as st
import time
from sklearn.model_selection import train_test_split
from utils.general.pdf_viewer import display_pdf_with_controls
from utils.general.ui import spacer
from utils.particle_swarm_optimization.ui import (
    display_parameter_controls,
    display_model_controls,
    display_results,
)
from utils.particle_swarm_optimization.algorithms import PSO_FeatureSelection
from utils.particle_swarm_optimization.functions import load_data

st.set_page_config(page_title="AT06: Particle Swarm Optimization")

tab1, tab2 = st.tabs(["📄 Assignment Paper", "🔧 PSO Implementation"])

with tab1:
    st.title("Assignment 06: Particle Swarm Optimization")
    pdf_path = "Assignment_Sheets/06/SEA_Exercise06_Particle_Swarm_Optimization.pdf"
    try:
        display_pdf_with_controls(pdf_path)
    except Exception:
        st.info("Assignment PDF not found. Proceed to the implementation tab.")

with tab2:
    st.title("Assignment 06: Particle Swarm Optimization Implementation")
    
    st.header("Introduction")
    st.markdown("""
    **Particle Swarm Optimization (PSO)** is a population-based metaheuristic algorithm inspired by the social behavior of bird flocking or fish schooling. 
    In PSO, a group of candidate solutions (particles) move through the search space. Each particle adjusts its position based on its own best known position 
    and the best known position of the entire swarm. In this assignment, we apply PSO to **Feature Selection** on the **Covertype** dataset. The goal is to select a subset of features that maximizes classification accuracy 
    while minimizing the number of selected features.
    """)

    spacer(6)

    cont = st.container(border=True)
    with cont:
        col1, col2 = st.columns(2)
        with col1:
            st.badge("**Key Features:**", color="blue")
            st.markdown("""
            - **Population-based**: Multiple particles search simultaneously
            - **Social Learning**: Particles learn from the global best
            - **Cognitive Learning**: Particles remember their personal best
            - **Continuous with Thresholding**: Position in [0, 1], selected if > 0.5
            """)

        with col2:
            st.badge("**Problem Characteristics:**", color="blue")
            st.markdown("""
            - **Dataset**: Covertype (Forest Cover Type Prediction)
            - **Objective**: Maximizing fitness with $\\alpha \cdot acc + (1-\\alpha) \cdot (1 - n/N)$
            - **Classifier**: Random Forest (robust baseline)
            """)

    spacer(12)
    
    st.subheader("Dataset Preview")
    st.markdown("""
    The **Covertype** dataset contains tree observations from four areas of the Roosevelt National Forest in Colorado.
    All observations are cartographic variables from 30 meter x 30 meter sections of forest.
    The target variable is the **Cover_Type**.
    """)
    
    # Load data for preview
    with st.spinner("Loading dataset preview..."):
        _, _, _, display_df = load_data(sample_size=5000)
    
    st.dataframe(display_df, width='stretch')

    spacer(24)

    st.header("Methods")
    
    st.markdown("""
    ### Particle Swarm Optimization Algorithm
    
    The PSO algorithm for Feature Selection follows these steps:
    
    1. **Initialization**: 
       - Initialize a swarm of particles with random positions $x$ and velocities $v$
       - Position $x_i \in [0, 1]$ represents the probability/score of selecting feature $i$
    
    2. **Evaluation**:
        - Threshold position: If $x_i > 0.5$, feature $i$ is selected
           - Train a classifier (Random Forest) on selected features
           - Calculate Fitness: $F = \\alpha \\cdot acc(\\mathbf{x}) + (1-\\alpha) \\cdot \\left(1 - \\frac{N_{selected}}{N_{total}}\\right)$       
    
     3. **Update** (for each particle):
        - Update Velocity: $v_{id}(t+1) = w \cdot v_{id}(t) + c_1 r_1 (pbest_{id}-x_{id}) + c_2 r_2 (nbest_{id}-x_{id}) + c_g r_3 (gbest_{d}-x_{id})$ where $nbest$ is the best among the $k$ nearest neighbors
        - Update Position: $x_{id}(t+1) = x_{id}(t) + v_{id}(t+1)$
        - Clamp position to $[0, 1]$
    
    4. **Termination**: Repeat steps 2-3 until max iterations reached
    
    ### Key Parameters
    
    - **Inertia Weight ($w$)**: Controls the impact of the previous velocity. High $w$ promotes exploration, low $w$ promotes exploitation
    - **Cognitive Coefficient ($c_1$)**: Importance of the particle's own experience (personal best)
    - **Social Coefficient ($c_2$)**: Attraction toward the best neighbor
    - **Global Coefficient ($c_g$)**: Keeps everyone nudged toward the global best
    - **Penalty Factor ($\\alpha$)**: Controls the trade-off between accuracy and feature reduction
    """)

    spacer(24)

    st.header("Configuration & Execution")
    
    st.markdown("""
    Configure the PSO parameters and run the feature selection process.
    The algorithm will use a subsample of the Covertype dataset for performance.
    """)

    spacer(12)

    # Parameters
    params = display_parameter_controls()

    spacer(8)

    st.subheader("Model Parameters")
    st.markdown("Tune the Random Forest evaluator to trade accuracy against runtime.")
    model_params = display_model_controls()

    spacer(12)

    if st.button("Run Particle Swarm Optimization", type="primary", width='stretch'):
        
        # Load Data
        with st.spinner("Loading and preparing data (100k rows)..."):
            X, y, feature_names, _ = load_data(sample_size=100000)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Initialize solver
        pso = PSO_FeatureSelection(
            num_particles=params["num_particles"],
            num_iterations=params["max_iterations"],
            w=params["w"],
            c1=params["c1"],
            c2=params["c2"],
            c_global=params["c_global"],
            alpha=params["alpha"],
            neighbor_count=params["neighbor_count"],
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            model_params=model_params,
        )

        # Create visualization placeholders
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        
        def on_progress(iteration: int, best_fitness: float, best_accuracy: float):
            progress_bar.progress(min(1.0, iteration / params["max_iterations"]))
            status_text.text(f"Iteration {iteration}/{params['max_iterations']} | Best Fitness: {best_fitness:.4f} | Accuracy: {best_accuracy:.4f}")

        with st.spinner("Running particle swarm optimization..."):
            results = pso.run(progress_callback=on_progress)

        progress_bar.empty()
        status_text.empty()

        spacer(12)
        
        # Store results in session state
        st.session_state['pso_results'] = results
        st.session_state['pso_feature_names'] = feature_names
        st.session_state['pso_params'] = params
        st.session_state['pso_model_params'] = model_params
    
    spacer(24)
    
    # Display results if they exist
    if 'pso_results' in st.session_state:
        results = st.session_state['pso_results']
        feature_names = st.session_state['pso_feature_names']
        model_params = st.session_state.get('pso_model_params', {})
        
        display_results(
            history=results["history"],
            best_fitness=results["best_fitness"],
            best_accuracy=results["best_accuracy"],
            best_features=results["best_features"],
            feature_names=feature_names,
            best_position=results["best_position"]
        )
        
        spacer(24)
        
        st.header("Discussion")
        
        st.markdown("""
        ### Algorithm Behavior
        
        **Pro and Cons:**
        - **Pros:**
          - Simple to understand
          - Effective at exploring large search spaces
        - **Cons:**
          - May converge prematurely to local optima
          - **Very performance sensitive** to parameter settings
          - Computationally intensive due to multiple evaluations per iteration

        **Convergence:**
        - The convergence plot shows how the fitness and accuracy improve over iterations
        - PSO typically converges quickly in early iterations as particles are pulled towards the global best

        **Feature Selection:**
        - Fitness combines accuracy and subset size: $\\alpha \cdot acc + (1-\\alpha) \cdot (1 - n/N)$
        - Higher $\\alpha$ focuses on accuracy, often retaining more features to preserve predictive power
        - Lower $\\alpha$ rewards compact subsets, even if their accuracy degrades
        
        **Parameter Influence:**
        - **Inertia ($w$)**: A higher inertia allows particles to maintain their trajectory, supporting exploration. A lower inertia helps fine-tune the solution (exploitation)
        - **Coefficients ($c_1, c_2, c_g$)**: Balancing these controls whether particles follow their own path or the swarm's path
        - **Penalty factor ($\\alpha$)**: Use values close to 1 when accuracy is the priority, and smaller values when aggressive feature reduction is desired
        """)
