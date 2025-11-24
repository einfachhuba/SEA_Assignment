import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from .config import (
    DEFAULT_NUM_PARTICLES,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_INERTIA,
    DEFAULT_C1,
    DEFAULT_C2,
    DEFAULT_GLOBAL_COEFF,
    DEFAULT_NEIGHBOR_COUNT,
    DEFAULT_PENALTY_FACTOR,
    DEFAULT_SELECTION_THRESHOLD,
    DEFAULT_RF_ESTIMATORS,
    DEFAULT_RF_MAX_DEPTH,
    DEFAULT_RF_RANDOM_STATE,
    DEFAULT_RF_WARM_START,
    DEFAULT_RF_INCREMENTAL,
    DEFAULT_RF_INCREMENTAL_STEP,
)

def display_parameter_controls():
    """
    Displays controls for PSO parameters.
    """
    with st.expander("PSO Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            num_particles = st.slider("Number of Particles", min_value=10, max_value=100, value=DEFAULT_NUM_PARTICLES, step=5)
            max_iterations = st.slider("Max Iterations", min_value=10, max_value=200, value=DEFAULT_MAX_ITERATIONS, step=10)
            inertia = st.slider("Inertia Weight (w)", min_value=0.0, max_value=1.0, value=DEFAULT_INERTIA, step=0.05, help="Controls the impact of previous velocity.")
            
        with col2:
            c1 = st.slider("Cognitive Coefficient (c1)", min_value=0.0, max_value=4.0, value=DEFAULT_C1, step=0.1, help="Importance of personal best.")
            c2 = st.slider("Neighborhood Coefficient (c2)", min_value=0.0, max_value=4.0, value=DEFAULT_C2, step=0.1, help="Influence of best neighbor.")
            c_global = st.slider("Global Coefficient (c_g/c3)", min_value=0.0, max_value=4.0, value=DEFAULT_GLOBAL_COEFF, step=0.1, help="Influence of global best (admiration).")

        neighbor_count = st.slider(
            "Neighborhood Size",
            min_value=1,
            max_value= max(2, DEFAULT_NEIGHBOR_COUNT * 4),
            value=DEFAULT_NEIGHBOR_COUNT,
            step=1,
            help="Number of nearest neighbors considered for local best.",
        )

        penalty_factor = st.slider(
            "Penalty Factor (alpha)",
            min_value=0.05,
            max_value=0.95,
            value=DEFAULT_PENALTY_FACTOR,
            step=0.01,
            help="Penalty for number of features used.",
        )
            
    return {
        "num_particles": num_particles,
        "max_iterations": max_iterations,
        "w": inertia,
        "c1": c1,
        "c2": c2,
        "c_global": c_global,
        "neighbor_count": neighbor_count,
        "alpha": penalty_factor
    }

def display_model_controls():
    """Model configuration controls for the Random Forest evaluator."""
    with st.expander("Model Configuration", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            n_estimators = st.slider(
                "Random Forest Trees",
                min_value=20,
                max_value=400,
                step=10,
                value=DEFAULT_RF_ESTIMATORS,
                help="Higher values improve stability but increase runtime.",
            )
            random_state = st.number_input(
                "Model Seed",
                min_value=0,
                max_value=10_000,
                value=DEFAULT_RF_RANDOM_STATE,
                help="Keeps the Random Forest reproducible.",
            )

        with col2:
            depth_values = list(range(4, 33))
            depth_options = ["Unlimited"] + depth_values
            default_index = 0
            if DEFAULT_RF_MAX_DEPTH in depth_values:
                default_index = depth_values.index(DEFAULT_RF_MAX_DEPTH) + 1
            max_depth_option = st.selectbox(
                "Maximum Depth",
                options=depth_options,
                index=default_index,
                help="Limit tree depth to reduce overfitting and speed up training.",
            )
            warm_start = st.checkbox(
                "Warm Start Trees",
                value=DEFAULT_RF_WARM_START,
                help="Reuse the same forest and add trees incrementally.",
            )

        incremental = st.checkbox(
            "Incremental Tree Growth",
            value=DEFAULT_RF_INCREMENTAL,
            help="Add trees in chunks to observe intermediate convergence.",
            disabled=not warm_start,
        )
        chunk = st.slider(
            "Trees per Increment",
            min_value=5,
            max_value=100,
            step=5,
            value=DEFAULT_RF_INCREMENTAL_STEP,
            disabled=not (warm_start and incremental),
        )

    return {
        "n_estimators": n_estimators,
        "max_depth": None if max_depth_option == "Unlimited" else int(max_depth_option),
        "random_state": random_state,
        "warm_start": warm_start,
        "incremental": warm_start and incremental,
        "incremental_step": chunk if warm_start and incremental else None,
    }

def display_results(history, best_fitness, best_accuracy, best_features, feature_names, best_position=None):
    """
    Displays the results of the PSO run.
    """    
    col1, col2, col3 = st.columns(3)
    col1.metric("Best Fitness", f"{best_fitness:.4f}")
    col2.metric("Best Accuracy", f"{best_accuracy:.4f}")
    col3.metric("Selected Features", f"{len(best_features)} / {len(feature_names)}")
    
    st.subheader("Convergence Plot")
    
    # Plot fitness over iterations
    iterations = range(len(history))
    fitness_values = [h['fitness'] for h in history]
    accuracy_values = [h['accuracy'] for h in history]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(iterations), y=fitness_values, mode='lines+markers', name='Best Fitness'))
    fig.add_trace(go.Scatter(x=list(iterations), y=accuracy_values, mode='lines', name='Best Accuracy', line=dict(dash='dash')))
    
    fig.update_layout(
        title="Fitness and Accuracy over Iterations",
        xaxis_title="Iteration",
        yaxis_title="Value",
        hovermode="x unified"
    )
    st.plotly_chart(fig)
    
    st.subheader("Feature Importance (Particle Position)")
    if best_position is not None:
        # Create a dataframe for plotting
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Score': best_position,
            'Selected': best_position > DEFAULT_SELECTION_THRESHOLD
        })
        # Sort by score
        importance_df = importance_df.sort_values(by='Score', ascending=False)
        
        # Plot
        fig_imp = go.Figure()
        fig_imp.add_trace(go.Bar(
            x=importance_df['Feature'],
            y=importance_df['Score'],
            marker_color=importance_df['Selected'].map({True: 'green', False: 'red'}),
            name='Feature Score'
        ))
        fig_imp.add_hline(y=DEFAULT_SELECTION_THRESHOLD, line_dash="dash", line_color="black", annotation_text="Selection Threshold")
        fig_imp.update_layout(
            title="Feature Selection Scores", 
            xaxis_title="Feature", 
            yaxis_title="Score (Probability)",
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig_imp)
    else:
        selected_names = [feature_names[i] for i in best_features]
        st.write(", ".join(selected_names))
