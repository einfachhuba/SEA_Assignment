"""
Streamlit UI components for Neural Architecture Search visualization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List

from .config import DE_CONFIG, TRAINING_CONFIG, DATASET_CONFIG, FITNESS_WEIGHTS


def display_parameter_controls() -> Dict:
    """
    Display DE parameter controls in Streamlit.
    
    Returns:
        Dictionary of parameter values
    """
    st.subheader("Differential Evolution Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        population_size = st.slider(
            "Population Size",
            min_value=8,
            max_value=20,
            value=DE_CONFIG["population_size"],
            step=2,
            help="Number of candidate architectures in each generation"
        )
        
        generations = st.slider(
            "Generations",
            min_value=4,
            max_value=15,
            value=DE_CONFIG["generations"],
            step=1,
            help="Number of evolution iterations"
        )
        
        mutation_factor = st.slider(
            "Mutation Factor (F)",
            min_value=0.3,
            max_value=1.0,
            value=DE_CONFIG["mutation_factor"],
            step=0.05,
            help="Differential weight for mutation (higher = more exploration)"
        )
    
    with col2:
        crossover_rate = st.slider(
            "Crossover Rate (CR)",
            min_value=0.5,
            max_value=1.0,
            value=DE_CONFIG["crossover_rate"],
            step=0.05,
            help="Probability of inheriting from mutant vector"
        )
        
        seed = st.number_input(
            "Random Seed",
            min_value=0,
            max_value=9999,
            value=DE_CONFIG["seed"],
            help="Seed for reproducibility"
        )
        
        st.info(f"**Expected evaluations:** ~{population_size * (generations + 1)} models (with caching)")
    
    return {
        "population_size": population_size,
        "generations": generations,
        "mutation_factor": mutation_factor,
        "crossover_rate": crossover_rate,
        "seed": seed,
    }


def display_training_config() -> Dict:
    """
    Display training configuration controls.
    
    Returns:
        Dictionary of training parameters
    """
    st.subheader("Training Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        epochs = st.slider(
            "Training Epochs",
            min_value=2,
            max_value=6,
            value=TRAINING_CONFIG["epochs"],
            help="Number of epochs per model evaluation"
        )
        
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[1e-4, 5e-4, 1e-3, 5e-3],
            value=TRAINING_CONFIG["learning_rate"],
            format_func=lambda x: f"{x:.0e}",
            help="Fixed learning rate for all models"
        )
    
    with col2:
        batch_size = st.select_slider(
            "Batch Size",
            options=[64, 128, 256],
            value=TRAINING_CONFIG["batch_size"],
            help="Batch size for training"
        )
        
        train_samples = st.slider(
            "Training Samples",
            min_value=5000,
            max_value=30000,
            value=DATASET_CONFIG["train_samples"],
            step=5000,
            help="Number of Fashion-MNIST training samples"
        )
    
    return {
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "train_samples": train_samples,
    }


def display_fitness_weights() -> Dict:
    """
    Display fitness function weight controls.
    
    Returns:
        Dictionary of fitness weights
    """
    st.subheader("Fitness Function Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Penalties:**")
        param_penalty = st.slider(
            "Parameter Penalty Weight",
            min_value=0.0,
            max_value=0.5,
            value=FITNESS_WEIGHTS["param_penalty"],
            step=0.05,
            help="Penalty for models exceeding parameter limit"
        )
        
        time_penalty = st.slider(
            "Time Penalty Weight",
            min_value=0.0,
            max_value=0.5,
            value=FITNESS_WEIGHTS["time_penalty"],
            step=0.05,
            help="Penalty for slow training times"
        )
    
    with col2:
        st.markdown("**Limits:**")
        param_limit = st.number_input(
            "Parameter Limit",
            min_value=50000,
            max_value=200000,
            value=FITNESS_WEIGHTS["param_limit"],
            step=10000,
            help="Target maximum number of parameters"
        )
        
        time_limit = st.slider(
            "Time Limit (s/epoch)",
            min_value=10.0,
            max_value=60.0,
            value=FITNESS_WEIGHTS["time_limit"],
            step=5.0,
            help="Target maximum time per epoch"
        )
    
    return {
        "param_penalty": param_penalty,
        "time_penalty": time_penalty,
        "param_limit": param_limit,
        "time_limit": time_limit,
    }


def display_architecture_summary(arch_params: Dict):
    """Display a summary card of the architecture."""
    st.markdown("### Best Architecture Found")
    
    num_conv = arch_params["num_conv_layers"]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Conv Layers", num_conv)
        for i in range(1, num_conv + 1):
            st.text(f"Layer {i}: {arch_params[f'filters_{i}']} filters, {arch_params[f'kernel_size_{i}']}x{arch_params[f'kernel_size_{i}']}")
    
    with col2:
        st.metric("FC Units", arch_params["fc_units"])
        for i in range(1, num_conv + 1):
            pooling = arch_params[f"pooling_{i}"]
            st.text(f"Pool {i}: {pooling}")
    
    with col3:
        dropout_info = []
        for i in range(1, num_conv + 1):
            if arch_params[f"dropout_{i}"]:
                dropout_info.append(f"Layer {i}: {arch_params[f'dropout_rate_{i}']:.2f}")
            else:
                dropout_info.append(f"Layer {i}: None")
        
        st.metric("Dropout Layers", sum(arch_params[f"dropout_{i}"] for i in range(1, num_conv + 1)))
        for info in dropout_info:
            st.text(info)


def display_metrics_cards(metrics: Dict):
    """Display key metrics in card format."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Validation Accuracy",
            f"{metrics['val_accuracy']:.2f}%",
            help="Final validation accuracy"
        )
    
    with col2:
        st.metric(
            "Fitness Score",
            f"{metrics['fitness']:.4f}",
            help="Overall fitness (accuracy - penalties)"
        )
    
    with col3:
        st.metric(
            "Parameters",
            f"{metrics['num_params']:,}",
            delta=f"{metrics['param_ratio']:.2f}x limit" if metrics['param_ratio'] > 1 else "Within limit",
            delta_color="inverse",
            help="Total trainable parameters"
        )
    
    with col4:
        st.metric(
            "Avg Epoch Time",
            f"{metrics['avg_epoch_time']:.2f}s",
            delta=f"{metrics['time_ratio']:.2f}x limit" if metrics['time_ratio'] > 1 else "Within limit",
            delta_color="inverse",
            help="Average training time per epoch"
        )


def plot_convergence(generation_details: List[Dict]):
    """Plot fitness convergence over generations."""
    df = pd.DataFrame(generation_details)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df["generation"],
        y=df["best_fitness"],
        mode='lines+markers',
        name='Best Fitness',
        line=dict(color='green', width=3),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=df["generation"],
        y=df["mean_fitness"],
        mode='lines+markers',
        name='Mean Fitness',
        line=dict(color='blue', width=2, dash='dash'),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title="Fitness Convergence Over Generations",
        xaxis_title="Generation",
        yaxis_title="Fitness Score",
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig


def plot_training_curves(metrics: Dict):
    """Plot training loss and validation accuracy curves."""
    epochs = list(range(1, len(metrics["train_losses"]) + 1))
    
    fig = go.Figure()
    
    # Training loss on primary y-axis
    fig.add_trace(go.Scatter(
        x=epochs,
        y=metrics["train_losses"],
        mode='lines+markers',
        name='Training Loss',
        line=dict(color='red', width=2),
        yaxis='y1'
    ))
    
    # Validation accuracy on secondary y-axis
    fig.add_trace(go.Scatter(
        x=epochs,
        y=metrics["val_accuracies"],
        mode='lines+markers',
        name='Val Accuracy (%)',
        line=dict(color='green', width=2),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title="Best Model Training Curves",
        xaxis_title="Epoch",
        yaxis=dict(title="Training Loss", side='left'),
        yaxis2=dict(title="Validation Accuracy (%)", overlaying='y', side='right'),
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig


def plot_population_diversity(fitness_history: List[np.ndarray]):
    """Plot population diversity over generations."""
    generations = list(range(len(fitness_history)))
    std_devs = [np.std(fitness) for fitness in fitness_history]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=generations,
        y=std_devs,
        mode='lines+markers',
        name='Fitness Std Dev',
        line=dict(color='purple', width=2),
        marker=dict(size=6),
        fill='tozeroy',
        fillcolor='rgba(128, 0, 128, 0.2)'
    ))
    
    fig.update_layout(
        title="Population Diversity (Fitness Standard Deviation)",
        xaxis_title="Generation",
        yaxis_title="Standard Deviation",
        hovermode='x',
        template='plotly_white',
        height=350
    )
    
    return fig


def plot_cache_efficiency(generation_details: List[Dict]):
    """Plot cache hit efficiency over generations."""
    df = pd.DataFrame(generation_details)
    
    # Calculate evaluations per generation
    df["unique_evals"] = df["cache_size"].diff().fillna(df["cache_size"])
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df["generation"],
        y=df["unique_evals"],
        name='Unique Evaluations',
        marker_color='steelblue'
    ))
    
    fig.update_layout(
        title="Unique Architecture Evaluations Per Generation (Cache Efficiency)",
        xaxis_title="Generation",
        yaxis_title="Unique Architectures Evaluated",
        hovermode='x',
        template='plotly_white',
        height=350
    )
    
    return fig


def display_test_results(test_results: Dict):
    """Display final test performance statistics."""
    st.markdown("### Final Test Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Mean Test Accuracy",
            f"{test_results['mean_test_accuracy']:.2f}%",
            help="Average test accuracy across multiple runs"
        )
    
    with col2:
        st.metric(
            "Std Dev",
            f"±{test_results['std_test_accuracy']:.2f}%",
            help="Standard deviation of test accuracy"
        )
    
    with col3:
        st.metric(
            "Mean Training Time",
            f"{test_results['mean_test_time']:.2f}s",
            help="Average total training time"
        )
    
    # Show individual runs
    with st.expander("Individual Test Runs"):
        test_df = pd.DataFrame({
            "Run": list(range(1, len(test_results["test_accuracies"]) + 1)),
            "Test Accuracy (%)": test_results["test_accuracies"],
            "Training Time (s)": test_results["test_times"]
        })
        st.dataframe(test_df, width='stretch')


def display_prediction_examples(predictions_data: Dict, class_labels: list):
    """Display a grid of example predictions with images."""
    st.markdown("### Prediction Examples")
    
    example_images = predictions_data["example_images"]
    example_preds = predictions_data["example_predictions"]
    example_labels = predictions_data["example_labels"]
    
    # Display in 4x4 grid
    num_examples = len(example_images)
    cols_per_row = 4
    num_rows = (num_examples + cols_per_row - 1) // cols_per_row
    
    for row in range(num_rows):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            idx = row * cols_per_row + col_idx
            if idx < num_examples:
                with cols[col_idx]:
                    # Convert tensor to displayable format
                    img = example_images[idx].squeeze().numpy()
                    # Denormalize from [-1, 1] to [0, 1]
                    img = (img + 1) / 2
                    
                    # Create figure
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(2, 2))
                    ax.imshow(img, cmap='gray')
                    ax.axis('off')
                    
                    # Color code: green if correct, red if wrong
                    is_correct = example_preds[idx] == example_labels[idx]
                    border_color = 'green' if is_correct else 'red'
                    
                    for spine in ax.spines.values():
                        spine.set_edgecolor(border_color)
                        spine.set_linewidth(3)
                        spine.set_visible(True)
                    
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # Labels
                    pred_label = class_labels[example_preds[idx]]
                    true_label = class_labels[example_labels[idx]]
                    
                    if is_correct:
                        st.success(f"Correct! {pred_label}")
                    else:
                        st.error(f"Incorrect! Pred: {pred_label}")
                        st.caption(f"True: {true_label}")


def plot_confusion_matrix(predictions_data: Dict, class_labels: list):
    """Plot confusion matrix as a heatmap."""
    from sklearn.metrics import confusion_matrix
    
    all_preds = predictions_data["all_predictions"]
    all_labels = predictions_data["all_labels"]
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Normalize by row (true labels) to show percentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=cm_normalized,
        x=class_labels,
        y=class_labels,
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate='True: %{y}<br>Pred: %{x}<br>Count: %{text}<br>Percent: %{z:.1f}%<extra></extra>',
        colorbar=dict(title="Accuracy (%)")
    ))
    
    fig.update_layout(
        title="Confusion Matrix (normalized by true class)",
        xaxis_title="Predicted Class",
        yaxis_title="True Class",
        xaxis={'side': 'bottom'},
        yaxis={'autorange': 'reversed'},
        width=700,
        height=700,
    )
    
    return fig
