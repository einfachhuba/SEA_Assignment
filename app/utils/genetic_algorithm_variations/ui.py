"""
UI components for Genetic Algorithm Variations
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any


def display_strategy_selection() -> Dict[str, str]:
    """
    Display strategy selection UI components.
    
    Returns:
        dict: Selected strategies
    """
    st.subheader("Algorithm Strategy Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Selection & Crossover**")
        
        selection_strategy = st.selectbox(
            "Parent Selection Strategy",
            options=['random', 'rank', 'roulette', 'tournament'],
            format_func=lambda x: {
                'random': 'Random Selection',
                'rank': 'Rank-based Selection',
                'roulette': 'Roulette Wheel Selection',
                'tournament': 'Tournament Selection'
            }[x],
            index=3  # Default to tournament
        )
        
        crossover_strategy = st.selectbox(
            "Crossover Strategy",
            options=['arithmetic', 'single_point', 'two_point', 'uniform'],
            format_func=lambda x: {
                'arithmetic': 'Arithmetic Crossover',
                'single_point': 'Single Point Crossover',
                'two_point': 'Two Point Crossover',
                'uniform': 'Uniform Crossover'
            }[x],
            index=3  # Default to uniform
        )
    
    with col2:
        st.markdown("**Mutation & Survival**")
        
        mutation_strategy = st.selectbox(
            "Mutation Strategy",
            options=['bit_flip', 'gaussian', 'insertion', 'uniform'],
            format_func=lambda x: {
                'bit_flip': 'Bit Flip Mutation',
                'gaussian': 'Gaussian Mutation',
                'insertion': 'Insertion Mutation',
                'uniform': 'Uniform Mutation'
            }[x],
            index=1  # Default to gaussian
        )
        
        survivor_strategy = st.selectbox(
            "Survivor Selection Strategy",
            options=['elitist', 'generational', 'steady_state', 'tournament_replacement'],
            format_func=lambda x: {
                'elitist': 'Elitist Replacement',
                'generational': 'Generational Replacement',
                'steady_state': 'Steady State',
                'tournament_replacement': 'Tournament Replacement'
            }[x],
            index=0  # Default to elitist
        )
    
    return {
        'selection': selection_strategy,
        'crossover': crossover_strategy,
        'mutation': mutation_strategy,
        'survivor': survivor_strategy
    }


def display_parameter_controls() -> Dict[str, Any]:
    """
    Display parameter control UI components.
    
    Returns:
        dict: Parameter values
    """
    st.subheader("Algorithm Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Population & Generations**")
        population_size = st.slider("Population Size", 20, 300, 100, step=10)
        max_generations = st.slider("Max Generations", 50, 1000, 500, step=50)
        
    with col2:
        st.markdown("**Genetic Operators**")
        mutation_rate = st.slider("Mutation Rate", 0.001, 0.1, 0.01, step=0.001, format="%.3f")
        crossover_rate = st.slider("Crossover Rate", 0.1, 1.0, 0.8, step=0.1)
        
    with col3:
        st.markdown("**Selection Parameters**")
        tournament_size = st.slider("Tournament Size", 2, 10, 3, step=1)
        elite_size = st.slider("Elite Size", 1, 20, 5, step=1)
    
    # Advanced parameters in expander
    with st.expander("Advanced Parameters"):
        col_a, col_b = st.columns(2)
        with col_a:
            fitness_threshold = st.slider(
                "Fitness Threshold", 
                0.8, 0.999, 0.90, 
                step=0.001, 
                format="%.3f",
                help="Algorithm stops when this fitness is reached. Lower = faster but less accurate reconstruction."
            )
            seed = st.number_input("Random Seed", value=42, step=1)
        with col_b:
            max_stagnation = st.slider("Max Stagnation Generations", 10, 200, 50, step=10)
    
    return {
        'population_size': population_size,
        'max_generations': max_generations,
        'mutation_rate': mutation_rate,
        'crossover_rate': crossover_rate,
        'tournament_size': tournament_size,
        'elite_size': elite_size,
        'fitness_threshold': fitness_threshold,
        'max_stagnation': max_stagnation,
        'seed': seed
    }


def display_image_controls() -> Dict[str, Any]:
    """
    Display image-related control UI components.
    
    Returns:
        dict: Image configuration
    """
    st.subheader("Image Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Image Source**")
        image_source = st.radio(
            "Choose image source:",
            options=['generated', 'upload'],
            format_func=lambda x: {
                'generated': 'Generated Pattern',
                'upload': 'Upload Image'
            }[x]
        )
        
    with col2:
        st.markdown("**Image Size**")
        image_size = st.selectbox(
            "Image Dimensions",
            options=[(8, 8), (16, 16), (24, 24), (32, 32), (48, 48)],
            format_func=lambda x: f"{x[0]}×{x[1]} pixels",
            index=1  # Default to 16x16
        )
    
    uploaded_file = None
    pattern_type = None
    
    if image_source == 'upload':
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            help="Upload a grayscale or color image. It will be converted to grayscale and resized."
        )
    
    elif image_source == 'generated':
        pattern_type = st.selectbox(
            "Pattern Type",
            options=['checkerboard', 'circles', 'default_test', 'gradient', 'noise', 'squares'],
            format_func=lambda x: {
                'checkerboard': 'Checkerboard',
                'circles': 'Concentric Circles',
                'default_test': 'Default Test Pattern (Geometric Shapes)',
                'gradient': 'Gradient',
                'noise': 'Random Noise',
                'squares': 'Nested Squares'
            }[x],
            index=2  # Default to default_test pattern
        )
    
    return {
        'image_source': image_source,
        'image_size': image_size,
        'uploaded_file': uploaded_file,
        'pattern_type': pattern_type
    }


def display_initialization_controls() -> str:
    """
    Display initialization strategy controls.
    
    Returns:
        str: Selected initialization strategy
    """
    st.subheader("Population Initialization Strategy")
    
    initialization_strategy = st.selectbox(
        "Initialization Method",
        options=['edge_based', 'gaussian_noise', 'local_optimization', 'oversampling_selection', 'random'],
        format_func=lambda x: {
            'edge_based': 'Edge-based Initialization',
            'gaussian_noise': 'Gaussian Noise Initialization',
            'local_optimization': 'Local Optimization (Hill Climbing)',
            'oversampling_selection': 'Oversampling and Selection',
            'random': 'Random Initialization'
        }[x]
    )
    
    # Display description of selected strategy
    descriptions = {
        'edge_based': "Initialize based on edge information from target image. Can provide a good starting point for images with clear boundaries.",
        'gaussian_noise': "Initialize around the mean pixel value of target image with Gaussian noise. Balanced approach for most images.",
        'local_optimization': "Initialize randomly then apply hill climbing to each individual (20 iterations). Significantly improves initial fitness but takes longer.",
        'oversampling_selection': "Create 3x larger population, evaluate all individuals, and select the best ones. Computationally expensive but provides high-quality initial population.",
        'random': "Initialize with completely random pixel values. Simple but may require more generations to converge."
    }
    
    st.info(descriptions[initialization_strategy])
    
    return initialization_strategy

def display_strategy_explanations():
    """Display explanations of different strategies."""
    st.subheader("Strategy Explanations")
    
    with st.expander("Selection Strategies"):
        st.markdown("""
        **Random Selection:** Randomly choose parents. No selection pressure - all individuals equally likely.
        
        **Tournament Selection:** Hold tournaments between random individuals, select winners as parents. Provides selection pressure while maintaining diversity.
        
        **Roulette Wheel Selection:** Selection probability proportional to fitness. Better individuals more likely to be selected.
        
        **Rank-based Selection:** Selection based on fitness ranking rather than absolute values. Reduces selection pressure when fitness values vary widely.
        """)
    
    with st.expander("Crossover Strategies"):
        st.markdown("""
        **Single Point Crossover:** Cut chromosomes at random point, exchange tails. Simple and commonly used.
        
        **Two Point Crossover:** Cut at two points, exchange middle section. Can preserve more structure than single point.
        
        **Uniform Crossover:** Each gene independently chosen from either parent. High disruption but good mixing.
        
        **Arithmetic Crossover:** Create offspring as weighted average of parents. Produces intermediate solutions.
        """)
    
    with st.expander("Mutation Strategies"):
        st.markdown("""
        **Bit Flip Mutation:** Replace genes with completely random values. High disruption, good for escaping local optima.
        
        **Gaussian Mutation:** Add normally distributed noise to genes. Smaller changes, good for fine-tuning.
        
        **Insertion Mutation:** Select random positions and insert pixel values at different positions. Reorders gene sequence, useful for preserving values while changing structure.
        
        **Uniform Mutation:** Add uniformly distributed noise to genes. Medium disruption level.
        """)
    
    with st.expander("Survivor Selection Strategies"):
        st.markdown("""
        **Generational Replacement:** Replace entire population with offspring. Fast generational turnover.
        
        **Elitist Replacement:** Combine parents and offspring, select best individuals. Preserves good solutions.
        
        **Steady State:** Replace worst parents with best offspring gradually. Slower population turnover.
        
        **Tournament Replacement:** Tournament between parent and offspring for survival. Balanced approach.
        """)


def display_results_summary(results: Dict[str, Any], strategies: Dict[str, str], target_image: np.ndarray):
    """
    Display comprehensive results summary with visualizations.
    
    Args:
        results: GA results dictionary
        strategies: Selected strategies
        target_image: Original target image
    """
    st.subheader("Results Summary")
    
    # Metrics overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Final Fitness",
            f"{results['best_fitness']:.4f}",
            delta=None
        )
    
    with col2:
        st.metric(
            "Generations",
            f"{results['generations']}",
            delta=None
        )
    
    with col3:
        st.metric(
            "MSE (Mean Squared Error)",
            f"{results['final_metrics']['mse']:.2f}",
            delta=None,
            help="Lower is better - measures average squared pixel difference"
        )
    
    with col4:
        st.metric(
            "SSIM (Structural Similarity)",
            f"{results['final_metrics']['ssim']:.4f}",
            delta=None,
            help="Range [0-1], higher is better - measures structural similarity"
        )
    
    # Additional metrics
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric(
            "PSNR (Peak Signal-to-Noise Ratio)",
            f"{results['final_metrics']['psnr']:.2f} dB",
            delta=None,
            help="Higher is better - measures image quality in decibels"
        )
    
    with col6:
        st.metric(
            "Fitness Evaluations",
            f"{results['fitness_evaluations']:,}",
            delta=None
        )
    
    with col7:
        st.metric(
            "Execution Time",
            f"{results['execution_time']:.2f}s",
            delta=None
        )
    
    with col8:
        avg_time_per_gen = results['execution_time'] / results['generations']
        st.metric(
            "Time/Generation",
            f"{avg_time_per_gen:.3f}s",
            delta=None
        )
    
    # Image comparison
    st.subheader("Image Comparison")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Target Image**")
        fig_target = go.Figure(data=go.Heatmap(
            z=target_image,
            colorscale='gray',
            showscale=True,
            colorbar=dict(title="Intensity")
        ))
        fig_target.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        fig_target.update_xaxes(showticklabels=False)
        fig_target.update_yaxes(showticklabels=False, autorange='reversed')  # Reverse y-axis for correct orientation
        st.plotly_chart(fig_target, use_container_width=True)
    
    with col2:
        st.markdown("**Reconstructed Image**")
        fig_result = go.Figure(data=go.Heatmap(
            z=results['best_individual'],
            colorscale='gray',
            showscale=True,
            colorbar=dict(title="Intensity")
        ))
        fig_result.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        fig_result.update_xaxes(showticklabels=False)
        fig_result.update_yaxes(showticklabels=False, autorange='reversed')  # Reverse y-axis for correct orientation
        st.plotly_chart(fig_result, use_container_width=True)
    
    with col3:
        st.markdown("**Absolute Difference**")
        diff = np.abs(target_image - results['best_individual'])
        fig_diff = go.Figure(data=go.Heatmap(
            z=diff,
            colorscale='hot',
            showscale=True,
            colorbar=dict(title="Difference")
        ))
        fig_diff.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        fig_diff.update_xaxes(showticklabels=False)
        fig_diff.update_yaxes(showticklabels=False, autorange='reversed')  # Reverse y-axis for correct orientation
        st.plotly_chart(fig_diff, use_container_width=True)
    
    # Evolution plots
    st.subheader("Evolution Over Generations")
    
    # Fitness evolution
    fig_fitness = go.Figure()
    
    generations = list(range(len(results['history']['best_fitness'])))
    
    fig_fitness.add_trace(go.Scatter(
        x=generations,
        y=results['history']['best_fitness'],
        mode='lines',
        name='Best Fitness',
        line=dict(color='green', width=2)
    ))
    
    fig_fitness.add_trace(go.Scatter(
        x=generations,
        y=results['history']['avg_fitness'],
        mode='lines',
        name='Average Fitness',
        line=dict(color='blue', width=2)
    ))
    
    fig_fitness.add_trace(go.Scatter(
        x=generations,
        y=results['history']['worst_fitness'],
        mode='lines',
        name='Worst Fitness',
        line=dict(color='red', width=1, dash='dash')
    ))
    
    fig_fitness.update_layout(
        title="Fitness Evolution",
        xaxis_title="Generation",
        yaxis_title="Fitness",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_fitness, use_container_width=True)
    
    # Metrics evolution
    col1, col2 = st.columns(2)
    
    with col1:
        fig_mse = go.Figure()
        fig_mse.add_trace(go.Scatter(
            x=generations,
            y=results['history']['mse'],
            mode='lines',
            name='MSE',
            line=dict(color='red', width=2)
        ))
        fig_mse.update_layout(
            title="MSE (Mean Squared Error) Evolution",
            xaxis_title="Generation",
            yaxis_title="Mean Squared Error",
            height=300
        )
        st.plotly_chart(fig_mse, use_container_width=True)
        
        fig_diversity = go.Figure()
        fig_diversity.add_trace(go.Scatter(
            x=generations,
            y=results['history']['diversity'],
            mode='lines',
            name='Diversity',
            line=dict(color='purple', width=2)
        ))
        fig_diversity.update_layout(
            title="Population Diversity",
            xaxis_title="Generation",
            yaxis_title="Avg Pixel Difference",
            height=300
        )
        st.plotly_chart(fig_diversity, use_container_width=True)
    
    with col2:
        fig_psnr = go.Figure()
        fig_psnr.add_trace(go.Scatter(
            x=generations,
            y=results['history']['psnr'],
            mode='lines',
            name='PSNR',
            line=dict(color='green', width=2)
        ))
        fig_psnr.update_layout(
            title="PSNR (Peak Signal-to-Noise Ratio) Evolution",
            xaxis_title="Generation",
            yaxis_title="Peak Signal-to-Noise Ratio",
            height=300
        )
        st.plotly_chart(fig_psnr, use_container_width=True)
        
        fig_ssim = go.Figure()
        fig_ssim.add_trace(go.Scatter(
            x=generations,
            y=results['history']['ssim'],
            mode='lines',
            name='SSIM',
            line=dict(color='blue', width=2)
        ))
        fig_ssim.update_layout(
            title="SSIM (Structural Similarity) Evolution",
            xaxis_title="Generation",
            yaxis_title="Structural Similarity",
            height=300
        )
        st.plotly_chart(fig_ssim, use_container_width=True)
    
    # Fitness evaluations
    fig_evals = go.Figure()
    fig_evals.add_trace(go.Scatter(
        x=generations,
        y=results['history']['fitness_evaluations'],
        mode='lines',
        name='Fitness Evaluations',
        line=dict(color='orange', width=2),
        fill='tozeroy'
    ))
    fig_evals.update_layout(
        title="Cumulative Fitness Evaluations",
        xaxis_title="Generation",
        yaxis_title="Total Evaluations",
        height=300
    )
    st.plotly_chart(fig_evals, use_container_width=True)


def css_styling():
    """Apply custom CSS styling."""
    st.markdown("""
    <style>
    .metric-container {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
    }
    
    .strategy-badge {
        background-color: #e8f4f8;
        border-radius: 15px;
        padding: 5px 10px;
        margin: 2px;
        display: inline-block;
        font-size: 0.8em;
    }
    
    .improvement-positive {
        color: #28a745;
        font-weight: bold;
    }
    
    .improvement-negative {
        color: #dc3545;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
