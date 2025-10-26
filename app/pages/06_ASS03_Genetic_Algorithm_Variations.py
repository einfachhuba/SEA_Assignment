import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time
from PIL import Image
from utils.general.pdf_viewer import display_pdf_with_controls
from utils.general.ui import spacer
from utils.genetic_algorithm_variations.ui import display_strategy_selection, display_strategy_explanations, display_parameter_controls, display_image_controls, display_initialization_controls, display_results_summary
from utils.genetic_algorithm_variations.image_utils import process_uploaded_image, generate_pattern_image
from utils.genetic_algorithm_variations.functions import create_default_test_image
from utils.genetic_algorithm_variations.algorithms import ImageReconstructionGA

st.set_page_config(page_title="AT03: Genetic Algorithm Variations")

tab1, tab2 = st.tabs(["📄 Assignment Paper", "🔧 Genetic Algorithm Variations Implementation"])

with tab1:
    st.title("Assignment 03: Genetic Algorithm Variations Document")
    pdf_path = "Assignment_Sheets/03/SEA_Exercise03_Genetic_Algorithm_Variations.pdf"

    display_pdf_with_controls(pdf_path)

with tab2:
    st.title("Assignment 03: Genetic Algorithm Variations Implementation")
    
    st.header("Introduction")
    st.markdown("""
    **Genetic Algorithm Variations** explore advanced strategies and improvements over classical genetic algorithms.
    This assignment focuses on **image reconstruction** - a high-dimensional optimization problem where we attempt to 
    reconstruct a target image by evolving a population of candidate pixel matrices using various genetic operators and strategies.
    
    Unlike our previous coffee brewing optimization (4D problem), image reconstruction presents challenges with 
    **hundreds to thousands of dimensions** (one per pixel), making it an excellent testbed for advanced GA techniques.
    """)

    spacer(6)

    cont = st.container(border=True)
    with cont:
        col1, col2 = st.columns(2)
        with col1:
            st.badge("**Key Improvements:**", color="blue")
            st.markdown("""
            - **Advanced Selection**: Tournament, roulette wheel, rank-based selection
            - **Diverse Crossover**: Single/two-point, uniform, arithmetic crossover
            - **Smart Mutation**: Gaussian, uniform, polynomial mutation strategies
            - **Survivor Strategies**: Elitist, steady-state, tournament replacement
            - **Intelligent Initialization**: Edge-based, oversampling & selection, local optimization, noise-based
            """)

        with col2:
            st.badge("**Problem Characteristics:**", color="blue")
            st.markdown("""
            - **High-dimensional**: 256+ variables (16x16 image = 256 pixels)
            - **Large search space**: 256^n possible solutions
            - **Clear fitness metric**: Pixel-wise similarity to target image
            - **Visualization friendly**: Can see population evolution visually
            - **Scalable complexity**: Can test different image sizes
            """)

    spacer(24)

    st.header("Methods")

    # Strategy explanations
    display_strategy_explanations()

    spacer(12)

    st.markdown("""
    ### Implementation Details

    **Problem Encoding:** Each individual represents a grayscale image as a 1D array of pixel intensities [0, 255].
    For a 16x16 image, each chromosome contains 256 genes (one per pixel).

    **Fitness Function:** Normalized pixel-wise Mean Squared Error (MSE):
    ```
    fitness = 1.0 - (MSE / max_possible_MSE)
    ```
    where `max_possible_MSE = 255²`, giving fitness values in [0, 1].

    **Termination Criteria:**
    - Fitness threshold reached (default: 0.95)
    - Maximum generations exceeded
    - Stagnation limit reached (no improvement for N generations)

    **Performance Metrics:**
    - **MSE (Mean Squared Error)**: Average squared pixel difference between images (lower is better)
    - **PSNR (Peak Signal-to-Noise Ratio)**: Image quality measure in decibels (higher is better, typically 20-40 dB)
    - **SSIM (Structural Similarity Index)**: Perceptual similarity measure (0-1 range, higher is better)
    """)

    spacer(24)

    st.header("Algorithm Configuration")

    # Image configuration
    image_config = display_image_controls()
    
    spacer(12)

    # Load or create target image
    target_image = None
    
    if image_config['image_source'] == 'generated':
        # Generate pattern based on selection
        if image_config['pattern_type'] == 'default_test':
            target_image = create_default_test_image(image_config['image_size'])
            st.info("Using default test pattern with geometric shapes.")
        else:
            target_image = generate_pattern_image(
                image_config['pattern_type'], 
                image_config['image_size']
            )
            st.success(f"Generated {image_config['pattern_type']} pattern!")
        
    elif image_config['image_source'] == 'upload' and image_config['uploaded_file']:
        try:
            # Load uploaded image
            pil_image = Image.open(image_config['uploaded_file'])
            target_image = process_uploaded_image(pil_image, image_config['image_size'])
            st.success("Image uploaded and preprocessed successfully!")
        except Exception as e:
            st.error(f"Error processing uploaded image: {e}")
            target_image = create_default_test_image(image_config['image_size'])
            st.info("Falling back to default test pattern.")
    
    else:
        target_image = create_default_test_image(image_config['image_size'])
        st.info("Using default test pattern.")

    # Display target image
    if target_image is not None:
        st.subheader("Target Image")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            fig_target = go.Figure(data=go.Heatmap(
                z=target_image,
                colorscale='gray',
                showscale=True
            ))
            fig_target.update_layout(
                title=f"Target Image ({target_image.shape[0]}x{target_image.shape[1]} pixels)",
                height=400,
                width=400
            )
            fig_target.update_xaxes(showticklabels=False)
            fig_target.update_yaxes(showticklabels=False, autorange='reversed')  # Reverse y-axis for correct orientation
            st.plotly_chart(fig_target, use_container_width=False)  # Don't stretch to full width

    spacer(12)

    # Strategy selection
    strategies = display_strategy_selection()
    
    spacer(12)

    # Initialization strategy
    initialization_strategy = display_initialization_controls()
    
    spacer(12)

    # Parameter controls
    params = display_parameter_controls()

    ## GA RUNNING LOGIC ##
    
    spacer(12)
    
    # Run button
    if st.button("🚀 Run Genetic Algorithm", type="primary", use_container_width=True):
        if target_image is None:
            st.error("Please configure a target image first!")
        else:
            # Create progress containers
            progress_bar = st.progress(0)
            status_text = st.empty()
            metrics_placeholder = st.empty()
            
            # Initialize GA
            ga = ImageReconstructionGA(
                target_image=target_image,
                population_size=params['population_size'],
                max_generations=params['max_generations'],
                mutation_rate=params['mutation_rate'],
                crossover_rate=params['crossover_rate'],
                tournament_size=params['tournament_size'],
                elite_size=params['elite_size'],
                selection_strategy=strategies['selection'],
                crossover_strategy=strategies['crossover'],
                mutation_strategy=strategies['mutation'],
                survivor_strategy=strategies['survivor'],
                initialization_strategy=initialization_strategy,
                fitness_threshold=params['fitness_threshold'],
                max_stagnation=params['max_stagnation'],
                seed=params['seed'] if params['seed'] != -1 else None
            )
            
            # Progress callback
            def update_progress(generation, max_gen, fitness, metrics):
                progress = (generation + 1) / max_gen
                progress_bar.progress(progress)
                status_text.text(f"Generation {generation + 1}/{max_gen} | Fitness: {fitness:.4f}")
                
                # Update metrics every 10 generations
                if generation % 10 == 0 or generation == max_gen - 1:
                    with metrics_placeholder.container():
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Current Fitness", f"{fitness:.4f}")
                        with col2:
                            st.metric("MSE (Mean Squared Error)", f"{metrics['mse']:.2f}")
                        with col3:
                            st.metric("PSNR (Peak SNR)", f"{metrics['psnr']:.2f} dB")
                        with col4:
                            st.metric("SSIM (Structural Similarity)", f"{metrics['ssim']:.4f}")
            
            # Run GA
            with st.spinner("Running Genetic Algorithm..."):
                results = ga.run(progress_callback=update_progress)
            
            progress_bar.empty()
            status_text.empty()
            metrics_placeholder.empty()
                        
            spacer(12)
            
            # Store results in session state for persistence
            st.session_state['ga_results'] = results
            st.session_state['ga_strategies'] = strategies
            st.session_state['ga_target_image'] = target_image
    
    # Display results if they exist in session state
    if 'ga_results' in st.session_state:
        display_results_summary(
            st.session_state['ga_results'],
            st.session_state['ga_strategies'],
            st.session_state['ga_target_image']
        )
    
    spacer(24)

    st.header("Discussion")

    st.markdown("""
    ### Algorithm Performance Analysis

    The genetic algorithm variations demonstrate several key behaviors in image reconstruction:

    **Selection Strategy Impact:**
    - **Random Selection**: Maintains high diversity but slow convergence
    - **Rank-based**: Robust performance across different fitness landscapes
    - **Roulette Wheel**: Can lead to premature convergence with high fitness variance
    - **Tournament Selection**: Provides good balance between exploration and exploitation

    **Crossover Strategy Effects:**
    - **Arithmetic**: Creates smooth blends, good for fine-tuning but slow initial progress
    - **Single/Two-point**: Can preserve image regions but may be too disruptive
    - **Uniform Crossover**: Excellent for image reconstruction - mixes pixels effectively

    **Mutation Strategy Characteristics:**
    - **Bit-flip Mutation**: High disruption, useful for escaping local optima
    - **Gaussian Mutation**: Best for fine-tuning pixel values with small adjustments
    - **Insertion Mutation**: Reorders pixel positions while preserving values, explores different spatial arrangements
    - **Uniform Mutation**: Applies random changes across the entire image, promoting diversity
                
    **Survivor Selection Effects:**
    - **Elitist Replacement**: Preserves best individuals, accelerates convergence but risks diversity loss
    - **Generational Replacement**: Full population turnover, maintains diversity but slower convergence
    - **Steady State**: Replaces worst parents with best offspring, slower turnover preserves population quality
    - **Tournament Replacement**: Competitive survival, maintains quality while allowing diversity
    """)

    spacer(12)

    st.markdown("""
    ### Exploration vs Exploitation Analysis

    **High Exploration Strategies:**
    - Random selection + Uniform crossover + Bit-flip mutation
    - Maintains diversity but slower convergence
    - Better for complex, multimodal landscapes

    **High Exploitation Strategies:**  
    - Tournament selection + Arithmetic crossover + Gaussian mutation
    - Faster convergence but risk of premature convergence
    - Better for simpler landscapes or fine-tuning

    **Balanced Approaches:**
    - Tournament selection + Uniform crossover + Gaussian mutation + Elitist survival
    - Combines effective exploration with selective pressure
    - Recommended for most image reconstruction tasks

    ### Computational Complexity

    **Computational Bottlenecks:**
    - Fitness evaluations dominate runtime (O(image_size) per evaluation)
    - Population diversity calculations: O(population_size²)
    - Selection operations: O(population_size x tournament_size)
    """)

    spacer(12)

    st.markdown("""
    ### Comparison with Classical GA

    **Advantages of Variations:**
    - **Better Selection Pressure**: Tournament/rank selection vs random selection
    - **Diverse Operators**: Multiple crossover/mutation strategies vs single fixed operator
    - **Intelligent Initialization**: Problem-aware starting population vs pure random
    - **Flexible Survival**: Multiple replacement strategies vs simple generational

    **Trade-offs:**
    - **Complexity**: More parameters to tune and strategies to choose
    - **Computational Cost**: Some strategies (rank selection, diversity calculation) more expensive
    """)

    spacer(24)

    st.header("Conclusion")

    st.markdown("""

    **Key Findings:**
    - **Strategy selection matters**: Performance can vary by orders of magnitude
    - **Balanced approaches work best**: Pure exploration or exploitation often suboptimal
    - **Scalability challenges**: Higher-dimensional problems require careful parameter tuning

    """)


