"""
Assignment 08: Anything Goes - Neural Architecture Search using Differential Evolution
"""

import streamlit as st
import torch
import sys
import os

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from utils.general.pdf_viewer import display_pdf_with_controls
from utils.general.ui import spacer
from utils.anything_goes.config import (
    TRAINING_CONFIG, DATASET_CONFIG, FITNESS_WEIGHTS
)
from utils.anything_goes.ui import (
    display_parameter_controls, display_training_config, display_fitness_weights,
    display_architecture_summary, display_metrics_cards, plot_convergence,
    plot_training_curves, plot_population_diversity, plot_cache_efficiency,
    display_test_results, display_prediction_examples, plot_confusion_matrix
)
from utils.anything_goes.functions import (
    load_fashion_mnist, test_final_model, get_predictions_and_examples, CNNArchitecture
)
from utils.anything_goes.algorithms import DifferentialEvolutionNAS


st.set_page_config(page_title="ASS08: Anything Goes - NAS")

tab1, tab2 = st.tabs(["📄 Assignment Paper", "🔧 Neural Architecture Search"])

with tab1:
    st.title("Assignment 08: Anything Goes - Network Architecture Search")
    pdf_path = "Assignment_Sheets/08/SEA_Exercise08_Anything_Goes.pdf"
    try:
        display_pdf_with_controls(pdf_path)
    except Exception:
        st.info("Assignment PDF not found. Proceed to the implementation tab.")

with tab2:
    st.title("Assignment 08: Neural Architecture Search with Differential Evolution")
    
    st.header("Introduction")
    st.markdown("""
    **Neural Architecture Search (NAS)** automates the design of neural network architectures by treating
    network design as an optimization problem. Instead of manually tuning layer configurations, NAS algorithms
    explore the space of possible architectures to find high-performing models.
    """)
    
    spacer(6)    
    st.header("Algorithm Justification")
    st.markdown("""
    ### Why Differential Evolution for NAS?
    
    **Differential Evolution** was selected over other metaheuristics for several key reasons:
    
    1. **Continuous Encoding with Discrete Mapping**: DE naturally operates on continuous vectors in [0,1], 
       which we map to discrete architectural choices (filter counts, kernel sizes, dropout flags). This encoding 
       is simpler than bit-string GAs or discrete state spaces for SA/ACO.
    
    2. **Robust Global Search**: Unlike Hill Climbing or Simulated Annealing (single-solution methods), DE maintains 
       a population that explores multiple promising regions simultaneously, reducing the risk of premature convergence 
       to suboptimal architectures.
    
    3. **Simplicity vs. Complexity Trade-off**: Compared to Particle Swarm Optimization (velocity dynamics, neighbor topologies) 
       or Ant Colony Optimization (pheromone matrices, construction graphs), DE has a straightforward mutation/crossover/selection 
       loop with only two main hyperparameters (F and CR).
    
    4. **Evaluation Efficiency**: DE's deterministic selection (keep better offspring) ensures monotonic improvement 
       of the best solution, unlike GAs where good individuals can be lost through stochastic selection. Combined with 
       caching, this minimizes wasted evaluations.
    
    **Considered Alternatives:**
    - **Genetic Algorithms**: More complex selection/crossover operators; risk of losing elite solutions.
    - **Simulated Annealing**: Single-solution search may get trapped; temperature tuning is problem-specific.
    - **ACO/PSO**: Require more careful design for discrete architecture spaces; additional hyperparameters.
    
    While a **hybrid approach** (e.g., DE for global search + local hill climbing for fine-tuning) could yield marginal 
    improvements, the high computational cost of training neural networks makes multi-phase strategies impractical when 
    each architecture evaluation requires minutes of GPU time.
    """)
    
    spacer(24)
    
    st.header("Methods")
    
    st.markdown("""
    ### Problem Formulation
    
    **Search Space:** Each CNN architecture is encoded as a 17-dimensional real vector $x \in [0,1]^{17}$ representing:
    - Number of convolutional layers (1-3)
    - Filter counts per layer (8, 16, 24, or 32)
    - Kernel sizes (3x3 or 5x5)
    - Pooling types (max or average)
    - Dropout flags (yes/no) and rates (0-0.5)
    - Fully connected layer size (32-128 neurons)
    
    Each continuous dimension is **discretized** via binning/thresholding to map to valid architectural choices.
    
    **Fitness Function:**  
    We maximize validation accuracy while penalizing large and slow models:
    
    $fitness = accuracy - \\alpha \\cdot max(0, params/limit - 1) - \\beta \\cdot max(0, time/limit - 1)$
    
    Where:
    - `accuracy`: Normalized validation accuracy $\in [0, 1]$
    - $\\alpha, \\beta$: Penalty weights for parameters and training time
    - `limit`: Target thresholds for parameters (~100k) and time (~30s/epoch)
    
    This multi-objective formulation encourages **efficient architectures** that balance accuracy with computational cost.
    
    ### Differential Evolution Algorithm
        
    1. **Initialization**: Generate population of N random vectors in $[0,1]^{17}$
    2. **For each generation:**
       - For each target vector $x_i$:
         - **Mutation**: Create mutant $v = x_a + F \cdot (x_b - x_c)$ using three random distinct vectors
         - **Crossover**: Mix mutant and target via binomial crossover (rate CR)
         - **Evaluation**: Decode trial vector -> build CNN -> train 2-6 epochs -> compute fitness
         - **Selection**: Keep trial if fitness(trial) $\geq$ fitness(target), else keep target
    3. **Termination**: Stop after G generations; return best vector found
    
    **Key Parameters:**
    - Population size N $ \in [8, 20]$: Balances diversity vs. computation
    - Mutation factor F $\in [0.3, 1.0]$: Controls exploration magnitude
    - Crossover rate CR $\in [0.5, 1.0]$: Probability of inheriting from mutant
    
    **Caching Optimization:** Architectures are hashed by their discrete parameters; duplicate evaluations 
    return cached fitness, significantly reducing redundant training.
    
    ### Training Protocol
    
    Each candidate CNN is trained for **4 epochs** (configurable 2-6) with:
    - Fixed learning rate: [1e-4, 5e-4, 1e-3, 5e-3] (configurable)
    - Batch size: [64, 128, 256] (configurable)
    - Fashion-MNIST subsample: [5k, 10k, 15k, 20k, 25k, 30k] for training (configurable)
    - Cross-entropy loss
    
    **Performance Testing:** After search convergence, the best architecture is retrained twice on independent 
    data splits to estimate mean and standard deviation of test accuracy.
    """)
    
    spacer(24)
    
    st.header("Configuration & Execution")
    
    spacer(12)
    
    # Parameter controls
    de_params = display_parameter_controls()
    
    spacer(8)
    
    train_config = display_training_config()
    
    spacer(8)
    
    fitness_weights = display_fitness_weights()
    
    spacer(12)
    
    # Update configs
    TRAINING_CONFIG.update({
        "epochs": train_config["epochs"],
        "learning_rate": train_config["learning_rate"],
        "batch_size": train_config["batch_size"],
    })
    
    DATASET_CONFIG["train_samples"] = train_config["train_samples"]
    
    FITNESS_WEIGHTS.update(fitness_weights)
    
    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.info(f"Using device: **{device}**")
    
    spacer(12)
    
    # Run button
    if st.button("Run Neural Architecture Search", type="primary", use_container_width=True):
        
        # Load data
        with st.spinner("Loading Fashion-MNIST dataset..."):
            train_loader, val_loader, test_loader = load_fashion_mnist(
                train_samples=DATASET_CONFIG["train_samples"],
                val_samples=DATASET_CONFIG["val_samples"],
                test_samples=DATASET_CONFIG["test_samples"],
                batch_size=TRAINING_CONFIG["batch_size"]
            )
        
        st.success(f"Loaded {DATASET_CONFIG['train_samples']} train, {DATASET_CONFIG['val_samples']} val, {DATASET_CONFIG['test_samples']} test samples")
        
        # Initialize DE solver
        solver = DifferentialEvolutionNAS(
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            population_size=de_params["population_size"],
            generations=de_params["generations"],
            mutation_factor=de_params["mutation_factor"],
            crossover_rate=de_params["crossover_rate"],
            seed=de_params["seed"]
        )
        
        # Progress tracking
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        
        def progress_callback(gen, best_fitness, mean_fitness, cache_size):
            progress = gen / de_params["generations"]
            progress_bar.progress(progress)
            status_text.text(
                f"Generation {gen}/{de_params['generations']}: "
                f"Best Fitness = {best_fitness:.4f}, "
                f"Mean Fitness = {mean_fitness:.4f}, "
                f"Cache Size = {cache_size}"
            )
        
        # Run DE search
        with st.spinner("Evolving CNN architectures..."):
            result = solver.run(progress_callback=progress_callback)
        
        progress_bar.empty()
        status_text.empty()
        
        st.success(f"Search completed! Best fitness: {result.best_fitness:.4f}")
        
        # Store in session state
        st.session_state["nas_result"] = result
        st.session_state["test_loader"] = test_loader
        st.session_state["train_val_loader"] = train_loader  # For final testing
    
    spacer(24)
    
    # Display results
    if "nas_result" in st.session_state:
        result = st.session_state["nas_result"]
        
        st.header("Results")
        
        # Architecture summary
        display_architecture_summary(result.best_architecture)
        
        spacer(12)
        
        # Metrics cards
        display_metrics_cards(result.best_metrics)
        
        spacer(12)
        
        # Plots
        col1, col2 = st.columns(2)
        
        with col1:
            fig_conv = plot_convergence(result.generation_details)
            st.plotly_chart(fig_conv, use_container_width=True)
            
            fig_div = plot_population_diversity(result.fitness_history)
            st.plotly_chart(fig_div, use_container_width=True)
        
        with col2:
            fig_train = plot_training_curves(result.best_metrics)
            st.plotly_chart(fig_train, use_container_width=True)
            
            fig_cache = plot_cache_efficiency(result.generation_details)
            st.plotly_chart(fig_cache, use_container_width=True)
        
        spacer(24)
        
        # Final testing
        st.header("Final Performance Testing")
        
        if st.button("Run Final Test Evaluation", type="secondary"):
            with st.spinner("Retraining best architecture on test set (2 runs)..."):
                # Train final model
                test_results = test_final_model(
                    arch_params=result.best_architecture,
                    train_loader=st.session_state["train_val_loader"],
                    test_loader=st.session_state["test_loader"],
                    device=device,
                    num_runs=2
                )
                
                # Build and train one final model for predictions
                final_model = CNNArchitecture(result.best_architecture)
                final_model = final_model.to(device)
                
                # Quick training (reuse existing data)
                from utils.anything_goes.functions import train_and_evaluate
                _ = train_and_evaluate(
                    final_model,
                    st.session_state["train_val_loader"],
                    st.session_state["test_loader"],
                    device,
                    epochs=TRAINING_CONFIG["epochs"],
                    lr=TRAINING_CONFIG["learning_rate"]
                )
                
                # Get predictions and examples
                predictions_data = get_predictions_and_examples(
                    final_model,
                    st.session_state["test_loader"],
                    device,
                    num_examples=16
                )
            
            st.session_state["test_results"] = test_results
            st.session_state["predictions_data"] = predictions_data
            st.session_state["final_model"] = final_model
        
        if "test_results" in st.session_state:
            display_test_results(st.session_state["test_results"])
            
            spacer(12)
            
            # Prediction examples
            if "predictions_data" in st.session_state:
                from utils.anything_goes.config import FASHION_MNIST_LABELS
                
                display_prediction_examples(
                    st.session_state["predictions_data"],
                    FASHION_MNIST_LABELS
                )
                
                spacer(12)
                
                # Confusion matrix
                st.markdown("### Confusion Matrix")
                st.markdown("""
                The confusion matrix shows how well the model distinguishes between different Fashion-MNIST classes.
                Diagonal elements represent correct predictions, while off-diagonal elements show misclassifications.
                """)
                
                fig_cm = plot_confusion_matrix(
                    st.session_state["predictions_data"],
                    FASHION_MNIST_LABELS
                )
                st.plotly_chart(fig_cm, use_container_width=True)

        
        spacer(24)
        
        st.header("Discussion")
        
        st.markdown("""
        ### Parameter Impact
        
        **Population Size:**
        - **Small (8-10)**: Faster iterations but risks premature convergence; may miss global optima.
        - **Large (16-20)**: Better exploration and diversity but significantly higher computational cost.
        
        **Generations:**
        - **Few (4-6)**: Quick results but may not fully converge; best for initial exploration.
        - **Many (10-15)**: Better convergence but marginal improvements per generation decrease over time.
        
        **Mutation Factor (F):**
        - **Low (0.3-0.5)**: Small mutations favor local refinement; useful in later stages.
        - **High (0.7-1.0)**: Bold mutations increase exploration; good for escaping local optima early.
        
        **Crossover Rate (CR):**
        - **Low (0.5-0.7)**: Preserves more of the target vector; conservative changes.
        - **High (0.8-1.0)**: Aggressive mixing with mutant; faster information propagation across population.
        
        ### Fitness Function Design
        
        The multi-objective fitness function with penalties serves several purposes:
        
        1. **Accuracy Primary:** The normalized accuracy term dominates the fitness, ensuring model quality is the top priority.
        2. **Parameter Penalty:** Discourages bloated models that overfit or consume excessive memory; promotes efficient architectures.
        3. **Time Penalty:** Favors faster-training models, which is crucial for practical deployment and iterative experimentation.
        
        The penalty thresholds were chosen to allow reasonable model complexity while avoiding computational excess. These can be tuned based on hardware constraints and deployment requirements.
        
        ### Conclusion
        
        Differential Evolution proves to be an effective and practical choice for Neural Architecture Search. Its simplicity, robustness, and efficient use of evaluations (via caching) 
        make it well-suited for optimizing CNN architectures on Fashion-MNIST. The resulting architectures demonstrate 
        strong validation performance while respecting computational budgets, validating the algorithm choice and 
        fitness function design.
        """)
    
    else:
        st.info("Configure parameters and click 'Run Neural Architecture Search' to begin optimization.")
