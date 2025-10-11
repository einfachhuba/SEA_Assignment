import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils.random_search.functions import parabola, sinusoidal
from utils.random_search.algorithms import random_search


st.set_page_config(page_title="TST: Random Search", layout="wide")
st.title("Random Search Algorithm")

st.header("Introduction")

st.markdown("""
**Random Search (RS)** is one of the simplest global optimization algorithms. It
explores the search space by evaluating randomly chosen candidate solutions and keeping
track of the best one found. RS does not require gradients, continuity, or any
assumptions about the objective function, which makes it applicable to a wide range of
problems. While extremely simple, it serves as a solid baseline to compare more advanced
algorithms such as Hill Climbing, Simulated Annealing, or Evolutionary Algorithms.
""")


cont = st.container(border=True)
with cont:
    st.badge("Main strengths:", color="blue")
    st.markdown("""
    - Simple and easy to implement
    - No assumptions about the function
    - Can serve as a baseline for comparison
    """)
    st.badge("Main weaknesses:", color="blue")
    st.markdown("""
    - inefficiency and poor scalability in high-dimensional spaces
    - No guarantee of finding the global optimum
    - Performance heavily dependent on the number of iterations
    """)


st.header("Methods")
st.markdown("""For doing a random search, we need to define a few parameters:
- **`fitness_function`**: The objective function to evaluate each candidate.
- **`bounds`**: The lower and upper limits of the search space.
- **`iterations`**: The number of random samples to generate.

""")

cont = st.container(border=True)
with cont:
    st.badge("Algorithm steps:", color="blue")
    st.markdown("""
    - Initialize the search bounds and the number of iterations.
    - Randomly sample a candidate solution within the bounds.
    - Evaluate its fitness using the provided fitness function.
    - Keep track of the best solution and its fitness value.
    - Repeat the sampling for the specified number of iterations.
    - Return the best candidate, its fitness, and the history of all evaluated candidates.
    """)


st.header("Results")

col1, col2 = st.columns(spec=[0.3,0.7])
with col1:
    #set up a control panel
    iterations = st.slider("Iterations", 1, 50, step=1)
    lower_bound = st.number_input("Lower Bound", -50.0, 0.0, -5.0)
    upper_bound = st.number_input("Upper Bound", 0.0, 50.0, 5.0)
    function_choice = st.selectbox("Objective Function", 
        ["Parabola", "Sinusoidal"])

    if function_choice == "Parabola":
        fitness_function = parabola
    else:
        fitness_function = sinusoidal

    best_candidate, best_fitness, history = random_search(
        fitness_function, 
        bounds=(lower_bound, upper_bound), 
        iterations=True,
        max_iterations=iterations)
    
    candidate_history, fitness_history = zip(*history)
    candidate_values = np.linspace(lower_bound, upper_bound, 400)
    fitness_values = [fitness_function(x) for x in candidate_values]

with col2:
    #plot everything
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(candidate_values, fitness_values, label="Objective Function", color='black')
    sc = ax.scatter(candidate_history, fitness_history, 
        c=np.linspace(0, 1, len(candidate_history)), cmap='Reds', s=30)
    ax.scatter(best_candidate, best_fitness, color="red", marker='*', s=200, label="Best Found")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title(f"Random Search on {function_choice}")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Iteration")
    ax.legend()
    st.pyplot(fig)



st.header("Discussion")
st.markdown("""Random Search is a **straightforward** and **easy-to-implement**
    optimization algorithm. It can be effective for **low-dimensional problems** or when
    the objective function is highly **irregular**. However, its performance degrades
    rapidly as the dimensionality increases. The choice of the **number of iterations**
    is crucial; too few iterations may lead to suboptimal solutions, while too many can
    be computationally expensive without significant gains. Overall, Random Search
    serves as a useful baseline for more sophisticated optimization techniques.
""")

cont = st.container(border=True)
with cont:
    st.badge("Key conclusions:", color="blue")
    st.markdown("""
    - **Expected results:** Probability of better solutions increases with iterations.
    - **Unexpected results:** Generally quite good performance observed even with limited
                iterations.
    - **Solution quality:** Depends on iterations and search space size; can be poor in high
                dimensions.
    - **Efficiency:** Very simple but inefficient; no exploitation of promising regions.
    - **Limitations:** No guarantee of global optimum, poor scalability.
    - **Possible improvements:** Adaptive sampling, hybrid with local search, increase
                iterations, narrow search bounds if prior knowledge exists.
    """)

