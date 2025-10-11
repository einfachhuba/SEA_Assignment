import numpy as np

rng = np.random.default_rng(42)

def parabola(x): 
    return - (x - 2)**2 + 4


def sinusoidal(x): 
    return np.sin(2 * x) + 0.3 * x