import numpy as np

def quadratic(x):
    """Quadratic function: f(x) = x^2"""
    return np.square(x)

def sinusoidal(x):
    """Sinusoidal function: f(x) = sin(x)"""
    return np.sin(x)

def ackley(x):
    """
    Ackley function - highly multimodal with many local optima.
    
    Global minimum at origin (0 for 1D, (0,0) for 2D) with f=0
    
    Args:
        x: scalar (1D) or array-like [x, y] (2D)
    
    Returns:
        Function value (for minimization)
    """
    # Convert to array if scalar
    x_arr = np.atleast_1d(x)
    
    a = 20
    b = 0.2
    c = 2 * np.pi
    n = len(x_arr)
    
    sum1 = -a * np.exp(-b * np.sqrt(np.sum(np.square(x_arr)) / n))
    sum2 = -np.exp(np.sum(np.cos(c * x_arr)) / n)
    result = sum1 + sum2 + a + np.exp(1)
    
    return float(result)

def rosenbrock(x):
    """
    Rosenbrock function - has a narrow valley leading to the global minimum.
    
    1D: Uses x as first dimension, assumes y=x (degenerate case)
    2D: Full formula with independent x and y
    
    Global minimum at (1,1) for 2D with f=0, or x=1 for 1D with f=0
    
    Args:
        x: scalar (1D) or array-like [x, y] (2D)
    
    Returns:
        Function value (for minimization)
    """
    # Convert to array if scalar
    x_arr = np.atleast_1d(x)
    
    if len(x_arr) == 1:
        result = 100.0 * np.square(x_arr[0] - np.square(x_arr[0])) + np.square(1 - x_arr[0])
    else:
        result = 100.0 * np.square(x_arr[1] - np.square(x_arr[0])) + np.square(1 - x_arr[0])
    
    return float(result)

def rastrigin(x):
    """
    Rastrigin function - highly multimodal with regularly distributed local minima.
    
    Global minimum at origin (0 for 1D, (0,0) for 2D) with f=0
    
    Args:
        x: scalar (1D) or array-like [x, y] (2D)
    
    Returns:
        Function value (for minimization)
    """
    # Convert to array if scalar
    x_arr = np.atleast_1d(x)
    
    n = len(x_arr)
    result = 10 * n + np.sum(np.square(x_arr) - 10 * np.cos(2 * np.pi * x_arr))
    
    return float(result)

