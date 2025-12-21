"""
Benchmark test fitness function for PSO (Generalized Rastrigin function)

Author: Converted from MATLAB code by Soumya D. Mohanty
"""

import numpy as np
from typing import Dict, Tuple, Any
from pso import standard_to_real, check_standard_bounds


def rastrigin_fitness(x_std: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generalized Rastrigin benchmark fitness function for PSO testing.
    
    The Rastrigin function is a common test function for optimization algorithms.
    It has many local minima but a single global minimum at the origin.
    
    Args:
        x_std: Standardized coordinates (rows = particles, columns = dimensions)
               Values should be in [0, 1]
        params: Dictionary containing:
               - 'rmin': Lower bounds for each dimension (array)
               - 'rmax': Upper bounds for each dimension (array)
    
    Returns:
        Tuple of:
        - fitness_vals: Array of fitness values (lower is better)
        - x_real: Real coordinates corresponding to x_std
    
    Notes:
        - For standardized coordinates, fitness = infinity if any point falls
          outside [0, 1]
        - The function is computed as: sum(x^2 - 10*cos(2*pi*x) + 10)
        - Global minimum is at x = 0 for all dimensions
    
    Example:
        >>> params = {'rmin': np.array([-5, -5]), 'rmax': np.array([5, 5])}
        >>> x_std = np.array([[0.5, 0.5], [0.0, 0.0]])
        >>> fitness, x_real = rastrigin_fitness(x_std, params)
    """
    n_rows = x_std.shape[0]
    
    # Storage for fitness values
    fitness_vals = np.zeros(n_rows)
    
    # Check for out of bound coordinates
    valid_pts = check_standard_bounds(x_std)
    
    # Set fitness for invalid points to infinity
    fitness_vals[~valid_pts] = np.inf
    
    # Convert valid points to real coordinates
    x_real = x_std.copy()
    if np.any(valid_pts):
        x_real[valid_pts, :] = standard_to_real(x_std[valid_pts, :], params)
    
    # Compute fitness for valid points
    for i in range(n_rows):
        if valid_pts[i]:
            x = x_real[i, :]
            fitness_vals[i] = np.sum(x**2 - 10*np.cos(2*np.pi*x) + 10)
    
    return fitness_vals, x_real
