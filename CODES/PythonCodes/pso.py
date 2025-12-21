"""
Particle Swarm Optimization (PSO) Engine

This module implements local-best (lbest) PSO with ring topology.
The PSO algorithm operates in standardized coordinates [0,1]^d internally
and uses fitness functions that handle coordinate transformations.

Author: Converted from MATLAB code by Soumya D. Mohanty
"""

import numpy as np
from typing import Callable, Dict, Optional, Any


def standard_to_real(x_std: np.ndarray, params: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Convert standardized coordinates to real coordinates using non-uniform range limits.
    
    Args:
        x_std: Standardized coordinates (rows = points, columns = dimensions)
               Values should be in [0, 1]
        params: Dictionary containing 'rmin' and 'rmax' arrays defining range limits
    
    Returns:
        Real coordinates corresponding to x_std
    
    Example:
        >>> params = {'rmin': np.array([-5, -5]), 'rmax': np.array([5, 5])}
        >>> x_std = np.array([[0.5, 0.5]])
        >>> x_real = standard_to_real(x_std, params)
    """
    rmin = params['rmin']
    rmax = params['rmax']
    rng_vec = rmax - rmin
    
    # Broadcast multiplication for vectorized operation
    r_vec = x_std * rng_vec + rmin
    return r_vec


def real_to_standard(x_real: np.ndarray, params: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Convert real coordinates to standardized coordinates.
    
    Args:
        x_real: Real coordinates (rows = points, columns = dimensions)
        params: Dictionary containing 'rmin' and 'rmax' arrays defining range limits
    
    Returns:
        Standardized coordinates in [0, 1]
    
    Note:
        If rmin == rmax for any coordinate, its standardized value is set to 0.
    """
    rmin = params['rmin']
    rmax = params['rmax']
    rng_vec = rmax - rmin
    
    # If rmin = rmax for any coordinate, set range to 1 to avoid division by zero
    rng_vec = np.where(rng_vec == 0, 1, rng_vec)
    
    x_std = (x_real - rmin) / rng_vec
    return x_std


def check_standard_bounds(x_std: np.ndarray) -> np.ndarray:
    """
    Check for points that are outside the standardized search range [0, 1].
    
    Args:
        x_std: Standardized coordinates (rows = points, columns = dimensions)
    
    Returns:
        Boolean array where True indicates valid points (all coordinates in [0, 1])
    
    Example:
        >>> x = np.array([[0.5, 0.5], [1.5, 0.5], [0.5, -0.1]])
        >>> valid = check_standard_bounds(x)
        >>> print(valid)  # [True, False, False]
    """
    valid_pts = np.all((x_std >= 0) & (x_std <= 1), axis=1)
    return valid_pts


def pso(fitfunc: Callable[[np.ndarray, Dict[str, Any]], np.ndarray],
        n_dim: int,
        pso_params: Optional[Dict[str, Any]] = None,
        output_level: int = 0,
        seed_matrix: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Local-best (lbest) PSO minimizer with ring topology.
    
    This function runs Particle Swarm Optimization on a fitness function.
    The PSO operates in standardized coordinates [0,1]^d internally.
    
    Args:
        fitfunc: Fitness function handle that accepts standardized coordinates
                 and a params dict, returns fitness values (lower is better)
        n_dim: Dimensionality of the search space
        pso_params: Optional dictionary to override default PSO parameters.
                   Supported fields:
                   - 'pop_size': Number of particles (default: 40)
                   - 'max_steps': Number of iterations (default: 2000)
                   - 'c1': Cognitive acceleration constant (default: 2)
                   - 'c2': Social acceleration constant (default: 2)
                   - 'max_velocity': Maximum velocity component (default: 0.5)
                   - 'start_inertia': Starting inertia weight (default: 0.9)
                   - 'end_inertia': Ending inertia weight (default: 0.4)
                   - 'end_inertia_iter': Iteration when end_inertia is reached (default: max_steps)
                   - 'boundary_cond': Boundary condition handling (default: '')
                   - 'nbrhd_sz': Neighborhood size for ring topology (default: 3, min: 3)
        output_level: Controls amount of information returned (default: 0)
                     0: Basic output (best_location, best_fitness, total_func_evals)
                     1: Adds 'all_best_fit' (fitness history)
                     2: Adds 'all_best_loc' (location history)
        seed_matrix: Optional array to seed initial particle locations (standardized coords)
    
    Returns:
        Dictionary with keys:
        - 'best_location': Best location found (real coordinates)
        - 'best_fitness': Best fitness value found
        - 'total_func_evals': Total number of fitness evaluations
        - 'all_best_fit': (if output_level >= 1) Best fitness per iteration
        - 'all_best_loc': (if output_level >= 2) Best location per iteration
    
    Example:
        >>> def rastrigin(x_std, params):
        ...     x_real = standard_to_real(x_std, params)
        ...     return np.sum(x_real**2 - 10*np.cos(2*np.pi*x_real) + 10, axis=1), x_real
        >>> params = {'rmin': np.array([-5]*5), 'rmax': np.array([5]*5)}
        >>> fitfunc_handle = lambda x: rastrigin(x, params)
        >>> result = pso(fitfunc_handle, 5)
    """
    # Default PSO parameters
    pop_size = 40
    max_steps = 2000
    c1 = 2
    c2 = 2
    max_velocity = 0.5
    start_inertia = 0.9
    end_inertia = 0.4
    end_inertia_iter = max_steps
    boundary_cond = ''
    nbrhd_sz = 3
    
    # Override defaults with user-provided parameters
    if pso_params is not None:
        pop_size = pso_params.get('pop_size', pop_size)
        max_steps = pso_params.get('max_steps', max_steps)
        c1 = pso_params.get('c1', c1)
        c2 = pso_params.get('c2', c2)
        max_velocity = pso_params.get('max_velocity', max_velocity)
        start_inertia = pso_params.get('start_inertia', start_inertia)
        end_inertia = pso_params.get('end_inertia', end_inertia)
        
        # If end_inertia_iter not specified and max_steps was changed, update it
        if 'end_inertia_iter' in pso_params:
            end_inertia_iter = pso_params['end_inertia_iter']
        elif 'max_steps' in pso_params:
            end_inertia_iter = max_steps
            
        boundary_cond = pso_params.get('boundary_cond', boundary_cond)
        nbrhd_sz = pso_params.get('nbrhd_sz', nbrhd_sz)
    
    # Ensure minimum neighborhood size
    nbrhd_sz = max(nbrhd_sz, 3)
    
    # Calculate neighborhood structure for ring topology
    left_nbrs = (nbrhd_sz - 1) // 2
    right_nbrs = nbrhd_sz - 1 - left_nbrs
    
    # Initialize output structure
    return_data = {
        'best_location': np.zeros(n_dim),
        'best_fitness': np.inf,
        'total_func_evals': 0
    }
    
    # Add optional output fields based on output_level
    if output_level >= 1:
        return_data['all_best_fit'] = np.zeros(max_steps)
    if output_level >= 2:
        return_data['all_best_loc'] = np.zeros((max_steps, n_dim))
    
    # Initialize particle swarm
    # Columns store: position, velocity, pbest, fitness_pbest, fitness_current,
    # fitness_lbest, inertia, local_best_location
    n_cols_pop = n_dim * 5 + 3  # position, velocity, pbest, lbest_loc + 3 fitness values + inertia
    pop = np.zeros((pop_size, n_cols_pop))
    
    # Define column indices
    coord_cols = slice(0, n_dim)
    vel_cols = slice(n_dim, 2*n_dim)
    pbest_cols = slice(2*n_dim, 3*n_dim)
    fit_pbest_col = 3*n_dim
    fit_current_col = 3*n_dim + 1
    fit_lbest_col = 3*n_dim + 2
    inertia_col = 3*n_dim + 3
    lbest_cols = slice(3*n_dim + 4, 4*n_dim + 4)
    
    # Initialize particle positions randomly in [0, 1]
    pop[:, coord_cols] = np.random.rand(pop_size, n_dim)
    
    # Seed initial locations if provided
    if seed_matrix is not None:
        n_row_seed, n_col_seed = seed_matrix.shape
        n_row_seed = min(n_row_seed, pop_size)
        n_col_seed = min(n_col_seed, n_dim)
        pop[:n_row_seed, :n_col_seed] = seed_matrix[:n_row_seed, :n_col_seed]
    
    # Initialize velocities
    pop[:, vel_cols] = np.random.rand(pop_size, n_dim) - pop[:, coord_cols]
    
    # Initialize personal bests
    pop[:, pbest_cols] = pop[:, coord_cols].copy()
    pop[:, fit_pbest_col] = np.inf
    pop[:, fit_current_col] = 0
    pop[:, fit_lbest_col] = np.inf
    pop[:, lbest_cols] = 0
    pop[:, inertia_col] = 0
    
    # Global best
    gbest_val = np.inf
    gbest_loc = 2 * np.ones(n_dim)
    
    # Fitness evaluation counter
    total_func_evals = 0
    
    # Main PSO loop
    for step in range(max_steps):
        # Evaluate fitness for all particles
        if boundary_cond == '':
            # Invisible wall boundary condition
            fitness_vals = fitfunc(pop[:, coord_cols])
            # If fitness function returns tuple (fitness, real_coords), extract fitness
            if isinstance(fitness_vals, tuple):
                fitness_vals = fitness_vals[0]
        else:
            # Special boundary condition (handled by fitness function)
            fitness_vals, _, pop[:, coord_cols] = fitfunc(pop[:, coord_cols])
        
        # Update fitness values
        pop[:, fit_current_col] = fitness_vals
        total_func_evals += pop_size
        
        # Update personal bests
        improved = fitness_vals < pop[:, fit_pbest_col]
        pop[improved, pbest_cols] = pop[improved, coord_cols]
        pop[improved, fit_pbest_col] = fitness_vals[improved]
        
        # Find local bests in ring topology neighborhoods
        for i in range(pop_size):
            # Define neighborhood indices (ring topology)
            nbr_indices = []
            for j in range(-left_nbrs, right_nbrs + 1):
                nbr_idx = (i + j) % pop_size
                nbr_indices.append(nbr_idx)
            
            # Find best in neighborhood
            nbr_fitness = pop[nbr_indices, fit_pbest_col]
            best_nbr_idx = nbr_indices[np.argmin(nbr_fitness)]
            pop[i, lbest_cols] = pop[best_nbr_idx, pbest_cols]
            pop[i, fit_lbest_col] = pop[best_nbr_idx, fit_pbest_col]
        
        # Update global best
        current_best_idx = np.argmin(pop[:, fit_pbest_col])
        current_best_fitness = pop[current_best_idx, fit_pbest_col]
        if current_best_fitness < gbest_val:
            gbest_val = current_best_fitness
            gbest_loc = pop[current_best_idx, pbest_cols].copy()
        
        # Store history if requested
        if output_level >= 1:
            return_data['all_best_fit'][step] = gbest_val
        if output_level >= 2:
            return_data['all_best_loc'][step, :] = gbest_loc
        
        # Update inertia weight (linear decrease)
        if step < end_inertia_iter:
            inertia = start_inertia - (start_inertia - end_inertia) * step / end_inertia_iter
        else:
            inertia = end_inertia
        pop[:, inertia_col] = inertia
        
        # Update velocities
        r1 = np.random.rand(pop_size, n_dim)
        r2 = np.random.rand(pop_size, n_dim)
        
        cognitive = c1 * r1 * (pop[:, pbest_cols] - pop[:, coord_cols])
        social = c2 * r2 * (pop[:, lbest_cols] - pop[:, coord_cols])
        
        pop[:, vel_cols] = (inertia * pop[:, vel_cols] + cognitive + social)
        
        # Limit velocities
        pop[:, vel_cols] = np.clip(pop[:, vel_cols], -max_velocity, max_velocity)
        
        # Update positions
        pop[:, coord_cols] = pop[:, coord_cols] + pop[:, vel_cols]
        
        # Handle boundary conditions (invisible wall)
        if boundary_cond == '':
            # Particles outside [0,1] get infinite fitness
            # They will be handled in the next fitness evaluation
            pass
    
    # Prepare final output
    return_data['best_fitness'] = gbest_val
    return_data['best_location'] = gbest_loc
    return_data['total_func_evals'] = total_func_evals
    
    return return_data


def get_default_pso_params() -> Dict[str, Any]:
    """
    Return a dictionary of default PSO parameters.
    
    Returns:
        Dictionary containing default PSO parameter values
    """
    return {
        'pop_size': 40,
        'max_steps': 2000,
        'c1': 2,
        'c2': 2,
        'max_velocity': 0.5,
        'start_inertia': 0.9,
        'end_inertia': 0.4,
        'end_inertia_iter': 2000,
        'boundary_cond': '',
        'nbrhd_sz': 3
    }
