"""
Particle Swarm Optimization (PSO) Engine

This module implements local-best (lbest) PSO with ring topology.
The PSO algorithm operates in standardized coordinates [0,1]^d internally
and uses fitness functions that handle coordinate transformations.

Author: Converted from MATLAB code by Soumya D. Mohanty
"""

import numpy as np
import os
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
    seed_matrix: Optional[np.ndarray] = None,
    rand_file: str = "") -> Dict[str, Any]:
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
        rand_file: Optional path to a text file containing a single column of
               floats in [0, 1). When provided, random draws are consumed
               sequentially from this file instead of np.random; when empty
               (default), np.random is used. The file is assumed to have
               sufficient values for the entire PSO run.
    
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

    if rand_file:
        rand_seq = np.loadtxt(rand_file)
        rand_index = [0]

        def rand(shape):
            target_shape = shape if isinstance(shape, tuple) else (shape,)
            count = int(np.prod(target_shape))
            start = rand_index[0]
            end = start + count
            rand_index[0] = end
            # Always reshape in column-major order
            return rand_seq[start:end].reshape(target_shape, order='F')
    else:
        def rand(shape):
            target_shape = shape if isinstance(shape, tuple) else (shape,)
            flat = np.random.rand(int(np.prod(target_shape)))
            # Always reshape in column-major order
            return flat.reshape(target_shape, order='F')
    
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
    
    # Debug log file setup: uncomment this block to clear tmp.txt before a run
    # try:
    #     if os.path.exists('tmp.txt'):
    #         open('tmp.txt', 'w').close()
    # except Exception:
    #     # Silently ignore file I/O issues to not interrupt optimization
    #     pass

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
    n_cols_pop = n_dim * 4 + 4  # position, velocity, pbest, lbest_loc + 3 fitness values + inertia
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
    pop[:, coord_cols] = rand((pop_size, n_dim))
    
    # Seed initial locations if provided
    if seed_matrix is not None:
        n_row_seed, n_col_seed = seed_matrix.shape
        n_row_seed = min(n_row_seed, pop_size)
        n_col_seed = min(n_col_seed, n_dim)
        pop[:n_row_seed, :n_col_seed] = seed_matrix[:n_row_seed, :n_col_seed]
    
    # Initialize velocities
    pop[:, vel_cols] = rand((pop_size, n_dim)) - pop[:, coord_cols]
    
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
        # Debug logging: uncomment this block to write the swarm matrix to tmp.txt each step
        # try:
        #     with open('tmp.txt', 'a') as f:
        #         np.savetxt(f, pop, fmt='%.6f')
        #         # f.write('===================\n')
        # except Exception:
        #     # Silently ignore file I/O issues to not interrupt optimization
        #     pass
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
        
        # Update global best
        current_best_idx = np.argmin(pop[:, fit_current_col])
        current_best_fitness = pop[current_best_idx, fit_current_col]
        if current_best_fitness < gbest_val:
            gbest_val = current_best_fitness
            gbest_loc = pop[current_best_idx, coord_cols].copy()
        
        # Find local bests in ring topology neighborhoods
        for i in range(pop_size):
            # Define neighborhood indices (ring topology)
            nbr_indices = []
            for j in range(-left_nbrs, right_nbrs + 1):
                nbr_idx = (i + j) % pop_size
                nbr_indices.append(nbr_idx)
            
            # Find best in neighborhood (by current fitness)
            nbr_fitness = pop[nbr_indices, fit_current_col]
            best_nbr_idx = nbr_indices[np.argmin(nbr_fitness)]
            best_nbr_fit = pop[best_nbr_idx, fit_current_col]
            # Update local best only if it improves upon stored lbest
            if best_nbr_fit < pop[i, fit_lbest_col]:
                pop[i, lbest_cols] = pop[best_nbr_idx, coord_cols]
                pop[i, fit_lbest_col] = best_nbr_fit
        
       
        # Store history if requested
        if output_level >= 1:
            return_data['all_best_fit'][step] = gbest_val
        if output_level >= 2:
            return_data['all_best_loc'][step, :] = gbest_loc
        
        # Update inertia weight to match MATLAB schedule
        if end_inertia_iter is not None and end_inertia_iter > 1 and step < end_inertia_iter:
            inertia = max(
                start_inertia - ((start_inertia - end_inertia) / (end_inertia_iter - 1)) * step,
                end_inertia,
            )
        else:
            inertia = end_inertia
        pop[:, inertia_col] = inertia
        
        # Update velocities per particle (matching MATLAB code)
        for k in range(pop_size):
            chi1 = np.diag(rand(n_dim))
            chi2 = np.diag(rand(n_dim))
            
            cognitive = c1 * (pop[k, pbest_cols] - pop[k, coord_cols]) @ chi1
            social = c2 * (pop[k, lbest_cols] - pop[k, coord_cols]) @ chi2
            
            pop[k, vel_cols] = inertia * pop[k, vel_cols] + cognitive + social
            
            # Limit velocities
            pop[k, vel_cols] = np.clip(pop[k, vel_cols], -max_velocity, max_velocity)
        
        # Update positions
        pop[:, coord_cols] = pop[:, coord_cols] + pop[:, vel_cols]

        
    
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
