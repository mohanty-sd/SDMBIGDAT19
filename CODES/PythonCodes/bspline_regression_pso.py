"""
B-spline Regression PSO Wrapper

PSO-based regression spline with optimized breakpoint locations.

Author: Converted from MATLAB code by Soumya D. Mohanty
"""

import numpy as np
from typing import Dict, Optional, Any
from pso import pso
from fitness_bspline import bspline_regression_fitness


def bspline_regression_pso(in_params: Dict[str, Any],
                          pso_params: Optional[Dict[str, Any]] = None,
                          n_runs: int = 1) -> Dict[str, Any]:
    """
    Regression spline with PSO-based breakpoint optimization.
    
    This function runs multiple independent PSO optimizations to find the best
    breakpoint locations for B-spline regression.
    
    Args:
        in_params: Input parameters dictionary with fields:
                  - 'dataY': Data vector
                  - 'dataX': Time stamps
                  - 'nBrks': Number of breakpoints to optimize
                  - 'rmin': Minimum value for standardized breakpoint params
                  - 'rmax': Maximum value for standardized breakpoint params
        pso_params: PSO parameters (None uses defaults)
        n_runs: Number of independent PSO runs
    
    Returns:
        Dictionary with keys:
        - 'all_runs_output': List of dicts, one per run, containing:
            - 'fit_val': Fitness value
            - 'brk_pts': Breakpoints
            - 'bspl_coeffs': B-spline coefficients
            - 'est_sig': Estimated signal
            - 'total_func_evals': Total fitness evaluations
        - 'best_run': Index of best run (0-based)
        - 'best_fitness': Best fitness value
        - 'best_sig': Best estimated signal
        - 'best_brks': Best breakpoints
    
    Example:
        >>> t = np.linspace(0, 1, 200)
        >>> data = np.sin(2 * np.pi * 3 * t) + 0.1 * np.random.randn(len(t))
        >>> params = {
        ...     'dataY': data,
        ...     'dataX': t,
        ...     'nBrks': 10,
        ...     'rmin': 0.0,
        ...     'rmax': 1.0
        ... }
        >>> result = bspline_regression_pso(params, n_runs=4)
    """
    data_x = in_params['dataX']
    data_y = in_params['dataY']
    n_brks_4_srch = in_params['nBrks']
    n_int_brks = n_brks_4_srch - 2
    n_bsplines = n_int_brks - 2
    rmin_val = in_params['rmin']
    rmax_val = in_params['rmax']
    
    n_samples = len(data_y)
    n_dim = n_brks_4_srch
    
    # Parameters for fitness function
    rmin = np.full(n_dim, rmin_val)
    rmax = np.full(n_dim, rmax_val)
    
    params = {
        'dataY': data_y,
        'dataX': data_x,
        'rmin': rmin,
        'rmax': rmax
    }
    
    # Create fitness function handle
    def fit_handle(x):
        return bspline_regression_fitness(x, params)
    
    # Storage for results from all runs
    all_runs_output = []
    
    # Run PSO multiple times independently
    for run_idx in range(n_runs):
        # Set random seed for reproducibility
        np.random.seed(run_idx + 1)
        
        # Run PSO
        pso_out = pso(fit_handle, n_dim, pso_params)
        
        # Extract results
        fit_val = pso_out['best_fitness']
        
        # Get breakpoints, coefficients, and estimated signal
        _, brk_pts, bspl_coeffs, est_sig = fit_handle(pso_out['best_location'].reshape(1, -1))
        
        # Store results for this run
        run_result = {
            'fit_val': fit_val,
            'brk_pts': brk_pts[0, :],
            'bspl_coeffs': bspl_coeffs[0, :],
            'est_sig': est_sig[0, :],
            'total_func_evals': pso_out['total_func_evals']
        }
        all_runs_output.append(run_result)
    
    # Find best run
    fit_vals = np.array([r['fit_val'] for r in all_runs_output])
    best_run = np.argmin(fit_vals)
    
    # Prepare output
    out_results = {
        'all_runs_output': all_runs_output,
        'best_run': best_run,
        'best_fitness': all_runs_output[best_run]['fit_val'],
        'best_sig': all_runs_output[best_run]['est_sig'],
        'best_brks': all_runs_output[best_run]['brk_pts']
    }
    
    return out_results
