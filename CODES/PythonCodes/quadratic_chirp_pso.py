"""
Quadratic Chirp PSO Wrapper

Regression of quadratic chirp signal using PSO optimization.

Author: Converted from MATLAB code by Soumya D. Mohanty
"""

import numpy as np
from typing import Dict, Optional, Any
from pso import pso
from fitness_quadratic_chirp import qc_fitness, generate_qc_signal


def quadratic_chirp_pso(in_params: Dict[str, Any], 
                        pso_params: Optional[Dict[str, Any]] = None,
                        n_runs: int = 1) -> Dict[str, Any]:
    """
    Regression of quadratic chirp signal using PSO.
    
    This function runs multiple independent PSO optimizations to find the best
    quadratic chirp parameters [a1, a2, a3] that fit the provided data.
    
    Args:
        in_params: Input parameters dictionary with fields:
                  - 'dataY': Data vector (time series)
                  - 'dataX': Time stamps
                  - 'dataXSq': dataX**2
                  - 'dataXCb': dataX**3
                  - 'rmin': Minimum values for [a1, a2, a3]
                  - 'rmax': Maximum values for [a1, a2, a3]
        pso_params: PSO parameters (None uses defaults)
        n_runs: Number of independent PSO runs
    
    Returns:
        Dictionary with keys:
        - 'all_runs_output': List of dicts, one per run, containing:
            - 'fit_val': Fitness value
            - 'qc_coefs': Coefficients [a1, a2, a3]
            - 'est_sig': Estimated signal
            - 'total_func_evals': Total fitness evaluations
        - 'best_run': Index of best run (0-based)
        - 'best_fitness': Best fitness value
        - 'best_sig': Best estimated signal
        - 'best_qc_coefs': Best coefficients [a1, a2, a3]
    
    Example:
        >>> t = np.linspace(0, 1, 100)
        >>> true_coefs = np.array([10, -20, 15])
        >>> data = generate_qc_data(t, 10, true_coefs)
        >>> params = {
        ...     'dataY': data,
        ...     'dataX': t,
        ...     'dataXSq': t**2,
        ...     'dataXCb': t**3,
        ...     'rmin': np.array([0, -50, -50]),
        ...     'rmax': np.array([50, 50, 50])
        ... }
        >>> result = quadratic_chirp_pso(params, n_runs=8)
    """
    n_samples = len(in_params['dataX'])
    n_dim = 3  # Three parameters: a1, a2, a3
    
    # Create fitness function handle with embedded parameters
    def fit_handle(x):
        return qc_fitness(x, in_params)
    
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
        
        # Get coefficients (need to call fitness function to get real coords)
        _, qc_coefs = fit_handle(pso_out['best_location'].reshape(1, -1))
        qc_coefs = qc_coefs[0, :]
        
        # Generate estimated signal
        est_sig = generate_qc_signal(in_params['dataX'], 1.0, qc_coefs)
        
        # Compute amplitude via matched filtering
        est_amp = np.dot(in_params['dataY'], est_sig)
        est_sig = est_amp * est_sig
        
        # Store results for this run
        run_result = {
            'fit_val': fit_val,
            'qc_coefs': qc_coefs,
            'est_sig': est_sig,
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
        'best_qc_coefs': all_runs_output[best_run]['qc_coefs']
    }
    
    return out_results
