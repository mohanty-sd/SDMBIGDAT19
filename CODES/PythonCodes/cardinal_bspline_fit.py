"""
Cardinal B-spline Regression (Baseline with Uniformly Spaced Knots)

Spline fit using uniformly spaced breakpoints (non-optimized).

Author: Converted from MATLAB code by Soumya D. Mohanty
"""

import numpy as np
from typing import Dict
from scipy.interpolate import BSpline
from scipy.linalg import lstsq


def evaluate_bspline_basis(knots: np.ndarray, data_x: np.ndarray, 
                           degree: int = 3) -> np.ndarray:
    """
    Evaluate cubic B-spline basis functions at given points.
    
    Args:
        knots: Simple knot sequence (5 knots for one cubic B-spline)
        data_x: Points at which to evaluate the B-spline
        degree: Degree of B-spline (default: 3 for cubic)
    
    Returns:
        Array of B-spline values at data_x points
    """
    # Add multiplicity to knots for a single cubic B-spline basis
    if len(knots) < 5:
        return np.zeros_like(data_x)
    
    # Create proper knot vector with multiplicity at ends
    full_knots = np.concatenate([
        [knots[0]] * degree,  # Multiplicity at start
        knots[1:-1],           # Interior knots
        [knots[-1]] * degree   # Multiplicity at end
    ])
    
    # Number of basis functions
    n_basis = len(full_knots) - degree - 1
    if n_basis <= 0:
        return np.zeros_like(data_x)
    
    # Create coefficients with 1 at the first position
    coefs = np.zeros(n_basis)
    coefs[0] = 1.0
    
    # Create B-spline object
    bspl = BSpline(full_knots, coefs, degree, extrapolate=False)
    
    # Evaluate
    vals = bspl(data_x)
    vals = np.nan_to_num(vals, nan=0.0)  # Replace NaN with 0
    
    return vals


def cardinal_bspline_fit(data_x: np.ndarray, data_y: np.ndarray, 
                        n_int_brks: int) -> Dict[str, np.ndarray]:
    """
    Fit data with a linear combination of cardinal B-splines.
    
    Uses uniformly spaced interior breakpoints (no optimization).
    
    Args:
        data_x: Time stamps (values of independent variable)
        data_y: Data vector
        n_int_brks: Number of uniformly spaced interior breakpoints
                   (there will always be two breakpoints at data_x[0] and data_x[-1])
    
    Returns:
        Dictionary with keys:
        - 'est_sig': Estimated signal
        - 'fit_val': Fitness value (L2 norm of residual)
        - 'brk_pts': Breakpoints (including end breakpoints)
        - 'bspl_coefs': B-spline coefficients
    
    Raises:
        ValueError: If inconsistent data dimensions or insufficient knots
    
    Example:
        >>> t = np.linspace(0, 1, 100)
        >>> data = np.sin(2 * np.pi * 3 * t) + 0.1 * np.random.randn(len(t))
        >>> result = cardinal_bspline_fit(t, data, n_int_brks=8)
    """
    n_samples = len(data_x)
    smpl_intrvl = data_x[1] - data_x[0]
    
    # Sanity checks
    if len(data_y) != n_samples:
        raise ValueError('Inconsistent dataX and dataY dimensions')
    
    # Number of B-splines
    n_bsplines = n_int_brks - 2
    if n_bsplines < 1:
        raise ValueError('Increase the number of interior knots')
    
    # Generate uniformly spaced breakpoints
    brk_pt_gap = (data_x[-1] - data_x[0]) / (n_int_brks + 1)
    brk_pts = np.concatenate([
        [data_x[0]],
        data_x[0] + (np.arange(1, n_int_brks + 1) * brk_pt_gap),
        [data_x[-1]]
    ])
    
    # Preallocate storage
    b_vals = np.zeros((n_bsplines, n_samples))
    
    # Generate B-splines
    for i in range(n_bsplines):
        # Extract 5 breakpoints for this B-spline
        knots = brk_pts[i:i+5]
        
        # Determine valid range for evaluation
        strt_idx = max(0, min(int(np.ceil(knots[0] / smpl_intrvl)), n_samples - 1))
        stop_idx = max(0, min(int(np.floor(knots[-1] / smpl_intrvl)), n_samples - 1))
        
        if strt_idx <= stop_idx:
            # Evaluate B-spline at data points
            b_vals[i, strt_idx:stop_idx+1] = evaluate_bspline_basis(
                knots, data_x[strt_idx:stop_idx+1], degree=3
            )
    
    # Construct Gram matrix (transfer matrix)
    # The (i,j) element is sum_k b_i(t_k) * b_j(t_k)
    g_mat = b_vals @ b_vals.T
    
    # Source term
    f_mat = b_vals @ data_y
    
    # Solve for coefficients
    try:
        coeff_mat = np.linalg.solve(g_mat, f_mat)
    except np.linalg.LinAlgError:
        # If matrix is singular, use least squares
        coeff_mat, _, _, _ = lstsq(g_mat, f_mat)
    
    # Generate estimated signal
    est_sig = coeff_mat @ b_vals
    
    # Compute L2 norm of residual
    fit_val = np.linalg.norm(data_y - est_sig)
    
    # Output
    out_struct = {
        'est_sig': est_sig,
        'fit_val': fit_val,
        'brk_pts': brk_pts,
        'bspl_coefs': coeff_mat
    }
    
    return out_struct
