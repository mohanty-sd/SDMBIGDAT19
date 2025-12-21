"""
B-spline regression fitness function and utilities for PSO

Uses scipy.interpolate.BSpline for B-spline evaluation.

Author: Converted from MATLAB code by Soumya D. Mohanty
"""

import numpy as np
from typing import Dict, Tuple, Any
from scipy.interpolate import BSpline
from scipy.linalg import lstsq
from pso import standard_to_real, check_standard_bounds


def cl_scheme(tmin: float, scl_params: np.ndarray, tmax: float) -> np.ndarray:
    """
    Convert CL scheme parameters to real breakpoint locations.
    
    The CL (C. Leung) parameterization ensures breakpoints are confined to [tmin, tmax].
    
    Args:
        tmin: Minimum time boundary
        scl_params: Interior scale parameters (array)
        tmax: Maximum time boundary
    
    Returns:
        Array of breakpoint locations including tmin and tmax
    
    Note:
        This implements the parameterization from C. Leung that ensures
        breakpoints remain within bounds during optimization.
    """
    n_locs_in = len(scl_params)
    n_locs = n_locs_in + 2
    
    # Initialize time locations
    t_locs = np.zeros(n_locs)
    t_locs[0] = tmin
    t_locs[-1] = tmax
    
    # Build system of equations: fat_mat * t_locs = rhs_mat
    rhs_mat = np.zeros(n_locs)
    rhs_mat[0] = tmin
    rhs_mat[-1] = tmax
    
    fat_mat = -np.eye(n_locs)
    fat_mat[0, 0] = 1
    fat_mat[-1, -1] = 1
    
    # Fill off-diagonal elements
    for i in range(1, n_locs - 1):
        fat_mat[i, i-1] = 1 - scl_params[i-1]
        fat_mat[i, i+1] = scl_params[i-1]
    
    # Solve for breakpoint time coordinates
    t_locs = np.linalg.solve(fat_mat, rhs_mat)
    
    # Hard set endpoints to eliminate numerical errors
    t_locs[0] = tmin
    t_locs[-1] = tmax
    
    return t_locs


def heal_timestamps(tmin: float, tmax: float, min_gap_time: float, 
                    ck_t_vals: np.ndarray) -> np.ndarray:
    """
    Rectify time stamps to ensure minimum gap between consecutive values.
    
    Args:
        tmin: Minimum time boundary
        tmax: Maximum time boundary
        min_gap_time: Desired minimum gap between consecutive timestamps
        ck_t_vals: Time values to check and heal
    
    Returns:
        Healed time values with minimum gap enforced
    
    Raises:
        ValueError: If minimum gap is inconsistent with time boundaries
    """
    tol = 1e-4 * min_gap_time
    
    # Check if minimum gap is feasible
    if len(ck_t_vals) * min_gap_time > tmax - tmin:
        raise ValueError('Bad minimum gap: inconsistent with time boundaries')
    
    # Initial assumption is that time values are OK
    hld_t_vals = ck_t_vals.copy()
    
    # Check if healing is needed
    all_gaps = np.diff(np.concatenate([[tmin], hld_t_vals, [tmax]]))
    if not np.any(all_gaps < min_gap_time - tol):
        return hld_t_vals
    
    # Iteratively heal timestamps
    n_points = len(hld_t_vals)
    max_iter = 1000
    iter_count = 0
    
    while np.any(all_gaps < min_gap_time - tol) and iter_count < max_iter:
        # Distances to left and right neighbors
        ld = hld_t_vals - np.concatenate([[tmin], hld_t_vals[:-1]])
        rd = np.concatenate([hld_t_vals[1:], [tmax]]) - hld_t_vals
        
        # Limit distances to min_gap_time
        ld = np.minimum(ld, min_gap_time)
        rd = np.minimum(rd, min_gap_time)
        
        # Move to interior breakpoints that preserve gaps
        hld_t_vals = np.concatenate([[tmin], hld_t_vals, [tmax]])
        new_locs = hld_t_vals[:-2] + ld
        new_locs = np.minimum(new_locs, hld_t_vals[2:] - rd)
        hld_t_vals = new_locs
        
        # Recompute gaps
        all_gaps = np.diff(np.concatenate([[tmin], hld_t_vals, [tmax]]))
        iter_count += 1
    
    return hld_t_vals


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


def bspline_regression_fitness(x_std: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fitness function for B-spline regression with PSO-optimized breakpoints.
    
    For a given set of breakpoint locations (knots), this function:
    1. Converts standardized breakpoint parameters to real breakpoints
    2. Constructs B-spline basis functions
    3. Fits B-spline coefficients via least squares
    4. Returns residual sum of squares as fitness
    
    Args:
        x_std: Standardized coordinates (rows = particles, columns = breakpoint params)
        params: Dictionary containing:
               - 'rmin': Lower bounds for breakpoint params
               - 'rmax': Upper bounds for breakpoint params
               - 'dataX': Time stamps
               - 'dataY': Data values
    
    Returns:
        Tuple of:
        - fitness_vals: L2 norm of residuals (lower is better)
        - brk_pts: Real breakpoint locations
        - coeff_mat: B-spline coefficients
        - est_sig: Estimated signals
    
    Note:
        The number of B-splines = number of interior breakpoints - 2
    """
    n_vecs, n_brks = x_std.shape
    
    # Number of interior breakpoints (excluding endpoints)
    n_int_brks = n_brks - 2
    if n_int_brks < 3:
        raise ValueError('Number of interior breakpoints must be >= 3')
    
    # Number of B-splines
    n_bsplines = n_int_brks - 2
    
    # Storage for fitness values
    fitness_vals = np.zeros(n_vecs)
    
    # Check for out of bound coordinates
    valid_pts = check_standard_bounds(x_std)
    
    # Set fitness for invalid points to infinity
    fitness_vals[~valid_pts] = np.inf
    
    # Convert valid points to real coordinates
    x_real = x_std.copy()
    if np.any(valid_pts):
        x_real[valid_pts, :] = standard_to_real(x_std[valid_pts, :], params)
    
    # Data
    data_x = params['dataX']
    data_y = params['dataY']
    n_samples = len(data_x)
    smpl_intrvl = data_x[1] - data_x[0]
    
    # Storage for breakpoints and coefficients
    brk_pts = np.zeros((n_vecs, n_int_brks + 2))
    coeff_mat = np.zeros((n_vecs, n_bsplines))
    est_sig = np.zeros((n_vecs, n_samples))
    
    # For each particle (row in x_std)
    for i in range(n_vecs):
        if not valid_pts[i]:
            continue
        
        # Obtain breakpoints using CL scheme
        strt_brk_pt = (data_x[-1] - data_x[0]) * x_real[i, 0] + data_x[0]
        stop_brk_pt = (data_x[-1] - strt_brk_pt) * x_real[i, -1] + strt_brk_pt
        brk_pts[i, :] = cl_scheme(strt_brk_pt, x_real[i, 1:-1], stop_brk_pt)
        
        # Heal breakpoints if they are closer than sampling interval
        brk_pts[i, :] = heal_timestamps(data_x[0], data_x[-1], smpl_intrvl, brk_pts[i, :])
        
        # Generate B-spline basis functions
        b_vals = np.zeros((n_bsplines, n_samples))
        
        for j in range(n_bsplines):
            # Extract 5 breakpoints for this B-spline
            knots = brk_pts[i, j:j+5]
            
            # Determine valid range for evaluation
            strt_idx = max(0, min(int(np.ceil(knots[0] / smpl_intrvl)), n_samples - 1))
            stop_idx = max(0, min(int(np.floor(knots[-1] / smpl_intrvl)), n_samples - 1))
            
            if strt_idx <= stop_idx:
                # Evaluate B-spline at data points
                b_vals[j, strt_idx:stop_idx+1] = evaluate_bspline_basis(
                    knots, data_x[strt_idx:stop_idx+1], degree=3
                )
        
        # Construct Gram matrix (transfer matrix)
        g_mat = b_vals @ b_vals.T
        
        # Source term
        f_mat = b_vals @ data_y
        
        # Solve for coefficients using least squares
        try:
            coeff_mat[i, :] = np.linalg.solve(g_mat, f_mat)
        except np.linalg.LinAlgError:
            # If matrix is singular, use least squares
            coeff_mat[i, :], _, _, _ = lstsq(g_mat, f_mat)
        
        # Generate candidate signal
        est_sig[i, :] = coeff_mat[i, :] @ b_vals
        
        # Compute L2 norm of residual
        fitness_vals[i] = np.linalg.norm(data_y - est_sig[i, :])
    
    return fitness_vals, brk_pts, coeff_mat, est_sig
