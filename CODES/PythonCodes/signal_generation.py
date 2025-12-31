"""
B-spline signal generation utilities

Author: Converted from MATLAB code by Soumya D. Mohanty
"""

import numpy as np
from typing import Optional
from scipy.interpolate import BSpline


def generate_bspline_signal(data_x: np.ndarray, snr: float, brk_pts: np.ndarray) -> np.ndarray:
    """
    Generate a single B-spline signal.
    
    This generates a signal by evaluating a single cubic B-spline basis function.
    Uses breakpoints to create the knot vector with proper multiplicity.
    
    Args:
        data_x: Time stamps of data samples
        snr: Matched filtering signal-to-noise ratio
        brk_pts: Breakpoints defining the B-spline (at least 5 points for cubic)
    
    Returns:
        Signal vector with specified SNR
    
    Example:
        >>> t = np.linspace(0, 1, 100)
        >>> brk_pts = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        >>> signal = generate_bspline_signal(t, 10, brk_pts)
    """
    n_samples = len(data_x)
    
    # For cubic B-spline (degree 3), use the provided breakpoints as knots
    # Add multiplicity at endpoints if needed
    if len(brk_pts) < 5:
        return np.zeros(n_samples)
    
    degree = 3
    
    # Create knot vector with proper multiplicity (degree+1 at ends)
    # Take first 5 breakpoints for a single B-spline basis
    inner_knots = brk_pts[:5]
    knots = np.concatenate([
        [inner_knots[0]] * degree,  # Multiplicity at start
        inner_knots[1:-1],           # Interior knots
        [inner_knots[-1]] * degree   # Multiplicity at end
    ])
    
    # Create coefficient (just 1 for a single basis function)
    n_coefs = len(knots) - degree - 1
    if n_coefs <= 0:
        return np.zeros(n_samples)
    
    coefs = np.zeros(n_coefs)
    coefs[0] = 1.0  # Activate the first basis function
    
    # Create B-spline object
    bspl = BSpline(knots, coefs, degree, extrapolate=False)
    
    # Evaluate only within the valid range
    sig_vec = np.zeros(n_samples)
    valid_idx = (data_x >= inner_knots[0]) & (data_x <= inner_knots[-1])
    if np.any(valid_idx):
        sig_vec[valid_idx] = bspl(data_x[valid_idx])
        sig_vec = np.nan_to_num(sig_vec, nan=0.0)
    
    # Normalize to specified SNR
    sig_norm = np.linalg.norm(sig_vec)
    if sig_norm > 0:
        sig_vec = snr * sig_vec / sig_norm
    
    return sig_vec


def generate_bspline_data(data_x: np.ndarray, snr: float, brk_pts: np.ndarray,
                         noise_std: float = 1.0, random_state: Optional[int] = None) -> tuple:
    """
    Generate a data realization containing a single B-spline signal.
    
    The signal is embedded in white Gaussian noise.
    
    Args:
        data_x: Time stamps of data samples
        snr: Matched filtering signal-to-noise ratio
        brk_pts: Breakpoints defining the B-spline
        noise_std: Standard deviation of Gaussian noise (default: 1.0)
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (data, signal):
        - data: Data vector with signal + noise
        - signal: Clean signal vector
    
    Example:
        >>> t = np.linspace(0, 1, 100)
        >>> brk_pts = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        >>> data, signal = generate_bspline_data(t, 10, brk_pts, random_state=42)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate signal
    sig_vec = generate_bspline_signal(data_x, snr, brk_pts)
    
    # Add noise
    noise = noise_std * np.random.randn(len(data_x))
    data_y = sig_vec + noise
    
    return data_y, sig_vec
