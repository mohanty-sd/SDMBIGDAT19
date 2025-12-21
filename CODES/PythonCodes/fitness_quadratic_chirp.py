"""
Quadratic chirp signal generation and fitness function for PSO regression

Author: Converted from MATLAB code by Soumya D. Mohanty
"""

import numpy as np
from typing import Dict, Tuple, Any, Optional
from pso import standard_to_real, check_standard_bounds


def generate_qc_signal(data_x: np.ndarray, snr: float, qc_coefs: np.ndarray) -> np.ndarray:
    """
    Generate a quadratic chirp signal.
    
    The quadratic chirp signal is defined as:
    s(t) = A * sin(2π(a1*t + a2*t² + a3*t³))
    
    where A is chosen such that the signal has the specified matched filtering SNR.
    
    Args:
        data_x: Time stamps at which to compute signal samples
        snr: Matched filtering signal-to-noise ratio
        qc_coefs: Coefficients [a1, a2, a3] parametrizing the phase
    
    Returns:
        Signal vector evaluated at data_x time stamps
    
    Example:
        >>> t = np.linspace(0, 1, 100)
        >>> coefs = np.array([10, -20, 15])
        >>> signal = generate_qc_signal(t, 10, coefs)
    """
    phase_vec = qc_coefs[0] * data_x + qc_coefs[1] * data_x**2 + qc_coefs[2] * data_x**3
    sig_vec = np.sin(2 * np.pi * phase_vec)
    
    # Normalize to specified SNR
    sig_norm = np.linalg.norm(sig_vec)
    if sig_norm > 0:
        sig_vec = snr * sig_vec / sig_norm
    
    return sig_vec


def qc_fitness(x_std: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fitness function for quadratic chirp regression using PSO.
    
    This fitness function computes the sum of squared residuals after
    maximizing over the amplitude parameter. The amplitude is determined
    analytically using matched filtering (inner product maximization).
    
    Args:
        x_std: Standardized coordinates (rows = particles, columns = dimensions)
               Each row represents [a1, a2, a3] in standardized form
        params: Dictionary containing:
               - 'rmin': Lower bounds for [a1, a2, a3]
               - 'rmax': Upper bounds for [a1, a2, a3]
               - 'dataY': Data vector (time series)
               - 'dataX': Time stamps
               - 'dataXSq': dataX**2 (precomputed for efficiency)
               - 'dataXCb': dataX**3 (precomputed for efficiency)
    
    Returns:
        Tuple of:
        - fitness_vals: Negative squared correlation (lower is better)
        - x_real: Real coordinates [a1, a2, a3]
    
    Notes:
        The fitness is computed as the negative of the squared correlation
        between the normalized signal template and the data. This formulation
        analytically maximizes over the amplitude parameter.
    """
    n_vecs = x_std.shape[0]
    
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
    
    # Compute fitness for valid points
    for i in range(n_vecs):
        if valid_pts[i]:
            x = x_real[i, :]
            fitness_vals[i] = _ssr_qc(x, params)
    
    return fitness_vals, x_real


def _ssr_qc(x: np.ndarray, params: Dict[str, Any]) -> float:
    """
    Sum of squared residuals after maximizing over amplitude parameter.
    
    Args:
        x: Real coordinates [a1, a2, a3]
        params: Parameter dictionary with data
    
    Returns:
        Negative squared correlation (fitness value)
    """
    # Generate normalized quadratic chirp
    phase_vec = x[0] * params['dataX'] + x[1] * params['dataXSq'] + x[2] * params['dataXCb']
    qc = np.sin(2 * np.pi * phase_vec)
    
    # Normalize
    qc_norm = np.linalg.norm(qc)
    if qc_norm > 0:
        qc = qc / qc_norm
    
    # Compute fitness (negative squared correlation)
    correlation = np.dot(params['dataY'], qc)
    ssr_val = -(correlation ** 2)
    
    return ssr_val


def generate_qc_data(data_x: np.ndarray, snr: float, qc_coefs: np.ndarray, 
                     noise_std: float = 1.0, random_state: Optional[int] = None) -> np.ndarray:
    """
    Generate noisy quadratic chirp data for testing.
    
    Args:
        data_x: Time stamps
        snr: Signal-to-noise ratio (matched filtering SNR)
        qc_coefs: True coefficients [a1, a2, a3]
        noise_std: Standard deviation of additive Gaussian noise
        random_state: Random seed for reproducibility
    
    Returns:
        Noisy data vector
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate clean signal
    signal = generate_qc_signal(data_x, snr, qc_coefs)
    
    # Add noise
    noise = noise_std * np.random.randn(len(data_x))
    data = signal + noise
    
    return data
