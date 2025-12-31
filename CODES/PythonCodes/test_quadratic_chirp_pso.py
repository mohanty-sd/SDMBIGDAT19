"""
Test script for Quadratic Chirp PSO

Replicates test_crcbqcpso.m functionality.

Author: Converted from MATLAB code by Soumya D. Mohanty
"""

import numpy as np
import matplotlib.pyplot as plt
from fitness_quadratic_chirp import generate_qc_signal, generate_qc_data
from quadratic_chirp_pso import quadratic_chirp_pso


def test_quadratic_chirp_pso():
    """Test quadratic chirp regression using PSO."""
    print("="*60)
    print("Testing Quadratic Chirp PSO")
    print("="*60)
    
    # Generate synthetic data
    # Time samples
    n_samples = 512
    sampling_freq = 512
    time_vec = np.arange(n_samples) / sampling_freq
    
    # True signal parameters
    true_a1 = 10.0
    true_a2 = 3.0
    true_a3 = 3.0
    true_coefs = np.array([true_a1, true_a2, true_a3])
    
    print(f"\nTrue parameters: a1={true_a1}, a2={true_a2}, a3={true_a3}")
    
    # Generate signal with SNR=10
    signal_snr = 10.0
    np.random.seed(42)
    data = generate_qc_data(time_vec, signal_snr, true_coefs, noise_std=1.0)
    
    print(f"Generated data with {n_samples} samples")
    print(f"Signal SNR: {signal_snr}")
    
    # Save synthetic data to file for cross-checking in MATLAB
    data_filename = 'test_qc_synthetic_data.txt'
    np.savetxt(data_filename, data, fmt='%.10e')
    print(f"\nSynthetic data saved to: {data_filename}")
    print(f"This file can be used in ../MatlabCodes for cross-checking with MATLAB code.")
    
    # Setup PSO parameters
    in_params = {
        'dataY': data,
        'dataX': time_vec,
        'dataXSq': time_vec**2,
        'dataXCb': time_vec**3,
        'rmin': np.array([0.0, -50.0, -50.0]),
        'rmax': np.array([50.0, 50.0, 50.0])
    }
    
    # PSO settings (use reduced settings for faster testing)
    pso_params = {
        'pop_size': 40,
        'max_steps': 2000
    }
    
    # Run PSO with multiple independent runs
    n_runs = 8
    print(f"\nRunning PSO with {n_runs} independent runs...")
    print(f"Population size: {pso_params['pop_size']}")
    print(f"Max iterations: {pso_params['max_steps']}")
    
    results = quadratic_chirp_pso(in_params, pso_params, n_runs)
    
    # Display results
    print("\n" + "="*60)
    print("Results from all runs:")
    print("="*60)
    for i, run_result in enumerate(results['all_runs_output']):
        print(f"\nRun {i+1}:")
        print(f"  Fitness: {run_result['fit_val']:.6f}")
        print(f"  Coefficients: a1={run_result['qc_coefs'][0]:.4f}, "
              f"a2={run_result['qc_coefs'][1]:.4f}, a3={run_result['qc_coefs'][2]:.4f}")
        print(f"  Function evaluations: {run_result['total_func_evals']}")
    
    print("\n" + "="*60)
    print(f"Best run: {results['best_run'] + 1}")
    print(f"Best fitness: {results['best_fitness']:.6f}")
    print(f"Best coefficients: a1={results['best_qc_coefs'][0]:.4f}, "
          f"a2={results['best_qc_coefs'][1]:.4f}, a3={results['best_qc_coefs'][2]:.4f}")
    print(f"True coefficients: a1={true_a1:.4f}, a2={true_a2:.4f}, a3={true_a3:.4f}")
    print("="*60)
    
    # Compute errors
    est_coefs = results['best_qc_coefs']
    errors = est_coefs - true_coefs
    print(f"\nEstimation errors:")
    print(f"  Δa1 = {errors[0]:.4f}")
    print(f"  Δa2 = {errors[1]:.4f}")
    print(f"  Δa3 = {errors[2]:.4f}")
    
    # Plot results
    plot_results(time_vec, data, results, true_coefs, signal_snr)
    
    print("\n" + "="*60)
    print("Test completed successfully!")
    print("="*60)
    
    return results


def plot_results(time_vec, data, results, true_coefs, signal_snr):
    """Plot the data, true signal, and estimated signal."""
    # Generate true signal
    true_signal = generate_qc_signal(time_vec, signal_snr, true_coefs)
    
    # Get best estimated signal
    est_signal = results['best_sig']
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Data with true and estimated signals
    axes[0].plot(time_vec, data, 'k.', alpha=0.3, label='Data', markersize=2)
    axes[0].plot(time_vec, true_signal, 'b-', linewidth=2, label='True signal')
    axes[0].plot(time_vec, est_signal, 'r--', linewidth=2, label='Estimated signal')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Quadratic Chirp Signal Fitting')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    residuals = data - est_signal
    axes[1].plot(time_vec, residuals, 'k-', linewidth=0.5)
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=1)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Residual')
    axes[1].set_title('Residuals (Data - Estimated Signal)')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Fitness values from all runs
    fit_vals = [run['fit_val'] for run in results['all_runs_output']]
    axes[2].bar(range(1, len(fit_vals)+1), fit_vals, color='steelblue', alpha=0.7)
    axes[2].axhline(y=results['best_fitness'], color='r', linestyle='--', 
                    linewidth=2, label='Best fitness')
    axes[2].set_xlabel('Run number')
    axes[2].set_ylabel('Fitness value')
    axes[2].set_title('Fitness Values from All PSO Runs')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('test_qc_pso_results.png', dpi=150)
    print("\nSaved results plot to: test_qc_pso_results.png")
    
    # Create a zoomed-in view of the signal
    fig2, ax = plt.subplots(1, 1, figsize=(12, 6))
    zoom_samples = 100  # Show first 100 samples
    ax.plot(time_vec[:zoom_samples], data[:zoom_samples], 'k.', 
            alpha=0.5, label='Data', markersize=4)
    ax.plot(time_vec[:zoom_samples], true_signal[:zoom_samples], 'b-', 
            linewidth=2, label='True signal')
    ax.plot(time_vec[:zoom_samples], est_signal[:zoom_samples], 'r--', 
            linewidth=2, label='Estimated signal')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Quadratic Chirp Signal Fitting (Zoomed View)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_qc_pso_zoomed.png', dpi=150)
    print("Saved zoomed plot to: test_qc_pso_zoomed.png")


if __name__ == '__main__':
    test_quadratic_chirp_pso()
