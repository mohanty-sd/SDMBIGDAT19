"""
Test script for B-spline Regression PSO

Tests both PSO-optimized and cardinal (uniform) B-spline regression.

Author: Converted from MATLAB code by Soumya D. Mohanty
"""

import numpy as np
import matplotlib.pyplot as plt
from signal_generation import generate_bspline_data
from bspline_regression_pso import bspline_regression_pso
from cardinal_bspline_fit import cardinal_bspline_fit


def test_bspline_regression():
    """Test B-spline regression with PSO-optimized vs uniform knots."""
    print("="*60)
    print("Testing B-spline Regression PSO")
    print("="*60)
    
    # Generate synthetic data with a B-spline signal
    n_samples = 512
    sampling_freq = 512
    time_vec = np.arange(n_samples) / sampling_freq
    
    # Define true breakpoints for signal generation
    true_brk_pts = np.array([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    
    print(f"\nTrue breakpoints: {true_brk_pts}")
    
    # Generate signal with SNR=10
    signal_snr = 10.0
    np.random.seed(42)
    data, true_signal = generate_bspline_data(time_vec, signal_snr, true_brk_pts, 
                                              noise_std=1.0)
    
    print(f"Generated data with {n_samples} samples")
    print(f"Signal SNR: {signal_snr}")
    
    # Setup for B-spline regression
    n_brks = 10  # Number of breakpoints to optimize (including endpoints)
    n_int_brks = n_brks - 2  # Interior breakpoints
    
    print(f"\nFitting with {n_brks} breakpoints ({n_int_brks} interior)")
    
    # Test 1: Cardinal B-spline with uniform knots (baseline)
    print("\n" + "="*60)
    print("Test 1: Cardinal B-spline (uniform knots)")
    print("="*60)
    
    cardinal_result = cardinal_bspline_fit(time_vec, data, n_int_brks)
    
    print(f"Fitness (residual norm): {cardinal_result['fit_val']:.6f}")
    print(f"Number of B-spline coefficients: {len(cardinal_result['bspl_coefs'])}")
    print(f"Breakpoints: {cardinal_result['brk_pts']}")
    
    # Test 2: PSO-optimized breakpoints
    print("\n" + "="*60)
    print("Test 2: PSO-optimized B-spline")
    print("="*60)
    
    # Setup PSO parameters
    in_params = {
        'dataY': data,
        'dataX': time_vec,
        'nBrks': n_brks,
        'rmin': 0.0,
        'rmax': 1.0
    }
    
    # PSO settings (use reduced settings for faster testing)
    pso_params = {
        'pop_size': 40,
        'max_steps': 1000
    }
    
    # Run PSO with multiple independent runs
    n_runs = 4
    print(f"Running PSO with {n_runs} independent runs...")
    print(f"Population size: {pso_params['pop_size']}")
    print(f"Max iterations: {pso_params['max_steps']}")
    
    pso_result = bspline_regression_pso(in_params, pso_params, n_runs)
    
    # Display results
    print("\n" + "="*60)
    print("Results from all PSO runs:")
    print("="*60)
    for i, run_result in enumerate(pso_result['all_runs_output']):
        print(f"\nRun {i+1}:")
        print(f"  Fitness: {run_result['fit_val']:.6f}")
        print(f"  Breakpoints: {run_result['brk_pts']}")
        print(f"  Function evaluations: {run_result['total_func_evals']}")
    
    print("\n" + "="*60)
    print(f"Best run: {pso_result['best_run'] + 1}")
    print(f"Best fitness: {pso_result['best_fitness']:.6f}")
    print(f"Best breakpoints: {pso_result['best_brks']}")
    print("="*60)
    
    # Compare results
    print("\n" + "="*60)
    print("Comparison:")
    print("="*60)
    print(f"Cardinal B-spline fitness: {cardinal_result['fit_val']:.6f}")
    print(f"PSO-optimized fitness:     {pso_result['best_fitness']:.6f}")
    improvement = ((cardinal_result['fit_val'] - pso_result['best_fitness']) / 
                   cardinal_result['fit_val'] * 100)
    print(f"Improvement: {improvement:.2f}%")
    
    # Plot results
    plot_results(time_vec, data, true_signal, cardinal_result, pso_result, true_brk_pts)
    
    print("\n" + "="*60)
    print("Test completed successfully!")
    print("="*60)
    
    return cardinal_result, pso_result


def plot_results(time_vec, data, true_signal, cardinal_result, pso_result, true_brk_pts):
    """Plot comparison of cardinal and PSO-optimized B-spline fits."""
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Plot 1: Data with fits
    axes[0].plot(time_vec, data, 'k.', alpha=0.2, label='Data', markersize=1)
    axes[0].plot(time_vec, true_signal, 'b-', linewidth=2, label='True signal', alpha=0.7)
    axes[0].plot(time_vec, cardinal_result['est_sig'], 'g--', linewidth=2, 
                label='Cardinal B-spline')
    axes[0].plot(time_vec, pso_result['best_sig'], 'r-', linewidth=2, 
                label='PSO-optimized B-spline')
    
    # Plot true breakpoints
    for brk in true_brk_pts:
        axes[0].axvline(x=brk, color='b', linestyle=':', alpha=0.3, linewidth=1)
    
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('B-spline Regression: Comparison of Methods')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Residuals comparison
    cardinal_residuals = data - cardinal_result['est_sig']
    pso_residuals = data - pso_result['best_sig']
    
    axes[1].plot(time_vec, cardinal_residuals, 'g-', linewidth=1, alpha=0.7,
                label=f'Cardinal (norm={cardinal_result["fit_val"]:.2f})')
    axes[1].plot(time_vec, pso_residuals, 'r-', linewidth=1, alpha=0.7,
                label=f'PSO-optimized (norm={pso_result["best_fitness"]:.2f})')
    axes[1].axhline(y=0, color='k', linestyle='--', linewidth=1)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Residual')
    axes[1].set_title('Residuals Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Breakpoint locations
    axes[2].scatter(cardinal_result['brk_pts'], 
                   np.ones_like(cardinal_result['brk_pts']), 
                   s=100, marker='o', c='green', alpha=0.6, label='Cardinal knots')
    axes[2].scatter(pso_result['best_brks'], 
                   np.zeros_like(pso_result['best_brks']), 
                   s=100, marker='s', c='red', alpha=0.6, label='PSO-optimized knots')
    axes[2].scatter(true_brk_pts, 
                   -np.ones_like(true_brk_pts), 
                   s=100, marker='^', c='blue', alpha=0.6, label='True signal knots')
    
    axes[2].set_xlabel('Time (s)')
    axes[2].set_yticks([-1, 0, 1])
    axes[2].set_yticklabels(['True', 'PSO', 'Cardinal'])
    axes[2].set_title('Breakpoint Locations')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='x')
    axes[2].set_xlim([time_vec[0], time_vec[-1]])
    
    plt.tight_layout()
    plt.savefig('/home/runner/work/SDMBIGDAT19/SDMBIGDAT19/CODES/PythonCodes/test_bspline_results.png', 
                dpi=150)
    print("\nSaved results plot to: test_bspline_results.png")
    
    # Create zoomed-in view
    fig2, ax = plt.subplots(1, 1, figsize=(14, 6))
    zoom_samples = 150
    ax.plot(time_vec[:zoom_samples], data[:zoom_samples], 'k.', 
            alpha=0.3, label='Data', markersize=3)
    ax.plot(time_vec[:zoom_samples], true_signal[:zoom_samples], 'b-', 
            linewidth=2, label='True signal', alpha=0.7)
    ax.plot(time_vec[:zoom_samples], cardinal_result['est_sig'][:zoom_samples], 
            'g--', linewidth=2, label='Cardinal B-spline')
    ax.plot(time_vec[:zoom_samples], pso_result['best_sig'][:zoom_samples], 
            'r-', linewidth=2, label='PSO-optimized B-spline')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('B-spline Regression (Zoomed View)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/runner/work/SDMBIGDAT19/SDMBIGDAT19/CODES/PythonCodes/test_bspline_zoomed.png', 
                dpi=150)
    print("Saved zoomed plot to: test_bspline_zoomed.png")


if __name__ == '__main__':
    test_bspline_regression()
