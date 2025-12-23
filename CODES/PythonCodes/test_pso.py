"""
Test script for PSO with Rastrigin function

Replicates test_crcbpso.m functionality.

Author: Converted from MATLAB code by Soumya D. Mohanty
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pso import pso, get_default_pso_params
from fitness_test import rastrigin_fitness


def test_pso_basic():
    """Test PSO with default settings on Rastrigin function."""
    print("="*60)
    print("Testing PSO with Rastrigin Function")
    print("="*60)
    
    # Problem setup
    n_dim = 20
    rmin = -10.0
    rmax = 10.0
    
    ff_params = {
        'rmin': np.full(n_dim, rmin),
        'rmax': np.full(n_dim, rmax)
    }
    
    # Create fitness function handle
    def fit_func_handle(x):
        return rastrigin_fitness(x, ff_params)
    
    # Display default PSO settings
    print("\nDefault PSO settings:")
    default_params = get_default_pso_params()
    for key, value in default_params.items():
        print(f"  {key}: {value}")
    
    # Test 1: Call PSO with default settings
    print("\n" + "="*60)
    print("Test 1: Calling PSO with default settings")
    print("="*60)
    np.random.seed(0)
    pso_out1 = pso(fit_func_handle, n_dim)
    
    _, real_coord = fit_func_handle(pso_out1['best_location'].reshape(1, -1))
    print(f"Best fitness: {pso_out1['best_fitness']:.6f}")
    print(f"Best location (first 5 coords): {real_coord[0, :5]}")
    print(f"Total function evaluations: {pso_out1['total_func_evals']}")
    
    # Test 2: Call PSO with output_level=2 for history
    print("\n" + "="*60)
    print("Test 2: Calling PSO with output_level=2")
    print("="*60)
    np.random.seed(0)
    pso_out2 = pso(fit_func_handle, n_dim, output_level=2)
    
    _, real_coord = fit_func_handle(pso_out2['best_location'].reshape(1, -1))
    print(f"Best fitness: {pso_out2['best_fitness']:.6f}")
    print(f"Best location (first 5 coords): {real_coord[0, :5]}")
    print(f"Total function evaluations: {pso_out2['total_func_evals']}")
    
    # Test 3: Override PSO parameters
    print("\n" + "="*60)
    print("Test 3: Overriding PSO parameters")
    print("="*60)
    pso_params = {
        'max_steps': 30000,
        'max_velocity': 0.9
    }
    print(f"Changing max_steps to: {pso_params['max_steps']}")
    print(f"Changing max_velocity to: {pso_params['max_velocity']}")
    
    np.random.seed(0)
    pso_out3 = pso(fit_func_handle, n_dim, pso_params, output_level=2)
    
    _, real_coord = fit_func_handle(pso_out3['best_location'].reshape(1, -1))
    print(f"Best fitness: {pso_out3['best_fitness']:.6f}")
    print(f"Best location (first 5 coords): {real_coord[0, :5]}")
    print(f"Total function evaluations: {pso_out3['total_func_evals']}")
    
    # Plot results
    print("\n" + "="*60)
    print("Generating plots...")
    print("="*60)
    
    # Plot convergence for default settings
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(pso_out2['all_best_fit'])
    plt.xlabel('Iteration number')
    plt.ylabel('Global best fitness')
    plt.title('Default PSO settings')
    plt.grid(True)
    
    # Plot convergence for non-default settings
    plt.subplot(1, 2, 2)
    plt.plot(pso_out3['all_best_fit'])
    plt.xlabel('Iteration number')
    plt.ylabel('Global best fitness')
    plt.title('Non-default PSO settings')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('test_pso_convergence.png', dpi=150)
    print("Saved convergence plot to: test_pso_convergence.png")
    
    # If 2D, plot trajectory
    if n_dim == 2:
        print("2D trajectory plot would be generated here (n_dim == 2)")
    
    print("\n" + "="*60)
    print("Test completed successfully!")
    print("="*60)


def test_pso_2d():
    """Test PSO with 2D Rastrigin for trajectory visualization."""
    print("\n" + "="*60)
    print("Testing PSO with 2D Rastrigin (for trajectory plot)")
    print("="*60)
    
    # Problem setup
    n_dim = 2
    rmin = -10.0
    rmax = 10.0
    
    ff_params = {
        'rmin': np.full(n_dim, rmin),
        'rmax': np.full(n_dim, rmax)
    }
    
    # Create fitness function handle
    def fit_func_handle(x):
        return rastrigin_fitness(x, ff_params)
    
    # Run PSO with history tracking
    pso_params = {
        'max_steps': 500,
        'pop_size': 20
    }
    np.random.seed(42)
    pso_out = pso(fit_func_handle, n_dim, pso_params, output_level=2)
    
    print(f"Best fitness: {pso_out['best_fitness']:.6f}")
    _, real_coord = fit_func_handle(pso_out['best_location'].reshape(1, -1))
    print(f"Best location: {real_coord[0]}")
    
    # Create contour plot with trajectory
    plt.figure(figsize=(8, 8))
    
    # Create grid for contour plot
    x_grid = np.linspace(rmin, rmax, 200)
    y_grid = np.linspace(rmin, rmax, 200)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Standardize coordinates
    X_std = (X - rmin) / (rmax - rmin)
    Y_std = (Y - rmin) / (rmax - rmin)
    
    # Compute fitness values
    points = np.column_stack([X_std.ravel(), Y_std.ravel()])
    fit_vals, _ = fit_func_handle(points)
    fit_vals = fit_vals.reshape(X.shape)
    
    # Plot contours
    plt.contour(X, Y, fit_vals, levels=30, cmap='viridis')
    
    # Plot trajectory of best particle
    trajectory_real = []
    for loc_std in pso_out['all_best_loc']:
        _, loc_real = fit_func_handle(loc_std.reshape(1, -1))
        trajectory_real.append(loc_real[0])
    trajectory_real = np.array(trajectory_real)
    
    plt.plot(trajectory_real[:, 0], trajectory_real[:, 1], 'r.-', linewidth=2, 
             markersize=4, label='Best particle trajectory')
    plt.plot(trajectory_real[0, 0], trajectory_real[0, 1], 'go', markersize=10, 
             label='Start')
    plt.plot(trajectory_real[-1, 0], trajectory_real[-1, 1], 'r*', markersize=15, 
             label='End')
    
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('PSO Trajectory on 2D Rastrigin Function')
    plt.legend()
    plt.colorbar(label='Fitness')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_pso_trajectory_2d.png', dpi=150)
    print("Saved trajectory plot to: test_pso_trajectory_2d.png")
    
    print("\n" + "="*60)
    print("2D Test completed successfully!")
    print("="*60)


def test_pso_rand_file():
    """Test PSO using random numbers loaded from a file."""
    print("\n" + "="*60)
    print("Testing PSO with random numbers from file")
    print("="*60)

    n_dim = 2
    rmin = -5.0
    rmax = 5.0

    ff_params = {
        'rmin': np.full(n_dim, rmin),
        'rmax': np.full(n_dim, rmax)
    }

    def fit_func_handle(x):
        return rastrigin_fitness(x, ff_params)

    pso_params = {
        'max_steps': 5,
        'pop_size': 4
    }

    rand_file_path = Path(__file__).parent / "random_numbers.txt"
    np.random.seed(123)
    np.savetxt(rand_file_path, np.random.rand(200))

    try:
        pso_out = pso(fit_func_handle, n_dim, pso_params, rand_file=str(rand_file_path))
        assert 'best_fitness' in pso_out
        assert pso_out['best_location'].shape == (n_dim,)
        print("PSO completed successfully using file-based random numbers")
    finally:
        if rand_file_path.exists():
            rand_file_path.unlink()


def test_pso_rand_file_test3():
    """Use existing random_numbers.txt with Test 3 PSO params."""
    print("\n" + "="*60)
    print("Test 3b: PSO with rand_file and overridden params")
    print("="*60)

    # Problem setup matching Test 3
    n_dim = 20
    rmin = -10.0
    rmax = 10.0

    ff_params = {
        'rmin': np.full(n_dim, rmin),
        'rmax': np.full(n_dim, rmax)
    }

    def fit_func_handle(x):
        return rastrigin_fitness(x, ff_params)

    # Overridden PSO parameters (same as Test 3)
    pso_params = {
        'max_steps': 3,
        'max_velocity': 0.9
    }

    # Use the pre-existing random_numbers.txt in the same folder
    rand_file_path = Path(__file__).parent / "random_numbers.txt"
    print(f"Using random file: {rand_file_path}")

    # Run PSO with file-backed random numbers
    pso_out = pso(fit_func_handle, n_dim, pso_params, output_level=2, rand_file=str(rand_file_path))

    # Report summary
    print(f"Best fitness: {pso_out['best_fitness']:.6f}")
    _, real_coord = fit_func_handle(pso_out['best_location'].reshape(1, -1))
    print(f"Best location (first 5 coords): {real_coord[0, :5]}")
    print(f"Total function evaluations: {pso_out['total_func_evals']}")


if __name__ == '__main__':
    # Run basic tests
    #test_pso_basic()
    
    # Run 2D test for trajectory visualization
    #test_pso_2d()

    # Run file-based random number test
    # test_pso_rand_file()
    
    # Run Test 3 variant using the existing random_numbers.txt
    test_pso_rand_file_test3()
    
    print("\nAll tests completed!")
