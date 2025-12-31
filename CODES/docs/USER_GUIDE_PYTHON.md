# User Guide for PSO Python Code

## Table of Contents
1. [Introduction](#introduction)
2. [Installation and Prerequisites](#installation-and-prerequisites)
3. [Quick Start](#quick-start)
4. [Core Functions](#core-functions)
5. [Test Scripts and Examples](#test-scripts-and-examples)
6. [Creating Custom Fitness Functions](#creating-custom-fitness-functions)
7. [Advanced Usage](#advanced-usage)
8. [Troubleshooting](#troubleshooting)
9. [References](#references)

## Introduction

This directory contains Python implementations of Particle Swarm Optimization (PSO) and its applications to illustrative statistical regression problems, namely:

- Quadratic chirp signal regression
- B-spline regression with PSO-optimized knots

and a benchmark fitness function (Generalized Rastrigin).

The code was converted from MATLAB implementations developed for the textbook ["Swarm Intelligence Methods for Statistical Regression"](https://www.amazon.com/Swarm-Intelligence-Methods-Statistical-Regression/dp/0367670372), which was based on courses delivered at the BigDat 2017 (Bari, Italy) and 2019 (Cambridge, UK) international winter schools.

### What is PSO?

Particle Swarm Optimization is a computational method that optimizes a problem by iteratively improving candidate solutions (particles) with regard to a given measure of quality (fitness function). The particles move through the search space influenced by their own best-known position and the best-known positions of their neighbors.

## Installation and Prerequisites

### Required Python Version
- Python 3.8 or later recommended

### Required Libraries
The following Python packages are required to run this code:

1. **NumPy** (>= 1.20.0) - For numerical computations
2. **SciPy** (>= 1.7.0) - For B-spline functions and scientific computing
3. **Matplotlib** (>= 3.3.0) - For visualization

### Installation Steps

1. Clone or download this repository
2. Navigate to the `CODES/PythonCodes` directory
3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Verify installation by running a test:
   ```bash
   python test_pso.py
   ```

## Quick Start

### Running Your First PSO Optimization

This example demonstrates how to run the PSO algorithm on a simple test function (Generalized Rastrigin) using default settings.

```python
import numpy as np
from pso import pso
from fitness_test import rastrigin_fitness

# Define search space dimensionality
n_dim = 5

# Define fitness function parameters
ff_params = {
    'rmin': np.full(n_dim, -5.0),
    'rmax': np.full(n_dim, 5.0)
}

# Create fitness function handle
def fit_func_handle(x):
    return rastrigin_fitness(x, ff_params)

# Run PSO with default settings
pso_out = pso(fit_func_handle, n_dim)

# Display results
print(f"Best fitness: {pso_out['best_fitness']:.6f}")
_, real_coord = fit_func_handle(pso_out['best_location'].reshape(1, -1))
print(f"Best location: {real_coord[0]}")
```

## Core Functions

### Naming Conventions

The Python code follows a modular structure with descriptive names:
- Core PSO engine: `pso.py`
- Fitness functions: `fitness_*.py`
- Application wrappers: `*_pso.py`
- Test scripts: `test_*.py`

### 1. `pso` - Main PSO Implementation

**Purpose**: Local-best (lbest) PSO minimizer with ring topology neighborhood.

**Design Features**: The PSO code operates in **standardized coordinates** [0,1]^d internally. Fitness functions handle conversion to **real coordinates** using helper functions `standard_to_real()` and `real_to_standard()`.

**Function Signature**:
```python
def pso(fitfunc, n_dim, pso_params=None, output_level=0, seed_matrix=None, rand_file=""):
    """
    Args:
        fitfunc: Fitness function that takes standardized coordinates and params
        n_dim: Dimensionality of search space
        pso_params: Optional dict of PSO parameters
        output_level: 0 (basic), 1 (+history), 2 (+locations)
        seed_matrix: Optional initial particle locations (standardized coords)
        rand_file: Optional path to text file with random numbers [0,1)
                  When provided, PSO uses these values instead of np.random
    
    Returns:
        Dict with 'best_location', 'best_fitness', 'total_func_evals',
        and optionally 'all_best_fit' and 'all_best_loc'
    """
```

**PSO Parameters** (all optional, defaults shown):
```python
pso_params = {
    'pop_size': 40,              # Number of particles
    'max_steps': 2000,           # Number of iterations
    'c1': 2,                     # Cognitive acceleration constant
    'c2': 2,                     # Social acceleration constant
    'max_velocity': 0.5,         # Maximum velocity per component
    'start_inertia': 0.9,        # Initial inertia weight
    'end_inertia': 0.4,          # Final inertia weight
    'end_inertia_iter': 2000,    # Iteration to reach end_inertia
    'boundary_cond': '',         # Boundary condition ('' = invisible wall)
    'nbrhd_sz': 3                # Neighborhood size (min 3)
}
```

**Output Dictionary**:
```python
result = {
    'best_location': np.ndarray,     # Best location (real coordinates)
    'best_fitness': float,           # Best fitness value
    'total_func_evals': int,         # Total fitness evaluations
    'all_best_fit': np.ndarray,      # (if output_level >= 1) Fitness history per iteration
    'all_best_loc': np.ndarray       # (if output_level >= 2) Best location per iteration
}
```

### `get_default_pso_params` - Default PSO Settings

**Purpose**: Retrieve the default PSO parameter dictionary.

**Function Signature**:
```python
def get_default_pso_params() -> Dict[str, Any]:
    """
    Returns:
        Dictionary with default PSO parameters
    """
```

**Usage Example**:
```python
from pso import get_default_pso_params

# Get defaults
defaults = get_default_pso_params()
print(defaults)  # Shows all default settings

# Modify just one parameter
params = get_default_pso_params()
params['max_steps'] = 5000
result = pso(fit_func, n_dim, params)
```

### 2. Helper Functions

**`standard_to_real(x_std, params)`**
- Converts standardized [0,1] coordinates to real coordinates
- `params` must contain 'rmin' and 'rmax' arrays

**`real_to_standard(x_real, params)`**
- Converts real coordinates to standardized [0,1] coordinates
- Handles special case when rmin == rmax

**`check_standard_bounds(x_std)`**
- Returns boolean array indicating which particles are within [0,1] bounds

### 3. `quadratic_chirp_pso` - Quadratic Chirp PSO

**Purpose**: Regression of quadratic chirp signals using PSO.

**Function Signature**:
```python
def quadratic_chirp_pso(in_params, pso_params=None, n_runs=1):
    """
    Args:
        in_params: Dict with 'dataY', 'dataX', 'dataXSq', 'dataXCb',
                  'rmin', 'rmax'
        pso_params: Optional PSO parameters
        n_runs: Number of independent PSO runs
    
    Returns:
        Dict with 'all_runs_output', 'best_run', 'best_fitness',
        'best_sig', 'best_qc_coefs'
    """
```

**Signal Model**: `s(t) = A * sin(2π(a1*t + a2*t² + a3*t³))`
- PSO optimizes phase coefficients [a1, a2, a3]
- Amplitude A is analytically maximized via matched filtering

**Example**:
```python
import numpy as np
from quadratic_chirp_pso import quadratic_chirp_pso
from fitness_quadratic_chirp import generate_qc_data

# Generate synthetic data
t = np.linspace(0, 1, 512)
true_coefs = np.array([10, 3, 3])
data = generate_qc_data(t, snr=10, qc_coefs=true_coefs, random_state=42)

# Setup parameters
in_params = {
    'dataY': data,
    'dataX': t,
    'dataXSq': t**2,
    'dataXCb': t**3,
    'rmin': np.array([0, -50, -50]),
    'rmax': np.array([50, 50, 50])
}

# Run PSO
result = quadratic_chirp_pso(in_params, n_runs=8)
print(f"Best coefficients: {result['best_qc_coefs']}")
```

### 4. `bspline_regression_pso` - B-spline Regression PSO

**Purpose**: B-spline regression with PSO-optimized knot locations.

**Function Signature**:
```python
def bspline_regression_pso(in_params, pso_params=None, n_runs=1):
    """
    Args:
        in_params: Dict with 'dataX', 'dataY', 'nBrks',
                  'rmin', 'rmax'
        pso_params: Optional PSO parameters
        n_runs: Number of independent PSO runs
    
    Returns:
        Dict with 'all_runs_output', 'best_run', 'best_fitness',
        'best_sig', 'best_brks'
    """
```

**Example**:
```python
import numpy as np
from bspline_regression_pso import bspline_regression_pso
from signal_generation import generate_bspline_data

# Generate synthetic data
t = np.linspace(0, 1, 512)
true_brks = np.array([0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
data, signal = generate_bspline_data(t, snr=10, brk_pts=true_brks, random_state=42)

# Setup parameters
in_params = {
    'dataY': data,
    'dataX': t,
    'nBrks': 10,
    'rmin': 0.0,
    'rmax': 1.0
}

# Run PSO
result = bspline_regression_pso(in_params, n_runs=4)
print(f"Best breakpoints: {result['best_brks']}")
```

### 5. `cardinal_bspline_fit` - Baseline B-spline Fit

**Purpose**: B-spline regression with uniformly spaced knots (non-optimized baseline).

**Function Signature**:
```python
def cardinal_bspline_fit(data_x, data_y, n_int_brks):
    """
    Args:
        data_x: Time stamps
        data_y: Data values
        n_int_brks: Number of uniformly spaced interior breakpoints
    
    Returns:
        Dict with 'est_sig', 'fit_val', 'brk_pts', 'bspl_coefs'
    """
```

## Test Scripts and Examples

### `test_pso.py`
Tests the core PSO engine on the Rastrigin benchmark function.
- Demonstrates default and custom PSO parameters (via `get_default_pso_params()`)
- Shows convergence plots
- Includes 2D trajectory visualization
- **Note**: Includes `test_pso_rand_file_test3()` which requires `random_numbers.txt` file. If this file is missing, the test will abort gracefully and display MATLAB code to generate it.

Run:
```bash
python test_pso.py
```

### `test_quadratic_chirp_pso.py`
Tests quadratic chirp signal regression.
- Generates synthetic chirp data and saves to `test_qc_synthetic_data.txt` for cross-checking with MATLAB
- Runs multiple PSO optimizations (8 independent runs)
- Compares estimated vs true signal
- Visualizes results and residuals

Run:
```bash
python test_quadratic_chirp_pso.py
```

### `test_bspline_regression_pso.py`
Tests B-spline regression with PSO-optimized knots.
- Compares PSO-optimized vs uniformly spaced knots
- Visualizes breakpoint locations
- Shows improvement over baseline

Run:
```bash
python test_bspline_regression_pso.py
```

## Creating Custom Fitness Functions

Custom fitness functions must follow this interface:

```python
def custom_fitness(x_std: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Custom fitness function for PSO.
    
    Args:
        x_std: Standardized coordinates (rows=particles, cols=dimensions)
               Values in [0, 1]
        params: Dictionary containing:
               - 'rmin': Lower bounds (required)
               - 'rmax': Upper bounds (required)
               - ... other problem-specific parameters
    
    Returns:
        Tuple of:
        - fitness_vals: Array of fitness values (lower is better)
        - x_real: Real coordinates (after conversion from standardized)
    """
    # 1. Check bounds
    valid_pts = check_standard_bounds(x_std)
    
    # 2. Convert to real coordinates
    x_real = x_std.copy()
    if np.any(valid_pts):
        x_real[valid_pts] = standard_to_real(x_std[valid_pts], params)
    
    # 3. Compute fitness
    fitness_vals = np.zeros(len(x_std))
    fitness_vals[~valid_pts] = np.inf  # Invalid points
    
    for i, x in enumerate(x_real):
        if valid_pts[i]:
            # YOUR FITNESS COMPUTATION HERE
            fitness_vals[i] = your_fitness_computation(x, params)
    
    return fitness_vals, x_real
```

### Example: Sphere Function

```python
def sphere_fitness(x_std, params):
    """Sphere function: f(x) = sum(x^2)"""
    valid_pts = check_standard_bounds(x_std)
    fitness_vals = np.zeros(len(x_std))
    fitness_vals[~valid_pts] = np.inf
    
    x_real = x_std.copy()
    if np.any(valid_pts):
        x_real[valid_pts] = standard_to_real(x_std[valid_pts], params)
    
    for i in range(len(x_std)):
        if valid_pts[i]:
            fitness_vals[i] = np.sum(x_real[i]**2)
    
    return fitness_vals, x_real

# Usage
params = {'rmin': np.array([-10, -10]), 'rmax': np.array([10, 10])}
fit_func = lambda x: sphere_fitness(x, params)
result = pso(fit_func, n_dim=2)
```

## Advanced Usage

### Multiple Independent Runs

Run PSO multiple times and select the best result (recommended for stochastic reliability):

```python
# Using application wrappers (automatically handles multiple runs)
result = quadratic_chirp_pso(in_params, pso_params, n_runs=8)

# Manual approach
best_fitness = np.inf
best_result = None

for run in range(8):
    np.random.seed(run + 1)  # Different seed per run
    result = pso(fit_func, n_dim)
    if result['best_fitness'] < best_fitness:
        best_fitness = result['best_fitness']
        best_result = result
```

### Custom PSO Parameters

Override default parameters for specific problems:

```python
# High-dimensional problem - increase iterations
pso_params = {
    'max_steps': 5000
}

result = pso(fit_func, n_dim, pso_params)
```

### Tracking Convergence

Use `output_level` to monitor optimization progress:

```python
import matplotlib.pyplot as plt

result = pso(fit_func, n_dim, output_level=2)

# Plot fitness history
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(result['all_best_fit'])
plt.xlabel('Iteration')
plt.ylabel('Best Fitness')
plt.title('Convergence History')

# Plot 2D trajectory (if n_dim == 2)
if n_dim == 2:
    plt.subplot(1, 2, 2)
    plt.plot(result['all_best_loc'][:, 0], result['all_best_loc'][:, 1], '.-')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Best Particle Trajectory')

plt.tight_layout()
plt.show()
```

### Seeding Initial Locations

Provide good starting points if available:

```python
# Create seed matrix (rows=particles, cols=dimensions)
seed_matrix = np.array([
    [0.5, 0.5, 0.5],  # Particle 1 at center
    [0.2, 0.3, 0.4],  # Particle 2 near suspected optimum
])

result = pso(fit_func, n_dim=3, seed_matrix=seed_matrix)
```

### Using File-Based Random Numbers (MATLAB Cross-Validation)

For reproducibility and cross-validation with MATLAB implementations, PSO can use random numbers from a file instead of NumPy's generator:

```python
# Generate random numbers in MATLAB using:
# rng('default')
# x = rand(30000*40*20*2+40*20*2,1);
# save('../PythonCodes/random_numbers.txt','x','-ascii');

# Then use in Python:
result = pso(fit_func, n_dim=20, rand_file='random_numbers.txt')
```

This allows the Python PSO to consume the exact same random sequence as the MATLAB version for numerical comparison.

## Troubleshooting

### Common Issues

**1. Import Errors**
```
ModuleNotFoundError: No module named 'numpy'
```
**Solution**: Install required packages:
```bash
pip install -r requirements.txt
```

### Debugging Tips

1. **Check fitness function**: Test on known inputs first
2. **Monitor convergence**: Use `output_level=1` to track progress
3. **Verify bounds**: Ensure `rmin` and `rmax` are appropriate
4. **Start simple**: Test on low dimensions before scaling up

## References

1. Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. *Proceedings of the IEEE International Conference on Neural Networks (ICNN’95)*, 4, 1942–1948.

2. Mohanty, S. D. (2021). *Swarm Intelligence Methods for Statistical Regression*. Chapman and Hall/CRC.

## Additional Resources

- **MATLAB Version**: See `CODES/MatlabCodes/` for original MATLAB implementation
- **MATLAB User Guide**: See `CODES/docs/USER_GUIDE_MATLAB.md`
- **Code Documentation**: See `CODES/docs/CodeList_MATLAB.pdf` for function descriptions
- **Course Materials**: BigDat 2017 and 2019 winter school materials

For questions or issues, please refer to the repository's issue tracker.
