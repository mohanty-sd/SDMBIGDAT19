# Python PSO Implementation

This directory contains Python implementations of Particle Swarm Optimization (PSO) algorithms and their applications to regression problems.

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run test scripts:
   ```bash
   python test_pso.py                      # Test core PSO on Rastrigin function
   python test_quadratic_chirp_pso.py       # Test quadratic chirp regression
   python test_bspline_regression_pso.py    # Test B-spline regression
   ```

## Files

### Core Module
- `pso.py` - Main PSO engine with local-best topology

### Fitness Functions
- `fitness_test.py` - Rastrigin benchmark function
- `fitness_quadratic_chirp.py` - Quadratic chirp signal fitting
- `fitness_bspline.py` - B-spline regression with optimized knots

### Application Wrappers
- `quadratic_chirp_pso.py` - Multi-run PSO for quadratic chirp regression
- `bspline_regression_pso.py` - Multi-run PSO for B-spline regression
- `cardinal_bspline_fit.py` - Baseline B-spline with uniform knots

### Utilities
- `signal_generation.py` - Generate synthetic B-spline signals

### Test Scripts
- `test_pso.py` - Core PSO tests
- `test_quadratic_chirp_pso.py` - Quadratic chirp tests
- `test_bspline_regression_pso.py` - B-spline regression tests

## Documentation

See `../docs/USER_GUIDE_PYTHON.md` for detailed documentation.

## Requirements

- Python 3.8+
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- Matplotlib >= 3.3.0

## Notes

This is a Python port of the MATLAB code in `../MatlabCodes/`. The Python implementation preserves the original functionality and algorithmic structure while using standard scientific Python libraries (NumPy, SciPy, Matplotlib).
