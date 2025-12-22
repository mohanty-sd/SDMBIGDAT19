# Copilot Prompt: Reading Random Numbers from File for Reproducibility Testing

## Purpose

This prompt provides instructions for implementing functionality to read random number sequences from external text files in the Python PSO codebase. This feature is **critical for validation** — by using identical random number sequences in both MATLAB and Python implementations, we can verify that the translated Python code produces identical results to the original MATLAB code.

## Background

Random numbers are used in two key areas of the PSO codebase:

1. **PSO Algorithm** (`pso.py`):
   - Initial particle positions: `popsize × nDim` uniform random values in [0,1]
   - Initial particle velocities: `popsize × nDim` uniform random values in [0,1]
   - Velocity update coefficients: `2 × nDim` uniform random values per particle per iteration (chi1 and chi2)

2. **Data Generation Functions**:
   - `quadratic_chirp_data.py`: Gaussian noise for synthetic data
   - `bspline_data.py`: Gaussian noise for synthetic data

## Implementation Requirements

### 1. Random Number File Format

**Uniform Random Numbers** (for PSO):
- Plain text file with one random number per line
- Values should be in the range [0, 1]
- File should contain enough values for the entire PSO run:
  - Initial positions: `popsize × nDim`
  - Initial velocities: `popsize × nDim`
  - Velocity updates: `2 × popsize × nDim × maxSteps`
  - Total: `2 × popsize × nDim × (1 + maxSteps)`

**Gaussian Random Numbers** (for noise generation):
- Plain text file with one random number per line
- Standard normal distribution (mean=0, std=1)
- File should contain at least `nSamples` values

**Example file format** (`uniform_randoms.txt`):
```
0.8147236863931789
0.9057919370756192
0.1269868162935061
0.9133758561390194
0.6323592462254095
...
```

### 2. Function Modifications

#### A. PSO Engine (`pso.py`)

Add an optional parameter `rand_file_path` to the `pso()` function:

```python
def pso(fitfunc, n_dim, pso_params=None, output_level=0, seed_matrix=None, 
        rand_file_path=None):
    """
    PSO minimizer with optional random number file input.
    
    Parameters
    ----------
    ...
    rand_file_path : str, optional
        Path to text file containing uniform random numbers [0,1].
        If provided, random numbers are read from this file instead of 
        using numpy.random. This is used for reproducibility testing.
        The file must contain at least 2*popsize*n_dim*(1+max_steps) values.
    
    Notes
    -----
    When rand_file_path is provided, the file is read once at the start,
    and random numbers are consumed sequentially throughout the algorithm.
    """
```

**Implementation approach**:
- If `rand_file_path` is `None`, use `np.random.rand()` as normal
- If `rand_file_path` is provided:
  1. Load all random numbers from file into a numpy array at function start
  2. Create an iterator or index counter to track current position
  3. Replace each `np.random.rand(m, n)` call with slicing from the loaded array
  4. Advance the index/iterator by `m × n` after each use
  5. Optionally verify that enough random numbers are available

**Example implementation pattern**:
```python
if rand_file_path is not None:
    # Load random numbers from file
    rand_numbers = np.loadtxt(rand_file_path)
    
    # Validate sufficient numbers are available
    expected_count = 2 * popsize * n_dim * (1 + max_steps)
    if len(rand_numbers) < expected_count:
        raise ValueError(
            f"Random file contains {len(rand_numbers)} numbers but "
            f"{expected_count} are required for this PSO run"
        )
    
    rand_idx = 0
    
    def get_rand(shape):
        nonlocal rand_idx
        size = np.prod(shape)
        if rand_idx + size > len(rand_numbers):
            raise ValueError(
                f"Insufficient random numbers in file. Needed {rand_idx + size}, "
                f"but file only contains {len(rand_numbers)}"
            )
        result = rand_numbers[rand_idx:rand_idx+size].reshape(shape)
        rand_idx += size
        return result
else:
    def get_rand(shape):
        return np.random.rand(*shape)

# Then use get_rand() instead of np.random.rand() throughout
pop_positions = get_rand((popsize, n_dim))
pop_velocities = get_rand((popsize, n_dim))
# ... in velocity update loop:
chi1 = np.diag(get_rand((n_dim,)))
chi2 = np.diag(get_rand((n_dim,)))
```

#### B. Data Generation Functions

**For `generate_quadratic_chirp_data()` and `generate_bspline_data()`**:

Add an optional parameter `randn_file_path`:

```python
def generate_quadratic_chirp_data(data_x, snr, qc_coefs, randn_file_path=None):
    """
    Generate data containing a quadratic chirp signal.
    
    Parameters
    ----------
    ...
    randn_file_path : str, optional
        Path to text file containing standard normal random numbers.
        If provided, noise is read from this file instead of using 
        numpy.random.randn(). Used for reproducibility testing.
    
    Returns
    -------
    data_vec : ndarray
        Data containing signal plus noise
    sig_vec : ndarray
        Clean signal without noise
    """
    n_samples = len(data_x)
    sig_vec = generate_quadratic_chirp_signal(data_x, snr, qc_coefs)
    
    if randn_file_path is not None:
        noise_data = np.loadtxt(randn_file_path)
        if len(noise_data) < n_samples:
            raise ValueError(
                f"Noise file contains {len(noise_data)} values but "
                f"{n_samples} are required"
            )
        noise = noise_data[:n_samples]
    else:
        noise = np.random.randn(n_samples)
    
    data_vec = sig_vec + noise
    return data_vec, sig_vec
```

### 3. Test Script Modifications

Update test scripts (`test_pso.py`, `test_quadratic_chirp_pso.py`, `test_bspline_regression_pso.py`) to accept optional command-line arguments or configuration for random file paths:

```python
import argparse

def main():
    parser = argparse.ArgumentParser(description='Test PSO with optional random file')
    parser.add_argument('--rand-file', type=str, default=None,
                      help='Path to file with uniform random numbers for PSO')
    parser.add_argument('--randn-file', type=str, default=None,
                      help='Path to file with Gaussian random numbers for data generation')
    args = parser.parse_args()
    
    # ... existing test setup ...
    
    # Generate data with optional random file
    data_y, sig_y = generate_quadratic_chirp_data(
        data_x, snr, qc_coefs, randn_file_path=args.randn_file
    )
    
    # Run PSO with optional random file
    pso_out = pso(
        fit_func_handle, n_dim, pso_params, 
        rand_file_path=args.rand_file
    )
    
    # ... rest of test ...
```

### 4. Generating Random Number Files from MATLAB

To create test files with matching random sequences, add MATLAB utility scripts:

**`generate_pso_randoms.m`**:
```matlab
function generate_pso_randoms(filename, popsize, nDim, maxSteps)
% Generate random numbers for PSO and save to file
% Usage: generate_pso_randoms('pso_randoms.txt', 40, 5, 2000)

rng('default'); % Set seed for reproducibility

% Calculate total numbers needed
totalNumbers = 2 * popsize * nDim * (1 + maxSteps);

% Generate uniform random numbers
randNumbers = rand(totalNumbers, 1);

% Save to file
dlmwrite(filename, randNumbers, 'precision', '%.16f');

fprintf('Generated %d random numbers and saved to %s\n', totalNumbers, filename);
end
```

**`generate_noise_randoms.m`**:
```matlab
function generate_noise_randoms(filename, nSamples)
% Generate Gaussian random numbers for noise and save to file
% Usage: generate_noise_randoms('noise_randoms.txt', 512)

rng('default'); % Set seed for reproducibility

% Generate standard normal random numbers
randNumbers = randn(nSamples, 1);

% Save to file
dlmwrite(filename, randNumbers, 'precision', '%.16f');

fprintf('Generated %d Gaussian random numbers and saved to %s\n', nSamples, filename);
end
```

### 5. Validation Workflow

To verify Python code matches MATLAB output:

1. **Generate reference random files in MATLAB**:
   ```matlab
   generate_pso_randoms('pso_test_randoms.txt', 40, 5, 2000);
   generate_noise_randoms('noise_test_randoms.txt', 512);
   ```

2. **Run MATLAB code with default RNG**:
   ```matlab
   rng('default');
   % ... run test and save results ...
   ```

3. **Run Python code with same random files**:
   ```bash
   python test_pso.py --rand-file pso_test_randoms.txt
   python test_quadratic_chirp_pso.py \
       --rand-file pso_test_randoms.txt \
       --randn-file noise_test_randoms.txt
   ```

4. **Compare outputs**: Results should match to numerical precision limits.

### 6. Documentation Updates

Add a section to `USER_GUIDE_PYTHON.md`:

**"Reproducibility Testing and Validation"**
- Explain the purpose of random file functionality
- Document how to generate random files from MATLAB
- Provide examples of validation workflow
- Note that this is primarily for testing/validation, not typical usage

### 7. Important Considerations

**Order of random number consumption**:
- The order in which random numbers are consumed must exactly match MATLAB
- MATLAB's `rand(m, n)` generates numbers in **column-major order** (Fortran order)
- NumPy's default is **row-major order** (C order)
- When loading from file and reshaping, use order='F' to match MATLAB

**Example of handling order difference**:
```python
# MATLAB: rand(popsize, nDim) fills column-by-column
# To match in Python when reading from file:
rand_values = rand_numbers[rand_idx:rand_idx+popsize*n_dim]
# Reshape with Fortran order to match MATLAB's column-major layout
result = rand_values.reshape((popsize, n_dim), order='F')
rand_idx += popsize * n_dim
```

**Testing the order**:
To verify order matches MATLAB, create a small test:
```matlab
% MATLAB
rng('default');
A = rand(2, 3)
% Output: each column filled first
```

```python
# Python - incorrect (default C order)
np.random.seed(0)
A = np.random.rand(2, 3)  # Row-major, won't match MATLAB

# Python - correct (Fortran order when loading from file)
values = load_from_file()  # [v1, v2, v3, v4, v5, v6]
A = values.reshape((2, 3), order='F')  # Column-major, matches MATLAB
```

**Precision**:
- Save random numbers with sufficient precision (at least 16 decimal places)
- Use `%.16f` format in MATLAB's `dlmwrite()`
- Use `np.savetxt(..., fmt='%.16f')` in Python

**Error handling**:
- Verify file exists before attempting to read
- Check that file contains enough random numbers
- Provide clear error messages if file is malformed or insufficient

**Testing**:
- Include unit tests that verify random file reading works correctly
- Test edge cases (empty file, insufficient numbers, malformed data)
- Include at least one integration test comparing MATLAB and Python outputs

## Example Usage

```python
# Standard usage (default random generation)
pso_out = pso(fit_func, n_dim)

# Reproducibility testing mode (using random file)
pso_out = pso(fit_func, n_dim, rand_file_path='test_randoms.txt')

# Full validation test
data_y, sig_y = generate_quadratic_chirp_data(
    data_x, snr, [10, 3, 3], 
    randn_file_path='noise_randoms.txt'
)
fit_func = create_qc_fitness_function(data_y, data_x, params)
pso_out = pso(fit_func, 3, rand_file_path='pso_randoms.txt')
```

## Testing Checklist

- [ ] PSO runs successfully with and without random file
- [ ] Data generation works with and without random file
- [ ] File I/O errors are handled gracefully
- [ ] Random numbers are consumed in correct order
- [ ] Python results match MATLAB results when using same random file
- [ ] Documentation includes examples of validation workflow
- [ ] Command-line arguments work in test scripts

## Notes

- This functionality is **optional** and should not affect normal usage
- Random file mode is intended for **validation only**, not production use
- When `rand_file_path` is `None`, behavior should be identical to original implementation
- Consider adding a `verify_order=True` parameter that prints which random numbers are consumed when, to help debug order-of-consumption issues

---

**Implementation Recommendation**: Start with the PSO engine (`pso.py`) first, verify it works, then add to data generation functions, and finally update test scripts. This incremental approach makes debugging easier.
