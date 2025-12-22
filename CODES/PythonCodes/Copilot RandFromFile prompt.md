You are to modify the Python implementation of the PSO algorithm (`pso.py`) to optionally read its random numbers from an external text file instead of using `np.random.rand`. This feature is intended for reproducibility and testing with pre-generated random sequences.

#### Requirements

1. **Function Signature Update**  
   Add a new optional parameter `rand_file: str = ""` to the `pso` function.  
   - If `rand_file == ""` (default), use `np.random.rand` as before.  
   - If `rand_file` is a non-empty string, treat it as a path to a plain-text file containing a **single column** of floating-point numbers in `[0, 1)`, one per line.

2. **Random Number Consumption**  
   - Replace **every** call to `np.random.rand(*shape)` in `pso.py` with a call to a helper function that draws the required number of values from either:
     - The internal NumPy RNG (if `rand_file == ""`), or  
     - A preloaded list from the file (if `rand_file != ""`).
   - The file is assumed to contain **sufficient** random numbers for the entire PSO run (given `pop_size`, `n_dim`, `max_steps`, etc.). **No runtime validation is required.**
   - The sequence must be consumed **in order**, matching exactly how `np.random.rand` would have been called (e.g., first for initial positions, then velocities, then per-iteration velocity updates).

3. **Implementation Strategy**  
   - At the start of `pso`, if `rand_file` is provided:
     - Load the entire file into a 1D NumPy array (e.g., `rand_seq = np.loadtxt(rand_file)`).
     - Use a **mutable index counter** (e.g., a list `[0]` or a simple class) to track the next unused random number.
     - Create a local function, e.g. `def rand(shape):`, that slices `rand_seq` starting at the current index, advances the index by `np.prod(shape)`, and reshapes the slice to `shape`.
   - If `rand_file == ""`, `rand(shape)` simply returns `np.random.rand(*shape)`.

4. **Documentation Update**  
   - Update the docstring of `pso` to document the new `rand_file` parameter, its behavior, format expectations, and default value.

5. **Test Code Update**  
   - Modify `test_pso.py` to include a new test case:
     - Create or assume a file `random_numbers.txt` with enough values.
     - Call `pso(..., rand_file="random_numbers.txt")`.
     - Verify that the function runs without error (exact output validation is optional, but execution must succeed).
   - Keep existing tests intact.

6. **Assumptions**  
   - The user guarantees the file exists and has enough numbers.
   - No error handling for file I/O or insufficient length is required.

Please confirm understanding before generating code.