# Project Overview
Swing_Test is the implementation, debug and benchmarking project for [Swing algorithm](https://arxiv.org/abs/2401.09356).

It is a modular project featuring a static library (`libswing.a`), test and benchmark executables (`bin/`), and SLURM-based scripts for benchmarking and debugging on clusters. Future updates will include data analysis and graphing tools.

## Table of Contents
- [Project Structure and Components](#project-structure-and-components)
- [libswing - Static Library](#libswing---static-library)
- [Configuration Files](#config---algorithms-and-test-configurations)
- [Test/Benchmark Program](#test---benchmark-program)
- [OMPI Rules and Scripts](#ompi_rules---open-mpi-rule-file-generator)
- [Running the Tests](#scripts---run-the-tests)
- [Results Management](#results-management)
- [Building](#building)
- [Data Analysis and Visualization](#data-analysis-and-visualization)
- [TODO](#todo)

--- 

## Project Structure and Components

```
.
├── config                # .json config file, test file and test parser script
├── include               # Header file for libswing library containing the functions' signatures
├── libswing              # Libswing library source code
├── ompi_rules            # Open MPI dynamic rule file and script to modify it
├── plot                  # Python scripts for data analysis and visualization (in development)
├── results               # Results folder divided by system
├── scripts               # Main scripts to run benchmarks and debug
│   ├── environments          # Environment specific scripts
│   └── submit_wrapper.sh     # Wrapper to launch tests with `SBATCH` or run locally
├── test                  # Test program source code, includes benchmarking and debugging
├── Makefile              # Top-level Makefile
└── README.md             # This documentation
```

---

## `libswing` - Static Library

The `libswing/` directory contains the source code for the `libswing/` static library. This library is compiled into a static archive (`lib/libswing.a`) and is used by the other components in the project.

It provides various Swing implementations as well as other collective algorithms (copied from `coll` module of `OpenMPI`), in order to benchmark and compare `Swing` algorithms.

All algorithms written in this library are defined as **OVER** in `algorithms_config.json` since they are not internal MPI implementation but instead rely on MPI API.

### Modify `libswing`
Actual implementation must be declared in `include/libswing.h` and written in `libswing/libswing_<coll_type>.c`. Any helper function must be declared in `libswing/libswing_utils.h`.

To implement a new collective, arguments must be defined as a pre-compiler directive in `include/libswing.h` and called `<COLLECTIVE_TYPE>_ARGS`. Adhere to naming scheme of what is already written.

---

## `config` - Algorithms and test configurations

The selection of algorithms for benchmarking is determined using three key files: `algorithm_config.json`, `test.json`, and `parse_test.py`. 

- **`algorithm_config.json`** defines the available algorithms, specifying their constraints, dependencies, and applicable conditions. It contains metadata and categorizes algorithms based on collectives, such as `ALLREDUCE` and `REDUCE_SCATTER`, while also listing the required MPI versions and library dependencies.
- **`test.json`** serves as the test selection file, specifying parameters such as MPI type, version, collective operation, and relevant constraints like datatype and message sizes. It determines which tests should be run by filtering based on the required conditions.
- **`parse_test.py`** is responsible for validating `test.json` against a predefined schema and selecting appropriate algorithms. It ensures that the chosen algorithms meet the test's constraints, filtering them based on MPI version, message sizes, and other conditions. Additionally, it generates environment variables that define the selected test setup.

**NOTE:** the test script wrapper contains other variables to select when launching a test such as `LOCATION`, `TEST_TIME`, `N_NODES`, `DEBUG_MODE` and `NOTES`.


### `algorithm_config.json` – Defining Available Algorithms

The `algorithm_config.json` file defines the set of algorithms available for benchmarking and their constraints. It is structured as a JSON object with the following key sections:

- **`config_metadata`**: Contains metadata such as schema version, creation date, last modification date, and author information.
- **`collective`**: The main section, grouping algorithms under collective operations like `ALLREDUCE`, `ALLGATHER`, and `REDUCE_SCATTER`. 
  - Each algorithm has a **name** (e.g., `ring_ompi`, `recursive_doubling_ompi`) and a **description** explaining its function.
  - **`library`**: Specifies required MPI versions or dependencies (e.g., `ompi: "5.0.0"`).
  - **`dynamic_rule`**: Defines how the algorithm is selected dynamically (i.e. its number as defined in [Open MPI](https://docs.open-mpi.org/en/v5.0.6/tuning-apps/coll-tuned.html#tuning-collectives) or in [`OMPI_SWING`](https://github.com/DaXor-0/ompi_test) ). This field will be modified when `MPICH` algorithms are added.
  - **`tags`**: Labels like `internal`, `external`, `small_sizes`, or `large_sizes` for filtering.
  - **`constraints`**: Conditions that must be met, such as minimum `count` values or power-of-two restrictions.
  - **`additional_parameters`** (optional): Specifies extra tunable parameters, such as `segsize` for segmented algorithms.

This file ensures that only compatible algorithms are considered when selecting an execution plan for benchmarking.

### `test.json` – Specifying the Test Configuration

The `test.json` file defines the conditions and parameters for running a test, acting as a filtering mechanism for selecting appropriate algorithms. Its key fields include:

- **`libswing_version`**: Specifies the version of the `libswing` library.
- **`collective`**: Indicates which collective operation is being tested (e.g., `REDUCE_SCATTER`, `ALLGATHER`).
- **`MPI_Op`**: Defines the MPI operation (e.g., `MPI_SUM` for reduction).
- **`tags`**: Contains two lists:
  - `include`: Algorithms with these tags should be considered.
  - `exclude`: Algorithms with these tags should be skipped.
- **`specific`**: Similar to `tags`, but explicitly lists algorithms to include or exclude. It will override `tags` selection.
- **`cuda`**: A boolean value (`true`/`false`) indicating whether CUDA-based tests should be included.
- **`arr_counts`**: A list of message sizes (e.g., `8`, `64`, `512`, …) for which tests should be run.
- **`datatypes`**: Specifies the data types involved (`int8`, `int16`, `int32`, `int64`, etc.).

This file acts as an input to `parse_test.py`, which validates its structure and determines which algorithms meet the test's constraints.

### `parse_test.py` – Validating and Selecting Tests

The `parse_test.py` script plays a crucial role in processing the test configuration. It performs the following functions:

- **Validation**: Ensures that `test.json` follows the expected schema using `jsonschema`, checking required fields and formats.
- **Algorithm Filtering**: Matches algorithms from `algorithm_config.json` against the test conditions, verifying constraints like MPI version compatibility and message size requirements.
- **Environment Variable Exporting**: Generates environment variables defining the selected test setup, which needs to be exported for the test (it will be done automatically by the test scripts).

---

## `test` - Benchmark program

The `test/` directory contains a set of benchmark tests that are used to measure the performance of the `libswing` library functions as well as any other internal `MPI` algorithm. It compiles into the executable `bin/test`.

Algorithm selection is done via a command line parameters.

In case of internal MPI algorithm, library specific variables must be set beforehands. For this benchmarking suite, this is done automatically with the scripts.

The executable itself must be run with `srun` or `mpirun`/`mpiexec` and output is saved in `csv` format by rank 0.

#### Parameters:
- `<count>`: Number of data elements per process
- `<iterations>`: Total number of test iterations (including warm-up iterations)
- `<algorithm>`: Collective algorithm to test
- `<type_string>`: Data type (e.g., int32, float64)
- `<output_dir>`: Directory to save benchmark results

The collective type is selected via the environment variable `COLLECTIVE_TYPE`. Currently supported collectives include:
- `ALLREDUCE`
- `ALLGATHER`
- `REDUCE_SCATTER`
Additional collectives will be implemented in the future.

#### Saving benchmarking results
Before saving the results, a ground truth check on the last iteration is performed to ensure correctness.
Results are saved in files called `<count>_<algorithm>_<datatype>.csv`, stored in `results/<LOCATION>/test/data/` in the format:
| highest | rank0 | ... | rankN |
|---------|-------|-----|-------|
| ---     | ---   | ... | ---   |

Additionally, during the first run of a benchmarking test, a file is created to store node allocations and their corresponding MPI ranks.

### Implementing a new algorithm
For an already implemented collective:
1. External Algorithm (implemented in `libswing.h`):
  - Ensure the algorithm metadata in `scripts/config/algorithm_config.json` is correctly configured.
  - Update the switch statement in `get_<COLLECTIVE_TYPE>_function` in `test/test_utils.c` to include the new function.
2. Internal Algorithm (implemented inside the given MPI library):
  - Ensure the algorithm metadata in `scripts/config/algorithm_config.json` is correctly configured.
  - Ensure `ompi_rules/change_dynamic_rules.py` correctly modifies the dynamic rule file.

Note: Some algorithms may require additional parameters. This functionality is still under development.

For a new collective additional steps are required.
- in `test_utils.h`
  - add the collective to `coll_t` enum;
  - declare an allocator (for now all allocators have the same signature but it can change in the future);
  - `typedef` a function pointer for the specified collective and for its ground truth check;
  - define a wrapper if the function pointer typing does not correspond precisely to the one of the collective itself (for example if you use `size_t count` instead of `int count`)
  - populate `test_routine_t` struct accordingly;
  - define a test loop function using the macro;
  - declare a ground truth check function;
- create a file `test_<COLLECTIVE_TYPE>_utils.c`
  - include at minimum `libswing.h` `test_utils.h` and `mpi.h`
  - define the `allocator` and `ground_truth_check` functions declared in `test_utils.h`
- in `test_utils.c`
  - modify `get_collective_from_string` and `get_allocator`,
  - define a static inline `get_<COLLECTIVE_TYPE>_function` to return the normal collective function (or its wrapper) if the collective is internal and the collective defined in `libswing.h` if it's external.
  - modify the switch in `get_routine`, `test_loop` and `ground_truth_check` with custom behaviour for the new collective
  - modify `rand_sbuf_generator` and `debug_sbuf_generator` to correctly populate the sbuf of said collective (different collectives may have different sbuf dimension for a given `count` parameter)

#### Debugging
When compiled with `-DDEBUG`, the program:
- will not save benchmark results
- initializes `sbuf` in a predefined way
- prints `rbuf` and `rbuf_gt` if ground truth check fails before invoking `MPI_Abort`

---

## `ompi_rules` - Open MPI Rule File Generator
The `ompi_rules/` directory contains a [dynamic rule file](https://docs.open-mpi.org/en/v5.0.6/tuning-apps/coll-tuned.html#tuning-collectives) for Open MPI to modify the algorithms to run for the benchmark.

### `change_dynamic_rules.py` - Python script
It also contains a script `ompi_rules/change_dynamic_rules.py` that modifies aforementioned file accordingly to the algorithm to run.

It looks for the current `<COLLECTIVE_TYPE>` environment variable, reads the dynamic rule file looking for a line containing the collective name and modify the corresponding algorithm line accordingly to `<algorithm>` given and the corresponding `dynamic_rule` field of `<ALGORITHM_CONFIG_FILE>`.

#### Parameters
As a command line argument it requires
- `<algorithm>`: Collective algorithm to test

It requires the following environmental variables:
- `<ALGORITHM_CONFIG_FILE>` containing the path to the algorithm_config_file
- `<DYNAMIC_RULE_FILE>` containing the path to the dynamic_rule_file
- `<COLLECTIVE_TYPE>` containing the collective type of the test

##### Note
Beware that the following environmental variables must be set after the rule file is modified:
```bash
export OMPI_MCA_coll_tuned_use_dynamic_rules=1
export OMPI_MCA_coll_tuned_dynamic_rules_filename=${DYNAMIC_RULE_FILE}
```
In the normal workflow, those variables are set up with the scripts to run tests and debug.

##### TODO
This part must be modified to allow `MPICH` algorithm selection.

---
## `scripts` - Run the tests

This test suite provides a structured framework for running benchmark tests efficiently across different environments. **Users should interact only with `submit_wrapper.sh`**, which handles environment setup, test execution, and submission to Slurm (if applicable). All other scripts are used internally.

#### Execution Flow  

1. **User runs `submit_wrapper.sh`**  
   - Sets environment variables based on `LOCATION`.  
   - Sources the appropriate `environment/<location>.sh` script.  
   - Parses `test.json` to determine test parameters.  
   - Prepares output directories and metadata.  
   - Submits the job via Slurm or locally.  

2. **Test execution proceeds**  
   - `run_test_suite.sh` runs the benchmarks using the selected algorithms.  
   - Results are stored in a structured directory format under `results/<LOCATION>/<TIMESTAMP>/<TEST_ID>`.
   - Results are compressed and the directory is added to `.gitignore`.

### `submit_wrapper.sh` – The Only Script Users Should Run  

This is the **main entry point** for running tests. It handles:
- **Setting up the test environment** (choosing the correct machine-specific settings).
- **Compiling the necessary code** and setting up result directories.
- **Launching the test**, either locally or via a Slurm job (if `LOCATION` is not `local`).  

Users **must not** run any other scripts manually except of this. To start a test, simply execute:  

```bash
bash submit_wrapper.sh
```

Key environment variables set in this script:
- `N_NODES`: Number of nodes to use for the test. Will also set --nodes if run on a Slurm based environment.
- `LOCATION`: Defines the machine where tests will run (must be configured correctly). Use `local` if debugging on your machine.
- `TIMESTAMP`: Defines the output directory in which the tests results will be saved. The resulting directory will always be `results/<LOCATION>/<TIMESTAMP>/`.
- `DEBUG_MODE`: Enables a simplified test mode for debugging. Still in development.
- `TEST_CONFIG_FILE`: Select the `test.json` file to use for test selection.
- `NOTES`: Notes to be added when generating test metadata.

Slurm specific environment variables:
- `TEST_TIME`: Equivalent to --time variable to define the allocation request time.
- `TASK_PER_NODE`: Equivalent to --ntasks-per-node to define how many tasks run per single node. For now this is here only to allow for particular `qos` selections and does not modify `srun` behaviour. May change in future updates.

Beware that other location specific SLUM variable are declared inside `environment/<location>.sh`.

### `environment/<location>.sh` – Machine-Specific Environment Configuration  

This folder contains machine-specific scripts to set environment variables for different clusters or systems.  
- The correct script is **automatically sourced** based on the `LOCATION` variable in `submit_wrapper.sh`.  
- If an appropriate environment script is missing, the test will not proceed.  

Example: If `LOCATION="leonardo"`, then `environment/leonardo.sh` will be loaded.

It must contain:
- `export <ENV>` to export all strictly necessary environment variables for the machine.
  - `CC` to determine which compiler to use
  - `RUN` to define the command to run tests (`srun`, `mpiexec`...)
  - `SWING_DIR` full path to Swing_Test directory.
  - `MPI_LIB` mpi library type.
  - `MPI_LIB_VERSION` version of said library.
  - `ACCOUNT` set slurm account.
  - `PARTITION` set slurm partition.
  - (optional) `QOS` set quality of service.
  - (optional) `RUNFLAGS` flags to invoke with the `RUN` command (for example Snellius cluster requires `RUNFLAGS=--mpi=pmix`).
  - (optional) any other flag to set to modify test behaviour (for example on Leonardo `UCX_IB_SL=1`).
- `load_python()` function to load correct python module.
- (optional) `load_other_env_var()` function to load environment variables dependant on test selection (for example the one to activate or deactivate MPI-Cuda support).


### `run_test_suite.sh` – The Core Test Execution Script  

This script is responsible for **actually running the test**.  
- It parses the given test configuration to determine which algorithms to run.
- Updates metadata with a new row for given test
- It runs the selected benchmarks based on the parsed test configurations, changing dynamic rules whenever necessary.
- After the tests are run it runs the script to compress results and add uncompressed results directory to `.gitignore` (staging it in the process).

### `utils.sh` – Utility Functions  

This script provides helper functions used throughout the test suite.  
- Functions include error handling, logging, and loading additional environment variables.  
- It is sourced by `submit_wrapper.sh` and should **not** be run independently.

---

## Results Management

In this repository, the results are stored in a directory that is automatically compressed into a `.tar` file to save storage space and keep the repository clean. The process works as follows:

1. **Results directory:** All raw results are saved in a dedicated directory, different for each system on which tests are run.
2. **Test metadata:** For each new test, `scripts/run_test_suite.sh` will invoke `results/generate_metadata.py` script, to add the metadata of the new test into a `.csv` file containing metadata of tests ran on a given system.
3. **Compression Script:** After the results are gathered, `results/compress_results.sh` compress the results directory into a `.tar` file. The script also add to `.gitignore` the uncompressed results subdirectory.

##### WARNING:
If test is not interrupted before completion, there is no need to run the compression or the metadata script manually as everything is done automatically by the test suite script. Otherwise, one must source the required environmental variables used in the test and then run the script.

---

## Building
The project is built with a recursive makefile approach in which a top-level `Makefile` will run the `make` commands for each subfolder.
If you want to build the project without relying on `submit_wrapper.sh`:
1. **Build all components**:
   ```bash
   $ make
   ```
2. **Clean all builds**:
   ```bash
   $ make clean
   ```

The `Makefile` automatically handles the compilation and linking for each component, creating the necessary binaries and libraries in the `bin` and `lib` directories.

If you want to compile and build individual parts of the project you can either run the `make` command inside the desired subdirectory, or run `make -C <directory>`. This works also with the `clean` command.

---

## Data Analysis and Visualization

Still in development.

Take a look, but it's still WIP.

---

## TODO
#### Makefile
- [ ] make the current static `libswing.a` a dynamic `libswing.so` to be added with `LD_PRELOAD`
#### Libswing modifications
- [ ] implement allreduce bdw without bitmaps
- [ ] write reduce scatter swing
- [ ] document functions and comment code
#### Test program
- [ ] implement logic for allgather_k_bruck (radix selection) and debug both internal and external
#### OMPI rules
- [ ] modify to let it work also with `MPICH` algorithm selection
#### Plot python scripts
- [ ] adjust everything to new structure
- [ ] avoid working only with summaries, or at least give the possibility of not doing it
- [ ] improve naming of output graphs so that they don't get overwritten by similar graphs
- [ ] refactor to separate graph drawing functions to separate files
