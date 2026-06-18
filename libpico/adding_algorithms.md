# Adding New Algorithms to libpico

This guide describes the steps required to add a new collective algorithm to
the libpico custom library. Once registered, the algorithm is automatically
discoverable by the TUI and executable via `submit_wrapper.sh`.

## Step-by-step

### 1. Implement the algorithm

Add your implementation in the appropriate `libpico/libpico_<collective>.c` file.
Use the MPI argument macro from `include/libpico.h` that matches your collective:

```c
// allreduce example
int my_new_algo(ALLREDUCE_MPI_ARGS) {
    // ... implementation ...
    return MPI_SUCCESS;
}
```

Available macros:

| Collective      | Macro                    |
|-----------------|--------------------------|
| allreduce       | `ALLREDUCE_MPI_ARGS`     |
| allgather       | `ALLGATHER_MPI_ARGS`     |
| alltoall        | `ALLTOALL_MPI_ARGS`      |
| bcast           | `BCAST_MPI_ARGS`         |
| gather          | `GATHER_MPI_ARGS`        |
| reduce          | `REDUCE_MPI_ARGS`        |
| reduce_scatter  | `REDUCE_SCATTER_MPI_ARGS`|
| scatter         | `SCATTER_MPI_ARGS`       |

No Makefile changes are needed: `libpico/Makefile` uses `$(wildcard *.c)`, so new
source files are compiled automatically.

### 2. Declare in the public header

Add the function prototype in `include/libpico.h`, next to the other functions
of the same collective:

```c
int my_new_algo(ALLREDUCE_MPI_ARGS);
```

### 3. Register in the algorithm dispatcher

Open `pico_core/pico_core_utils.c` and add a `CHECK_STR` entry in the
appropriate `get_<collective>_function()`:

```c
static inline allreduce_func_ptr get_allreduce_function(const char *algorithm) {
  CHECK_STR(algorithm, "my_new_algo_over", my_new_algo);
  // ... existing entries ...
}
```

The first argument is the string identifier used by the TUI and CLI to refer
to your algorithm (by convention suffixed with `_over`). `CHECK_STR` is defined
in `pico_core_utils.h` and resolves the string to the function pointer at
runtime.

### 4. Add a JSON algorithm descriptor

Create or update the algorithm list under
`config/algorithms/<Standard>/<Library>/<collective>.json`.

For MPI via libpico, use:
`config/algorithms/MPI/LibPico/<collective>.json`

For NCCL via libpico, use:
`config/algorithms/NCCL/PicoLib/<collective>.json`

Each entry is a JSON object with `desc`, `version`, `selection`, optional
`constraints`, and `tags`. Example:

```json
{
  "my_new_algo_over": {
    "desc": "Human-readable description of the algorithm",
    "version": "1.0.0",
    "selection": "pico",
    "tags": ["latency_optimal", "small_sizes"],
    "constraints": [
      {
        "key": "count",
        "conditions": [{"operator": ">=", "value": "comm_sz"}]
      }
    ]
  }
}
```

The TUI reads this file via `tui/config_loader.py` and displays the algorithm
in the Algorithms step.

## Summary of files to touch

| File | Action |
|---|---|
| `libpico/libpico_<collective>.c` | Add implementation |
| `include/libpico.h` | Add function prototype |
| `pico_core/pico_core_utils.c` | Add `CHECK_STR` in the right `get_<collective>_function()` |
| `config/algorithms/<Std>/<Lib>/<collective>.json` | Add JSON entry with metadata |
