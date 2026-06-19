# generate_ompi_tuning.py

Generate Open MPI tuning rules from benchmark results, producing a file
consumable by `OMPI_MCA_coll_tuned_dynamic_rules_filename`.

## Overview

1. Load the metadata CSV (`results/<system>_metadata.csv`) and filter for OMPI runs
2. Filter pipeline (applied in order):
   1. `--include-notes REGEX` — keep only timestamps where **all** rows match
   2. `--exclude-notes REGEX` — remove timestamps where **any** row matches
   3. `--include-timestamps` — keep only specified timestamps/ranges
   4. `--exclude-timestamps` — remove specified timestamps/ranges
3. For each timestamp, ensure data is summarized (`aggregated_results_summary.csv`)
4. Concatenate all aggregated results
5. Keep only algorithms recognized as OMPI (from `config/algorithms/MPI/Open-MPI/`)
6. For each `(collective_type, comm_size, array_dim)` tuple, select the best algorithm
7. Write the tuning file in OMPI dynamic rules format

## Selection criteria

### `bandwidth` (default)

Compute `bandwidth = (buffer_size × 8) / metric`, then select the algorithm
with the highest bandwidth for each group.

### `latency`

Directly select the algorithm with the lowest latency (metric) for each group.
Mathematically equivalent to `bandwidth` for a fixed `buffer_size`.

## Latency metrics

| Value | Description |
|--------|-------------|
| `median` (default) | Median execution time |
| `mean` | Mean execution time |
| `percentile_90` | 90th percentile |

## Flags

### Basic options

| Flag | Default | Description |
|------|---------|-------------|
| `--system` | — (required) | System name (e.g. `leonardo`, `mare_nostrum`) |
| `--ompi-lib` | — (required) | OMPI library display name from environment config (e.g. `"Open MPI 4.1.6"`) |
| `--results-dir` | `results/` | Path to results directory |
| `--output, -o` | `results/<system>/ompi_tuning_rules_<system>.txt` | Output file path |

### Filtering

| Flag | Description |
|------|-------------|
| `--include-notes REGEX` | Only timestamps where **all** rows match the regex |
| `--exclude-notes REGEX` | Exclude timestamps where **any** row matches the regex |
| `--include-timestamps` | Only keep these timestamps by name or range (`ts1,ts2-ts4,ts5`) |
| `--exclude-timestamps, -x` | Exclude timestamps by name or range (`ts1,ts2-ts4,ts5`) |
| `--list-notes` | Print all unique `notes` values in the metadata and exit |

### Selection

| Flag | Default | Description |
|------|---------|-------------|
| `--criterion` | `bandwidth` | Criterion: `bandwidth` (max) or `latency` (min) |
| `--metric` | `median` | Metric: `mean`, `median`, `percentile_90` |

### Verbosity

| Flag | Description |
|------|-------------|
| `--quiet, -q` | Errors and warnings only |
| `--verbose, -v` | Show detailed filtering decisions |
| `--annotate` | Add human-readable comments (message size + algorithm name) to output |

## Output file format

The file follows the Open MPI dynamic rules format, structured in nested
blocks:

```
<num_collectives>                          # number of collectives

# <COLLECTIVE_NAME> (<coll_id>)
<coll_id>                                  # collective numeric ID
<num_comm_sizes>                           # number of communicator sizes

# comm_size=<size>
<comm_size>                                # number of processes
<num_rules>                                # rules for this comm_size
<min_bytes> <algo_id> <segment> <radius>   # one row per algorithm switch
...
```

- `<algo_id>`: OMPI algorithm selection number, mapped from
  `config/algorithms/MPI/Open-MPI/<collective>.json`
- `<segment>` and `<radius>`: always `0` (not used by this generator)
- Rules are ordered by `array_dim` ascending; each time the best algorithm
  changes, a new row is added with a `min_bytes` threshold set to the first
  message size where the new algorithm performed better

## Examples

```bash
# Default (bandwidth, median)
python selector/generate_ompi_tuning.py \
    --system leonardo --ompi-lib "Open MPI 4.1.6"

# Select by latency
python selector/generate_ompi_tuning.py \
    --system leonardo --ompi-lib "Open MPI 4.1.6" --criterion latency

# Exclude a timestamp range
python selector/generate_ompi_tuning.py \
    --system leonardo --ompi-lib "Open MPI 4.1.6" \
    -x 2025_04_04___18_21_29-2025_04_06___20_44_58

# Filter by notes and custom output path
python selector/generate_ompi_tuning.py \
    --system mare_nostrum --ompi-lib "Open MPI 4.1.6" \
    --include-notes "rail 4" -o custom_rules.txt

# Include only specific timestamps
python selector/generate_ompi_tuning.py \
    --system leonardo --ompi-lib "Open MPI 4.1.6" \
    --include-timestamps 2025_04_05___10_00_00-2025_04_06___20_00_00

# Annotate tuning rules with comments
python selector/generate_ompi_tuning.py \
    --system leonardo --ompi-lib "Open MPI 4.1.6" \
    --include-notes "rail 4" --annotate -o annotated_rules.txt
```
