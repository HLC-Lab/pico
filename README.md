# [PICO](https://github.com/HLC-Lab/pico) — Performance Insights for Collective Operations

[![GitHub stars](https://img.shields.io/github/stars/HLC-Lab/pico?style=social)](https://github.com/HLC-Lab/pico/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](https://github.com/HLC-Lab/pico/issues)

> 💫 If you find **PICO** useful for your research or benchmarking work, please consider giving it a ⭐ on [GitHub](https://github.com/HLC-Lab/pico)!

---

**PICO** is a **lightweight**, **extensible**, and **reproducible** benchmarking suite for evaluating and tuning **collective communication operations** across diverse libraries and hardware platforms.

Built for researchers, developers, and system administrators, PICO streamlines the **entire benchmarking workflow**—from configuration to execution, tracing, and analysis—across MPI, NCCL, and user-defined collectives.

## ⭐ Highlights
- 📦 **Unified** micro-benchmarking of both CPU and GPU collectives, across a variety of MPI libraries (Open MPI, MPICH, Cray MPICH), NCCL and user-defined  algorithms.
- 🎛️ **Guided** configuration via a fully fledged Textual TUI or CLI-driven JSON/flag workflow with per-site presets.
- 📋 **Reproducible** runs through environment capture, metadata logging, and timestamped result directories.
- 🧩 Built-in **correctness checks** for custom collectives and automatic ground-truth validation.
- 🧭 **Per-phase instrumentation**, going beyond micro-benchmarking, hence the name PICO
- 🧵 Queue-friendly orchestration that compiles, ships, and archives jobs seamlessly on **SLURM clusters** or in local mode for debugging.
- 📊 **Bundled plotting, tracing, and scheduling utilities** for streamlined post-processing and algorithm engineering.

## Architecture at a Glance

```
📁 Configuration
 ├─ 🧩 Sources: Textual TUI • JSON • CLI flags
 └─ ⚙️ Validation & module loading via submit_wrapper.sh

🚀 Orchestration
 ├─ 🧵 scripts/orchestrator.sh iterates over:
 │    • Libraries × Collectives × Message Sizes
 └─ 🏗️ Builds binaries and dispatches jobs (SLURM or local)

🧠 Execution
 ├─ pico_core / libpico executables
 ├─ ✅ Correctness checks
 └─ 🧭 Optional per-phase instrumentation

📊 Results
 ├─ results/<system>/<timestamp>/
 │    • CSV metrics
 │    • Logs
 │    • Metadata
 │    • Archives
 └─ Post-processing utilities:
      • plot/ • tracer/ • schedgen/ • selector/
```

## 🚀 Quickstart

The recommended way to use **PICO** is through its **Textual TUI**, which guides you from configuration to job submission.

### ⚙️ 1. Configure Your Environment

Ensure you have at least one valid environment definition under `config/environment/` (TUI) or `config/environments/` (legacy CLI).

A working `local` sample is provided, modify it for your local machine.

For remote clusters, you should mirror one of the existing environment templates and adapt it to your site (a setup wizard to simplify this configuration is on its way!)

### 🧭 2. Create a virtual env and launch the TUI

Create and activate a Python virtual environment, then install the Python dependencies used by the **TUI** and analysis tools:
```bash
pip install -r requirements.txt
```

Start the interactive interface (see [tui/README.md](tui/README.md) for a full walkthrough of the four-step wizard) to configure the environment, select libraries, choose algorithms, and export.

```bash
python tui/main.py
```

### 🧩 3. Generate a Test Description

Within the TUI, define:

* The target collective(s)
* Message sizes and iteration counts
* Backends (MPI / NCCL / custom)
* Instrumentation and validation settings

The TUI will produce a **test descriptor file** encapsulating all these options.

The export lands in `tests/<name>.json` (full configuration) and `tests/<name>.sh` (shell exports).

### 🚀 4. Run the Benchmark

Execute the generated descriptor using the wrapper script, which handles compilation, dispatch, and archival:

```bash
scripts/submit_wrapper.sh -f [path_to_test_sh_file]
```

This command will orchestrate the full benchmarking workflow — locally or on SLURM clusters — using your defined environment.

### 🧰 Optional: CLI Workflow (Legacy)

You can still invoke **PICO** directly via the CLI to explore options or run ad-hoc tests. If that is desired, after step 1 do:

```bash
scripts/submit_wrapper.sh --help
```

> ⚠️ **Note:** The CLI path is currently *partially maintained*; some flags may be deprecated as functionality transitions to the TUI.

Example CLI invocation:
```bash
scripts/submit_wrapper.sh \
  --location leonardo \
  --nodes 8 \
  --ntasks-per-node 32 \
  --collectives allreduce,allgather \
  --types int32,double \
  --sizes 64,1024,65536 \
  --segment-sizes 0 \
  --time 01:00:00 \
  --gpu-awareness no
```
- Provide comma-separated lists for datatypes, message sizes, and segment sizes.
- Use `--gpu-awareness yes` and `--gpu-per-node` to benchmark NCCL or CUDA-aware MPI collectives.
- Pass `--debug yes` for quick validation runs with reduced iterations and debug builds.
- When `--compile-only yes` is set, the script stops after building `bin/pico_core` and its GPU counterpart.

### 💻 Dependencies
- A C/C++ compiler and MPI implementation (Open MPI, MPICH, or Cray MPICH). CUDA-aware MPI or NCCL is optional for GPU runs.
- (Optional) CUDA toolkit and a compatible NCCL build for GPU collectives.
- Python 3.9+ with `pip` for the TUI and analysis utilities (`pip install -r requirements.txt`).
- SLURM for cluster submissions; local mode is supported for functional testing.
- Basic build tools (`make`) and a Bash-compatible shell.

## 🧠 Core Components
- `pico_core/` — C benchmarking driver that allocates buffers, times collectives, checks results, and writes output.
- `libpico/` — Library of custom collective algorithms and instrumentation helpers, selectable alongside vendor MPI/NCCL paths.
- `scripts/submit_wrapper.sh` — Entry point that parses CLI flags or TUI exports, loads site modules, builds binaries, activates Python envs, and launches SLURM or local runs.
- `scripts/orchestrator.sh` — Node-side runner that sweeps libraries, algorithm sets, GPU modes, message sizes, and datatypes while invoking metadata capture and optional compression.
- `config/` — Declarative environment, library, and algorithm descriptions consumed by the TUI and CLI (modules to load, compiler wrappers, task/GPU limits). See [config/environment/README.md](config/environment/README.md) for the environment schema.
- `tui/` — Textual-based UI that guides the user through environment selection, library selection, algorithm mix, and exports the shell/JSON bundle for later submission. See [tui/README.md](tui/README.md) for usage and extension details.
- `plot/` — Python package and CLI (`python -m plot …`) that turns CSV summaries into line charts, bar charts, heatmaps, and tables. See [plot/README.md](plot/README.md) for available subcommands and data pipelines.
- `tracer/` — Tools for network-awareness studies (link utilization estimates, cluster job monitoring, scatterplots/boxplots). See [tracer/README.md](tracer/README.md) for details.
- `schedgen/` — Adapted SPCL scheduler generator used to derive algorithm schedules from communication traces. See [schedgen/README.md](schedgen/README.md) for usage and built-in algorithms.
- `selector/` — Open MPI tuning rule generation (`generate_ompi_tuning.py`) and dynamic rule selection helpers.
- `results/` — Storage for raw outputs, metadata CSVs (per system), and helper scripts such as `generate_metadata.py`.

## 💡 What Happens During a Run
1. Environment sourcing loads modules, compiler wrappers, MPI/NCCL paths, and queue defaults. In the CLI workflow this happens via `config/environments/<location>.sh`; in the TUI workflow the test descriptor (`tests/<name>.sh`) already carries all resolved settings.
2. The Makefile builds `libpico` first, then `pico_core` (CPU) and optionally `pico_core_cuda` (GPU) or `pico_core_nccl` (NCCL), honouring debug and instrumentation flags.
3. A Python virtual environment is activated and populated with plotting/tracing dependencies on demand.
4. `scripts/orchestrator.sh` iterates over every selected library, collective, datatype, message size, and GPU mode. For each combination it:
   - Prepares per-collective environment variables and propagates algorithm lists to the workers.
   - Generates metadata entries through `results/generate_metadata.py`, capturing cluster, job, library, GPU, and note fields.
   - Runs `pico_core`, which allocates buffers, initializes randomized inputs (deterministic when debugging), executes warmups, measures iterations, and compares the outcome against vendor MPI results.
   - Optionally enables LibPICO instrumentation tags to time internal algorithm phases.
5. Outputs are written under `results/<location>/<timestamp>/`; in non-debug runs the directory can be tarred and optionally deleted.

## 📈 Results and Analysis
- CSV files follow the `<count>_<algorithm>_<datatype>.csv` naming scheme (or `<count>_<algorithm>_<segsize>_<datatype>.csv` for segmented collectives). Instrumented builds append `_instrument` before the extension. Rows contain per-iteration timing or summary statistics depending on `--output-level` (supported values: `all`, `minimal`).
- Allocation maps (`alloc_<tasks>.csv`) record rank-to-node placement. GPU runs append `_GPU`.
- SLURM logs reside alongside the CSVs (`slurm_<jobid>.out/.err`) unless in debug mode.
- Metadata is appended to `results/<location>_metadata.csv`, enabling cross-run filtering by timestamp, collective, library version, GPU involvement, and notes.
- Example plotting commands:
```bash
python -m plot summary --summary-file results/leonardo/<timestamp>/summary.csv
python -m plot heatmap --system leonardo --nnodes 8 --collective allreduce
python -m plot boxplot --system lumi --nnodes 8 --notes "production"
```
- The tracer package (`tracer/trace_communications.py`) estimates traffic on global links for recorded allocations, while `tracer/sinfo` can processes week-long job snapshots from monitored clusters.
- `selector/generate_ompi_tuning.py` — Produce Open MPI tuning rules from benchmark results, consumable by `OMPI_MCA_coll_tuned_dynamic_rules_filename`. See [selector/README_generate_ompi_tuning.md](selector/README_generate_ompi_tuning.md) for usage.

## 🧪 Instrumentation and Custom Collectives
- Building with `-DPICO_INSTRUMENT` exposes the `PICO_TAG_BEGIN/END` macros defined in `include/libpico.h`. 
  - These can be inserted into LibPICO collective implementations to record per-phase timings, which are emitted into `_instrument.csv` files. Detailed usage and examples are provided in [libpico/instrument.md](./libpico/instrument.md).
  - Instrumentation is supported for CPU collectives; the macros are transparent when GPU paths are enabled.
- To add new algorithms, follow the step-by-step guide in [libpico/adding_algorithms.md](libpico/adding_algorithms.md). The TUI and CLI automatically surface new options once registered.

## 🧱 Extending PICO
- **Environments:** See [config/environment/README.md](config/environment/README.md) for the full schema reference and step-by-step guide to adding new cluster profiles. Real-world examples are available under `config/environment/`.
- **Libraries:** Update `<env>_libraries.json` to expose additional MPI/NCCL builds, compiler flags, GPU capabilities, and metadata strings. The TUI reads these files at runtime.

## 🗂️ Repository Layout
```
pico/
├── include/                # Public LibPICO API and instrumentation macros
├── libpico/                # Custom collective implementations
├── pico_core/              # Benchmark driver and MPI/NCCL glue code
├── config/                 # Environment, library, and algorithm JSON descriptors
├── scripts/                # Submission, orchestration, metadata, and shell helpers
├── tui/                    # Textual UI for configuration authoring
├── plot/                   # Plotting package and CLI
├── tracer/                 # Network tracing and allocation analysis tools
├── schedgen/               # Communication schedule generator (SPCL fork)
├── selector/               # Dynamic rule selection helpers for Open MPI
├── tests/                  # Sample exported configurations
└── results/                # Generated data, metadata CSVs, and helper scripts
```

## 🪪 Credits and License
**PICO** is developed by *Daniele De Sensi*, *Saverio Pasqualoni* and *Lorenzo Protano* at the Department of Computer Science, Sapienza University of Rome. The project is licensed under the **MIT License**.

Schedgen code was originally released by SPCL @ ETH Zurich under the **BSD 4-Clause license**. The version bundled with PICO includes targeted modifications to support its extended scheduling and tracing workflow.

### 📬 Contact
- [desensi@di.uniroma1.it](mailto:desensi@di.uniroma1.it)
- [saverio.pasqualoni@kaust.edu.sa](mailto:saverio.pasqualoni@kaust.edu.sa)
