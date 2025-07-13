# PICO - Performance Insights for Collective Operations

**PICO** is a benchmarking suite designed to unify and simplify the end-to-end workflow of collective communication benchmarking across multiple libraries and implementations.

## ✨ Features

- 📦 **Multi-library support**:
- 📊 **Integrated analysis and visualization** tools for interpreting benchmark data.
- 📝 **Metadata generation** support for better reproducibility.
- ⚙️ **Script-based orchestration** for streamlined benchmarking workflows.
- 🧰 **TUI for configuration generation** currently in development.

## 🚀 Getting Started

To explore usage options and submit benchmark jobs, run:

```bash
scripts/submit_wrapper.sh --help
````

This script is the main entry point for configuring and launching benchmarking workflows.

## 📁 Directory Overview

```
pico/
├── config/        # Benchmark configuration files and templates
├── include/       # Header files used throughout the suite
├── libbine/       # Custom library with non-defaults collective algorithms implementation
├── pico_core/     # Core benchmarking logic and orchestration
├── plot/          # Plotting scripts and analysis tools
├── results/       # Output results from benchmark runs
├── scripts/       # Submission and helper scripts (incl. submit_wrapper.sh)
├── selector/      # Backend and operation selection logic
├── tracer/        # Tracing and profiling utilities
├── tui/           # Terminal UI for config generation (in progress)
├── LICENSE        # License file
├── Makefile       # Build instructions
└── README.md      # Project documentation
```

## 🔧 Development Roadmap

* ✅ Support for OMPI, MPICH, CrayMPICH, and NCCL
* ✅ Plotting and analysis tools
* ✅ Metadata creation tools for improved reproducibility
* ⚙️ TUI for test config generation (in development)
* ⚙️ Simplified selection of important networking stack parameters


