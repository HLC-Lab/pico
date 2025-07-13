# PICO - Performance Insights for Collective Operations

**PICO** is a **lightweight**, **extensible**, and **reproducible** benchmarking suite for evaluating and tuning **collective communication operations** across diverse libraries and hardware platforms.

Built for researchers, developers, and system administrators, PICO streamlines the **entire benchmarking workflow**—from configuration to execution, tracing, and analysis—across MPI, NCCL, and user-defined collectives.

---

## 🔍 Why PICO?

Benchmarking collectives at scale is hard. Libraries vary, hardware stacks are complex, and existing tools often provide only coarse metrics with little automation.

**PICO addresses this by providing:**

- 📦 **Unified testing** across MPI (OMPI, MPICH, CrayMPICH), NCCL, and custom implementations.
- 🎛️ **Configuration via CLI or TUI**, with reproducible, declarative test descriptions.
- 📊 **Built-in analysis tools** and plotting scripts to extract insights quickly.
- 📋 **Automatic metadata collection**: environment variables, library/runtime versions, hardware info.
- 🔬 **Fine-grained instrumentation** for analyzing internal phases of collective operations.

---

## 💡 Key Features

- **Multi-library support**:
  - ✅ MPI (OMPI, MPICH, Cray MPICH)
  - ✅ NCCL (NVIDIA)
- **Extensibility**:
  - Plug in new collectives or libraries with minimal glue code.
- **Reproducibility**:
  - Complete environment capture (UCX, OFI, SHARP, etc.)
  - JSON-based configuration and results logging.
- **Integrated visualization**:
  - CSV output, tracer utilities, and publication-ready plots.
- **Active TUI development**:
  - CLI-based now, full interactive UI under construction.


## 🚀 Quickstart

To explore available benchmarking options or start a test, run:

```bash
scripts/submit_wrapper.sh --help
````

This command-line tool will guide you through job setup and execution.

> A full-featured **Terminal UI (TUI)** for interactive config generation is in active development.

---

## 📁 Project Structure

```
PICO/
├── config/        # Benchmark configuration files and templates
├── include/       # Header files
├── libbine/       # Benchmark infrastructure core
├── pico_core/     # Main orchestration and benchmarking engine
├── plot/          # Scripts for result visualization
├── results/       # Collected output and logs
├── scripts/       # Submission and utility scripts (e.g., submit_wrapper.sh)
├── selector/      # Backend/algorithm selection logic
├── tracer/        # Tracing utilities for network traffic analysis
├── tui/           # Terminal UI for guided test creation (in progress)
├── LICENSE        # License information
├── Makefile       # Build system
└── README.md      # This file
```

---

## 📊 Analysis and Tracing

* **CSV result summaries** and formatted terminal logs
* **Tracer utility** to visualize traffic across global links
* **Integrated plotting scripts** for high-quality publication-ready figures
* **Instrumentation hooks** for per-stage timing and bandwidth analysis

---

## 📌 Supported Platforms

* Validated on:

  * 🇪🇺 **Leonardo**
  * 🇪🇺 **LUMI**
  * 🇪🇺 **MareNostrum 5**

Compatible with **SLURM-based** job schedulers. Other environments can be added via JSON-based platform descriptors.

---

## 🧩 Development Roadmap

* ✅ Stable benchmarking and metadata capture
* ✅ NCCL + MPI support
* ✅ CLI-based submission
* 🔜 Full-featured TUI for test config generation
* 🔜 Simplified platform setup and expansion


## 🤝 Acknowledgments

PICO is developed by Daniele De Sensi ans Saverio Pasqualoni
Department of Computer Science, Sapienza University of Rome

> For questions or contributions, please contact:
> [desensi@di.uniroma1.it](mailto:desensi@di.uniroma1.it)
> [pasqualoni.1845572@studenti.uniroma1.it](mailto:pasqualoni.1845572@studenti.uniroma1.it)

