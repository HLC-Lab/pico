# Introduction
This directory behaves like a Python plotting library with a single CLI entry point.
Invoke `python -m plot <subcommand>` from the repository root. Each plot type lives in
its own module under `plot/plots/` and exposes a `generate_*` function so the plots can
also be scripted from Python. Example batch recipes are provided under `plot/scripts/`.

## Data pipelines

Three independent data pipelines feed the CLI:

| Pipeline | Subcommands | Input |
|---|---|---|
| **Summary-based** | `line`, `bar`, `cut`, `summary` | `aggregated_results_summary.csv` produced by `summarize_data.py` |
| **Metadata-based** | `heatmap`, `comparison-heatmap`, `boxplot`, `bine-heatmap` | Metadata CSV + auto-invokes `summarize_data.py` via subprocess |
| **Trace-based** | `refined` | Raw trace directories from instrumented runs |

A separate standalone script `summarize_instrumented.py` processes `*_instrument.csv` files
independently from the CLI.

## Subcommands

### Summary-based

| Subcommand | Required | Key optional | Produces |
|---|---|---|---|
| `line` | `--summary-file` | `--collective`, `--datatype`, `--algorithm`, `--error-col` (se/std/ci/iqr), `--error-mode` (band/bar) | Log-log line plot of mean latency vs buffer size per algorithm |
| `bar` | `--summary-file` | same as `line` + `--normalize-by` (reference algorithm), `--errorbars` (none/se/ci), `--std-threshold` | Normalized bar plot relative to a baseline algorithm |
| `cut` | `--summary-file` | same as `bar` | Like `bar` with a broken y-axis (two panels) to show both small and large differences |
| `summary` | `--summary-file` | all of the above | Runs **line + bar + cut** in one pass for every (datatype, collective, gpu_awareness) group |

### Metadata-based

These read results metadata (`results/<system>_metadata.csv`), auto-run `summarize_data.py`
if the summary CSV is missing, and compute bandwidth from latency means.

| Subcommand | Required | Key optional | Produces |
|---|---|---|---|
| `heatmap` | `--system`, `--collective`, `--nnodes` | `--tasks-per-node`, `--notes`, `--exclude`, `--metric` (mean/median/percentile_90), `--hide-y-labels` | Heatmap of winning algorithm family per (buffer_size × nnodes) cell |
| `comparison-heatmap` | `--system`, `--collective`, `--nnodes` | `--target-algo` (default: `ring_ompi`), `--show-names` | Heatmap of a specific algorithm's bandwidth ratio vs the best |
| `boxplot` | `--system`, `--nnodes` | `--tasks-per-node`, `--notes`, `--exclude`, `--metric` | Horizontal boxplot of Bine improvement distribution per collective |
| `bine-heatmap` | `--system`, `--collective` (ALLGATHER or REDUCE_SCATTER), `--runs` | `--metric` | Heatmap of best Bine variant per cell with ratio over baseline |

### Trace-based

| Subcommand | Required | Key optional | Produces |
|---|---|---|---|
| `refined` | `--baseline`, `--op-null`, `--no-memcpy`, `--no-memcpy-op-null` | `--nodes` (default: 8), `--messages`, `--collective`, `--congested`, `--label`, `--title` | Refined line plot + stacked latency bar chart from four trace directory variants |

## Files

```
plot/
├── __init__.py, __main__.py    # Entry point
├── cli.py                      # Argparse parser + handler functions
├── data.py                     # Summary CSV loading, filtering, normalization
├── utils.py                    # Styling, legends, error bars, metadata dataclass
├── summarize_data.py           # Standalone: raw CSVs → aggregated_results_summary.csv
├── summarize_instrumented.py   # Standalone: _instrument.csv → per-column means summary
├── table.py                    # LaTeX tables of Bine vs SOTA improvement (standalone)
├── grouped.py                  # Grouped bar chart (experimental)
├── plots/
│   ├── line_plot.py
│   ├── bar_plot.py
│   ├── cut_bar_plot.py
│   ├── box_plot.py
│   ├── family_heatmap.py
│   ├── comparison_heatmap.py
│   ├── plot_bine_heatmap.py
│   ├── refined_line_plot.py
│   ├── stacked_latency_plot.py
│   └── refined_loader.py       # Trace data loader
├── scripts/                    # Batch recipes (pico_paper.sh, heatmaps.sh, boxplots.sh, ...)
└── todo.md                     # Known TODOs
```
