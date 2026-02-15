# Introduction
This directory behaves like a Python plotting library with a single CLI entry point.
Invoke `python -m plot <subcommand>` from the repository root.  Each plot type lives in
its own module under `plot/plots/` and exposes a `generate_*` function so the plots can
also be scripted from Python.  Example batch recipes are provided under `plot/scripts/`.

Key commands:
- `python -m plot summary --summary-file …` recreates the legacy line, bar, and cut bar
  plots (see `python -m plot line|bar|cut --help` for individual plots).
- `python -m plot heatmap …` builds the algorithm-family heatmaps.
- `python -m plot comparison-heatmap …` produces the target-algorithm comparison heatmaps.
- `python -m plot bine-heatmap …` generates best-Bine variant heatmaps.
- `python -m plot boxplot …` generates the improvement boxplots.
- `python -m plot refined …` renders the refined line plot and stacked-latency bar chart.
