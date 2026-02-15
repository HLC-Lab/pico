# Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
# Licensed under the MIT License

"""
Concrete plotting helpers.  Each module exposes a single ``generate_*`` function
that takes a pre-filtered dataframe and renders one figure.
"""

from .bar_plot import generate_bar_plot
from .plot_bine_heatmap import BestBineHeatmapConfig, generate_best_bine_heatmap
from .cut_bar_plot import generate_cut_bar_plot

from .box_plot import BoxplotConfig, generate_boxplot
from .line_plot import generate_line_plot

__all__ = [
    "generate_bar_plot",
    "BestBineHeatmapConfig",
    "generate_best_bine_heatmap",
    "generate_cut_bar_plot",
    "generate_line_plot",
    "generate_boxplot",
]
