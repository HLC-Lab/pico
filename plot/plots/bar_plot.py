# Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
# Licensed under the MIT License

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..utils import (
    PlotMetadata,
    apply_adaptive_legend,
    build_tab10_palette,
    draw_errorbars,   # updated selectable version
    ensure_dir,
    format_bytes,
    sort_key,
    style_axes,
)


def _resolve_output_dir(system: str, output_dir: str | Path | None) -> Path:
    return ensure_dir(output_dir) if output_dir else ensure_dir(Path("plot") / system)


def generate_bar_plot(
    data: pd.DataFrame,
    *,
    metadata: PlotMetadata,
    collective: str,
    datatype: str,
    gpu_awareness: str,
    # error bar controls
    errorbars: str = "se",        # "none" | "se" | "ci"
    k: float = 1.96,              # multiplier for "se" mode (1.96 ~ 95%)
    threshold: float = 0.15,      # if error > threshold, draw a red marker instead
    marker_loc: float = 0.05,     # vertical offset for red marker
    output_dir: str | Path | None = None,
) -> Path:
    """
    Render the normalized bar plot for a specific ``collective`` / ``datatype`` pair.

    The incoming dataframe must already contain:
      - normalized_mean
      - and either:
          * normalized_se   (for errorbars="se")
          * normalized_ci_lower / normalized_ci_upper (for errorbars="ci")

    Backward compat:
      - if errorbars="se" and only normalized_std exists, we treat it as normalized_se,
        but you should prefer normalized_se from proper ratio propagation.
    """
    if data.empty:
        raise ValueError("No data available for generate_bar_plot.")

    # Backward compatibility shim (optional but practical)
    if errorbars == "se" and "normalized_se" not in data.columns and "normalized_std" in data.columns:
        data = data.copy()
        data["normalized_se"] = data["normalized_std"]

    sorted_algos = sorted(data["algo_name"].unique().tolist(), key=sort_key)

    plt.figure(figsize=(12, 6))
    palette = build_tab10_palette(sorted_algos)

    ax = sns.barplot(
        data=data,
        x="buffer_size",
        y="normalized_mean",
        hue="algo_name",
        hue_order=sorted_algos,
        errorbar=None,            # we draw error bars ourselves
        palette=palette,
        edgecolor="black",
        linewidth=1.0,
    )

    # Selectable error bars: none / SE / CI
    draw_errorbars(
        ax,
        data,
        sorted_algos,
        mode=errorbars,
        x_col="buffer_size",
        algo_col="algo_name",
        y_col="normalized_mean",
        se_col="normalized_se",
        k=k,
        ci_lower_col="normalized_ci_lower",
        ci_upper_col="normalized_ci_upper",
        threshold=threshold,
        loc=marker_loc,
    )

    # X tick labels formatting
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels([format_bytes(t.get_text()) for t in ax.get_xticklabels()])

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        new_labels = [
            " ".join(w.capitalize() for w in label.replace("_", " ").split())
            for label in labels
        ]
        apply_adaptive_legend(ax, handles=handles, labels=new_labels, loc="lower left")

    # Title
    if metadata.total_nodes == metadata.mpi_tasks:
        title = f"{metadata.system.capitalize()}, {collective.lower().capitalize()}, {metadata.nnodes} nodes {'GPU' if gpu_awareness == 'yes' else 'CPU'}"
    else:
        title = (
            f"{metadata.system.capitalize()}, {collective.lower().capitalize()}, "
            f"{metadata.nnodes} nodes ({metadata.mpi_tasks} tasks)"
            f"{'GPU' if gpu_awareness == 'yes' else 'CPU'}"
        )
    plt.title(title, fontsize=18)
    plt.xlabel("Message Size", fontsize=15)
    plt.ylabel("Normalized Execution Time", fontsize=15)

    style_axes(ax)
    plt.tight_layout()

    target_dir = _resolve_output_dir(metadata.system, output_dir)

    # Include error bar mode in filename so artifacts are distinguishable
    name = f"{collective.lower()}_{metadata.nnodes}_{datatype}_{metadata.timestamp}_{errorbars}_barplot{'_gpu_aware' if gpu_awareness == 'yes' else ''}_{errorbars}.pdf"
    full_path = target_dir / name

    plt.savefig(full_path, dpi=300)
    plt.close()
    return full_path