# Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
# Licensed under the MIT License

from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils import (
    PlotMetadata,
    apply_adaptive_legend,
    build_tab10_palette,
    draw_errorbars,
    ensure_dir,
    format_bytes,
    sort_key,
    style_axes,
)


def _resolve_output_dir(system: str, output_dir: str | Path | None) -> Path:
    return ensure_dir(output_dir) if output_dir else ensure_dir(Path("plot") / system)

def generate_cut_bar_plot(
    data: pd.DataFrame,
    *,
    metadata: PlotMetadata,
    collective: str,
    datatype: str,
    # error bar controls
    errorbars: str = "se",       # "none" | "se" | "ci"
    k: float = 1.96,             # multiplier for SE mode
    threshold: float = 0.5,      # error marker threshold (applies to chosen error metric)
    marker_loc: float = 0.05,    # base vertical offset for red marker
    output_dir: str | Path | None = None,
) -> Path:
    """
    Render the split bar plot that emphasizes small vs large differences.

    Expected columns:
      - normalized_mean
      - and either:
          * normalized_se (if errorbars="se")
          * normalized_ci_lower / normalized_ci_upper (if errorbars="ci")

    Backward compat:
      - if errorbars="se" and only normalized_std exists, it is treated as normalized_se.
    """
    if data.empty:
        raise ValueError("No data available for generate_cut_bar_plot.")

    # Backward compatibility shim (optional)
    if errorbars == "se" and "normalized_se" not in data.columns and "normalized_std" in data.columns:
        data = data.copy()
        data["normalized_se"] = data["normalized_std"]

    sorted_algos = sorted(data["algo_name"].unique().tolist(), key=sort_key)
    palette = build_tab10_palette(sorted_algos)

    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        sharex=True,
        gridspec_kw={"height_ratios": [1, 3]},
        figsize=(12, 8),
    )

    sns.barplot(
        ax=ax_top,
        data=data,
        x="buffer_size",
        y="normalized_mean",
        hue="algo_name",
        hue_order=sorted_algos,
        palette=palette,
        errorbar=None,
        edgecolor="black",
        linewidth=1.0,
    )
    sns.barplot(
        ax=ax_bot,
        data=data,
        x="buffer_size",
        y="normalized_mean",
        hue="algo_name",
        hue_order=sorted_algos,
        palette=palette,
        errorbar=None,
        edgecolor="black",
        linewidth=1.0,
    )

    if ax_top.get_legend():
        ax_top.get_legend().remove()

    y_min = 1.8
    y_max = min(data["normalized_mean"].max() * 1.1, 10.0)

    # Errorbars (selectable). Use different marker offsets for top vs bottom.
    draw_errorbars(
        ax_top,
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
        loc=(y_max - y_min) * 0.1,   # bigger offset on the top panel
    )
    draw_errorbars(
        ax_bot,
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

    ax_bot.set_ylim(0, y_min - 0.05)
    ax_top.set_ylim(y_min, y_max)

    # Indicate clipped bars on top panel
    top_limit = ax_top.get_ylim()[1]
    for container in ax_top.containers:
        for bar in container:
            if hasattr(bar, "get_height") and bar.get_height() > top_limit:
                x = bar.get_x() + bar.get_width() / 2.0
                ax_top.scatter(x, top_limit - 0.5, marker="^", color="black", s=100, zorder=4)

    ax_top.spines["bottom"].set_visible(True)
    ax_bot.spines["top"].set_visible(True)
    ax_top.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

    d = 0.005
    kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax_bot.transAxes)
    ax_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)


    style_axes(ax_top)
    style_axes(ax_bot)

    ax_bot.set_xlabel("Buffer Size", fontsize=15)
    ax_bot.set_ylabel("Normalized Mean Execution Time", fontsize=15)
    ax_top.set_ylabel("")

    if metadata.total_nodes == metadata.mpi_tasks:
        title = f"{metadata.system}, {collective.lower()}, {metadata.nnodes} nodes ({datatype})"
    else:
        title = (
            f"{metadata.system}, {collective.lower()}, {metadata.nnodes} nodes "
            f"({datatype}, {metadata.mpi_tasks} tasks)"
        )
    fig.suptitle(title, fontsize=18)

    ax_bot.set_xticks(ax_bot.get_xticks())
    new_labels = [format_bytes(tick.get_text()) for tick in ax_bot.get_xticklabels()]
    ax_bot.set_xticklabels(new_labels)

    handles, labels = ax_bot.get_legend_handles_labels()
    if handles:
        apply_adaptive_legend(ax_bot, handles=handles, labels=labels, loc="lower left")

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

    target_dir = _resolve_output_dir(metadata.system, output_dir)
    name = f"{collective.lower()}_{metadata.nnodes}_{datatype}_{metadata.timestamp}_barplot_cut_{errorbars}.pdf"
    full_path = target_dir / name
    plt.savefig(full_path, dpi=300)
    plt.close()
    return full_path
