# Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
# Licensed under the MIT License

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def ensure_dir(path: str | Path) -> Path:
    """
    Create ``path`` if it does not already exist and return it as ``Path``.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def format_bytes(value: float | int | str) -> str:
    """
    Convert raw byte counts into a human friendly representation.
    """
    try:
        x = float(value)
    except (ValueError, TypeError):
        return str(value)

    if x >= 1024**2:
        return f"{x / 1024**2:.0f} MiB"
    if x >= 1024:
        return f"{x / 1024:.0f} KiB"
    return f"{x:.0f} B"


def human_readable_size(num_bytes: float | int) -> str:
    """
    Convert bytes to a <value> <unit> string using IEC units.
    """
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{int(round(value))} {unit}"
        value /= 1024.0
    return f"{int(round(value))} PiB"


def sort_key(algo: str) -> tuple[int, str]:
    """
    Stable ordering for algorithm names so plots look consistent.
    """
    if algo.startswith("default"):
        return (0, algo)
    if not algo.endswith("over"):
        return (1, algo)
    if "bine" not in algo:
        return (2, algo)
    return (3, algo)


def build_tab10_palette(sorted_algos: Iterable[str]) -> Mapping[str, tuple[float, float, float]]:
    """
    Map each algorithm to a colour from matplotlib's ``tab10`` palette.
    The palette is cycled if more than ten algorithms are requested.
    """
    colors = plt.get_cmap("tab10").colors
    return {algo: colors[i % len(colors)] for i, algo in enumerate(sorted_algos)}


def style_axes(ax) -> None:
    """
    Apply the shared axis styling used by non-heatmap plots.
    """
    ax.set_facecolor("#fdfdfd")
    ax.grid(True, linestyle=":", linewidth=0.8, color="#999999", alpha=0.7)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.2)


def adaptive_legend_layout(num_entries: int) -> tuple[int, int]:
    """
    Choose legend columns and fontsize from the number of legend entries.
    """
    if num_entries > 10:
        return 3, 8
    if num_entries > 5:
        if num_entries > 8:
            return 2, 10
        return 2, 14
    if num_entries > 3:
        return 1, 16
    return 1, 20


def apply_adaptive_legend(
    ax,
    *,
    handles=None,
    labels: Sequence[str] | None = None,
    loc: str = "best",
    frameon: bool | None = None,
    format_label: Callable[[str], str] | None = None,
):
    """
    Apply a legend to ``ax`` using adaptive columns/font size based on entries.
    """
    if handles is None or labels is None:
        handles, labels = ax.get_legend_handles_labels()

    if not handles:
        return None

    legend_labels = list(labels)
    if format_label is not None:
        legend_labels = [format_label(label) for label in legend_labels]

    ncols, fontsize = adaptive_legend_layout(len(handles))
    kwargs = {"ncol": ncols, "loc": loc, "fontsize": fontsize}
    if frameon is not None:
        kwargs["frameon"] = frameon

    return ax.legend(handles, legend_labels, **kwargs)


def draw_errorbars(
    ax,
    data: pd.DataFrame,
    sorted_algos: Iterable[str],
    *,
    mode: str = "se",                 # "none" | "se" | "ci"
    x_col: str = "buffer_size",
    algo_col: str = "algo_name",
    y_col: str = "normalized_mean",
    # SE mode
    se_col: str = "normalized_se",
    k: float = 1.96,                  # 1.96 ~ 95% (normal approx). Use 1.0 for ±SE
    # CI mode
    ci_lower_col: str = "normalized_ci_lower",
    ci_upper_col: str = "normalized_ci_upper",
    # flagging
    threshold: float = np.inf,        # if error > threshold, draw a red marker instead
    loc: float = 0.05,                # vertical offset for the red marker
    marker_size: float = 50.0,
) -> None:
    """
    Draw error bars on a grouped bar plot.

    Assumptions:
      - Bars were drawn in algorithm order matching `sorted_algos`
        (i.e., container i corresponds to sorted_algos[i]).
      - Within each algo, bars are in x_col order (we enforce this by sorting rows).

    mode:
      - "none": do nothing
      - "se":  symmetric yerr = k * normalized_se
      - "ci":  asymmetric yerr derived from (normalized_ci_lower, normalized_ci_upper)
    """
    mode = mode.lower().strip()
    if mode == "none":
        return
    if mode not in {"se", "ci"}:
        raise ValueError("mode must be one of: 'none', 'se', 'ci'")

    containers = ax.containers
    x_order = sorted(data[x_col].unique())

    for idx, algo in enumerate(sorted_algos):
        if idx >= len(containers):
            continue

        container = containers[idx]

        algo_group = data[data[algo_col] == algo].copy()
        # Ensure row order matches bar order
        algo_group[x_col] = pd.Categorical(algo_group[x_col], categories=x_order, ordered=True)
        algo_group = algo_group.sort_values(x_col)

        for bar, (_, row) in zip(container, algo_group.iterrows()):
            x = bar.get_x() + bar.get_width() / 2.0
            y = float(bar.get_height())

            if mode == "se":
                se = float(row.get(se_col, 0.0))
                yerr = k * se
                if not np.isfinite(yerr) or yerr <= 0:
                    continue

                if yerr > threshold:
                    ax.scatter(x, y + loc, color="red", s=marker_size, zorder=5)
                else:
                    ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="black", capsize=3, zorder=4)

            elif mode == "ci":
                lo = float(row.get(ci_lower_col, y))
                hi = float(row.get(ci_upper_col, y))
                if not (np.isfinite(lo) and np.isfinite(hi)):
                    continue

                err_lo = max(0.0, y - lo)
                err_hi = max(0.0, hi - y)
                worst = max(err_lo, err_hi)

                if worst > threshold:
                    ax.scatter(x, y + loc, color="red", s=marker_size, zorder=5)
                else:
                    ax.errorbar(
                        x, y,
                        yerr=np.array([[err_lo], [err_hi]]),
                        fmt="none",
                        ecolor="black",
                        capsize=3,
                        zorder=4,
                    )


def format_time_units_ns(value, _pos) -> str:
    """
    Format nanosecond tick labels using sensible units.
    """
    if value < 1_000:
        return f"{int(value)}ns" if float(value).is_integer() else f"{value:.1f} ns"
    if value < 1_000_000:
        val = value / 1_000
        return f"{int(val)}µs" if float(val).is_integer() else f"{val:.1f} µs"

    val = value / 1_000_000
    return f"{int(val)}ms" if float(val).is_integer() else f"{val:.1f} ms"


def build_ratio_colormap() -> LinearSegmentedColormap:
    """
    Convenience to reproduce the red -> white -> green map used by legacy heatmaps.
    """
    return LinearSegmentedColormap.from_list("RedGreen", ["darkred", "white", "darkgreen"])


@dataclass(slots=True)
class PlotMetadata:
    """
    Lightweight container for summary metadata (system, collective, ...).
    """

    system: str
    timestamp: str
    mpi_lib: str
    nnodes: str
    tasks_per_node: int
    gpu_lib: str

    @property
    def total_nodes(self) -> int:
        value = self.nnodes
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            if "x" in value:
                product = 1
                for chunk in value.lower().split("x"):
                    if not chunk:
                        continue
                    product *= int(chunk)
                return product
            try:
                return int(value)
            except ValueError as exc:
                raise ValueError(f"Cannot parse nnodes value '{value}'") from exc
        return int(value)

    @property
    def mpi_tasks(self) -> int:
        return self.total_nodes * self.tasks_per_node
