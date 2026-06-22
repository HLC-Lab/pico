# Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
# Licensed under the MIT License

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams

from ..utils import build_ratio_colormap, ensure_dir, human_readable_size

# Match legacy styling defaults
matplotlib.rc("pdf", fonttype=42)
rcParams["figure.figsize"] = 6.75, 6.75

BIG_FONT_SIZE = 18
SMALL_FONT_SIZE = 15
FMT = ".2f"
SBRN_PALETTE = sns.color_palette("deep")
SOTA_PALETTE = [color for color in SBRN_PALETTE if color != sns.xkcd_rgb["red"]]
METRICS = ["mean", "median", "percentile_90"]


@dataclass(slots=True)
class FamilyHeatmapConfig:
    system: str
    collective: str
    nnodes: Iterable[str]
    tasks_per_node: int = 1
    notes: str | None = None
    exclude: str | None = None
    metric: str = "mean"
    reference: str = "all"
    hide_y_labels: bool = False
    output_dir: str | Path | None = None

    def sorted_nodes(self) -> list[str]:
        return [str(n) for n in self.nnodes]


def _metadata_filter(metadata: pd.DataFrame, cfg: FamilyHeatmapConfig, nodes: str) -> pd.DataFrame:
    if "tasks_per_node" in metadata.columns:
        filtered = metadata[
            (metadata["collective_type"].str.lower() == cfg.collective.lower())
            & (metadata["nnodes"].astype(str) == str(nodes))
            & (metadata["tasks_per_node"].astype(int) == cfg.tasks_per_node)
        ]
    else:
        filtered = metadata[
            (metadata["collective_type"].str.lower() == cfg.collective.lower())
            & (metadata["nnodes"].astype(str) == str(nodes))
        ]

    if cfg.notes:
        filtered = filtered[filtered["notes"].str.strip() == cfg.notes.strip()]
    else:
        filtered = filtered[filtered["notes"].isnull()]

    if cfg.system == "leonardo":
        filtered = filtered[~filtered["mpi_lib"].str.contains("OMPI_BINE", case=False)]
    return filtered


def _discover_summaries(cfg: FamilyHeatmapConfig) -> dict[str, str]:
    metadata_file = f"results/{cfg.system}_metadata.csv"
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file {metadata_file} not found.")

    metadata = pd.read_csv(metadata_file)
    summaries: dict[str, str] = {}

    for nodes in cfg.sorted_nodes():
        filtered = _metadata_filter(metadata, cfg, nodes)
        if filtered.empty:
            raise RuntimeError(f"Metadata file {metadata_file} does not contain the requested data.")
        last_entry = filtered.iloc[-1]
        summaries[nodes] = f"results/{cfg.system}/{last_entry['timestamp']}/"
    return summaries


def _load_summaries(cfg: FamilyHeatmapConfig) -> pd.DataFrame:
    summaries = _discover_summaries(cfg)
    frames: list[pd.DataFrame] = []

    for nodes, summary_dir in summaries.items():
        summary_path = os.path.join(summary_dir, "aggregated_results_summary.csv")
        if not os.path.exists(summary_path):
            subprocess.run(
                [
                    "python3",
                    "./plot/summarize_data.py",
                    "--result-dir",
                    summary_dir,
                ],
                stdout=subprocess.DEVNULL,
                check=False,
            )
        df = pd.read_csv(summary_path)
        df = df[df["collective_type"].str.lower() == cfg.collective.lower()]
        df = df[df["buffer_size"] != 4]
        df["Nodes"] = nodes
        frames.append(df)

    if not frames:
        raise RuntimeError("No data found for the requested configuration.")
    return pd.concat(frames, ignore_index=True)


def _algo_name_to_family(algo_name: str, system: str) -> str:
    lower = algo_name.lower()
    if lower.startswith("bine"):
        return "Bine"

    if system == "fugaku":
        mapping = [
            ("recursive-doubling", "Binomial"),
            ("recursive_doubling", "Binomial"),
            ("nonoverlap", "Non Overlapping"),
            ("non-overlap", "Non Overlapping"),
            ("blacc", "Blacc"),
            ("doublespread", "Double Spread"),
            ("recursive-halving", "Binomial"),
            ("torus", "Ring"),
            ("bruck", "Bruck"),
            ("default-default", "Default"),
            ("neighbor", "Neighbor"),
            ("ring", "Ring"),
            ("linear", "Linear"),
            ("gtbc", "GTBC"),
            ("trix", "Trix"),
            ("rdbc", "RDBC"),
            ("pairwise", "Pairwise"),
            ("knomial", "Knomial"),
            ("trinaryx", "Trix"),
            ("split-binary", "Binary"),
            ("binomial", "Binomial"),
            ("binary", "Binary"),
            ("bintree", "Binary"),
            ("crp", "CRP"),
            ("use-bcast", "Use Bcast"),
            ("simple", "Simple"),
            ("pipeline", "Pipeline"),
            ("chain", "Chain"),
        ]
    elif system in {"leonardo", "mare_nostrum"}:
        mapping = [
            ("default_ompi", "Default"),
            ("recursive_doubling", "Binomial"),
            ("ring", "Ring"),
            ("rabenseifner", "Binomial"),
            ("binary", "Binary"),
            ("binomial", "Binomial"),
            ("in_order", "In Order"),
            ("bruck", "Bruck"),
            ("knomial", "Knomial"),
            ("neighbor", "Neighbor"),
            ("linear", "Linear"),
            ("pairwise", "Pairwise"),
            ("recursive", "Binomial"),
            ("scatter_allgather", "Binomial"),
            ("sparbit", "Binomial"),
            ("nccl_collnet", "CollNet"),
            ("nccl_nvls", "NVLS"),
            ("pat", "Binomial"),
            ("allreduce_hier", "Bine"),
            ("nccl_tree", "Tree"),
            ("bine_lat", "Bine"),
        ]
    elif system == "lumi":
        mapping = [
            ("binomial_mpich", "Binomial"),
            ("default_mpich", "Default"),
            ("recursive_doubling", "Binomial"),
            ("ring", "Ring"),
            ("rabenseifner", "Binomial"),
            ("binary", "Binary"),
            ("binomial", "Binomial"),
            ("recursive_halving", "Binomial"),
            ("non_blocking", "Non Blocking"),
            ("non_commutativ", "Non Commutative"),
            ("bruck", "Bruck"),
            ("scatter_allgather", "Binomial"),
            ("knomial", "Knomial"),
            ("distance_doubling", "Binomial"),
            ("neighbor", "Neighbor"),
            ("scattered_mpich", "Scattered"),
            ("pairwise", "Pairwise"),
            ("sparbit", "Binomial"),
        ]
    else:
        mapping = []

    for marker, family in mapping:
        if marker in lower:
            return family
    raise ValueError(f"Unknown algorithm {algo_name} for system {system}")


def _augment_dataframe(df: pd.DataFrame, cfg: FamilyHeatmapConfig) -> pd.DataFrame:
    metric = cfg.metric
    df = df.loc[df.groupby(["buffer_size", "Nodes", "algo_family"])[f"bandwidth_{metric}"].idxmax()]

    new_rows = []
    for (buffer_size, nodes), group in df.groupby(["buffer_size", "Nodes"]):
        best = group.loc[group[f"bandwidth_{metric}"].idxmax()]

        others = group[group["algo_family"] != best["algo_family"]][f"bandwidth_{metric}"]
        if others.empty:
            continue
        second_best = group.loc[others.idxmax()]

        reference_row = group[group["algo_family"] == "Bine"]
        if reference_row.empty:
            continue
        reference_value = reference_row[f"bandwidth_{metric}"].values[0]

        ratio = reference_value / best[f"bandwidth_{metric}"]
        ratio = round(ratio, 1)

        if best["algo_family"] == "Bine":
            cell = best[f"bandwidth_{metric}"] / second_best[f"bandwidth_{metric}"]
        elif ratio >= 1.0:
            cell = ratio
        else:
            cell = best["algo_family"]
        new_rows.append(
            {
                "buffer_size": buffer_size,
                "Nodes": nodes,
                "cell": cell,
            }
        )

    return pd.DataFrame(new_rows)


def _algo_to_family(df: pd.DataFrame, cfg: FamilyHeatmapConfig) -> pd.DataFrame:
    df = df.copy()
    df["algo_family"] = df["algo_name"].apply(lambda name: _algo_name_to_family(name, cfg.system))
    df = df.drop(columns=["algo_name"])
    return df


def _family_name_to_letter_color(family_name: str) -> tuple[str, tuple[float, float, float]]:
    if family_name == "Default":
        return ("D", SOTA_PALETTE[0])
    if family_name == "Binomial":
        return ("N", SOTA_PALETTE[1])
    if family_name == "Bruck":
        return ("K", SOTA_PALETTE[2])
    if family_name == "Ring":
        return ("R", SOTA_PALETTE[3])
    if family_name == "Neighbor":
        return ("H", SOTA_PALETTE[4])
    if family_name == "Linear":
        return ("L", SOTA_PALETTE[5])
    if family_name == "GTBC":
        return ("G", SOTA_PALETTE[6])
    if family_name == "Pairwise":
        return ("P", SOTA_PALETTE[6])
    if family_name == "In Order":
        return ("I", SOTA_PALETTE[6])
    if family_name == "Knomial":
        return ("O", SOTA_PALETTE[6])
    if family_name == "Binary":
        return ("Y", SOTA_PALETTE[6])
    if family_name == "Non Blocking":
        return ("B", SOTA_PALETTE[6])
    if family_name == "Non Commutative":
        return ("C", SOTA_PALETTE[6])
    if family_name == "Scattered":
        return ("S", SOTA_PALETTE[6])
    if family_name == "Trix":
        return ("X", SOTA_PALETTE[6])
    if family_name == "Use Bcast":
        return ("U", SOTA_PALETTE[6])
    if family_name == "Simple":
        return ("M", SOTA_PALETTE[6])
    if family_name == "Blacc":
        return ("A", SOTA_PALETTE[6])
    if family_name == "CRP":
        return ("Z", SOTA_PALETTE[6])
    if family_name == "CollNet":
        return ("C", SOTA_PALETTE[6])
    if family_name == "NVLS":
        return ("V", SOTA_PALETTE[6])
    if family_name == "Tree":
        return ("T", SOTA_PALETTE[6])
    if family_name == "Bine":
        return ("B", SOTA_PALETTE[0])
    return ("?", SOTA_PALETTE[0])


def generate_family_heatmap(cfg: FamilyHeatmapConfig) -> Path:
    df = _load_summaries(cfg)
    df = df[["buffer_size", "Nodes", "algo_name", "mean", "median", "percentile_90"]]

    if cfg.exclude:
        df = df[~df["algo_name"].str.contains(cfg.exclude, case=False)]

    for metric in METRICS:
        if metric == cfg.metric:
            df[f"bandwidth_{metric}"] = ((df["buffer_size"] * 8.0) / (1e9)) / (df[metric].astype(float) / 1e9)
    for metric in METRICS:
        df = df.drop(columns=[metric])

    df = _algo_to_family(df, cfg)
    df = _augment_dataframe(df, cfg)

    numeric_df = df.copy()
    numeric_df["cell"] = pd.to_numeric(df["cell"], errors="coerce")

    heatmap_data = numeric_df.pivot(index="buffer_size", columns="Nodes", values="cell")
    heatmap_data = heatmap_data[cfg.sorted_nodes()]
    text_data = df.pivot(index="buffer_size", columns="Nodes", values="cell")
    text_data = text_data[cfg.sorted_nodes()]

    plt.figure()
    cmap = build_ratio_colormap()
    ax = sns.heatmap(
        heatmap_data,
        annot=True,
        cmap=cmap,
        fmt=FMT,
        center=1,
        cbar=True,
        annot_kws={"size": BIG_FONT_SIZE, "weight": "bold"},
        cbar_kws={"orientation": "horizontal", "location": "top", "aspect": 40},
    )

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=SMALL_FONT_SIZE)

    for i in range(text_data.shape[0]):
        for j in range(text_data.shape[1]):
            val = text_data.iloc[i, j]
            if isinstance(val, str):
                letter, color = _family_name_to_letter_color(val)
                plt.text(j + 0.5, i + 0.5, letter, ha="center", va="center", color=color, weight="bold", fontsize=BIG_FONT_SIZE)
            elif pd.isna(val):
                plt.text(j + 0.5, i + 0.5, "N/A", ha="center", va="center", color="black", weight="bold", fontsize=BIG_FONT_SIZE)

    for i in range(text_data.shape[0]):
        for j in range(text_data.shape[1]):
            if isinstance(text_data.iloc[i, j], str):
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color="#f0f0f0", lw=0, zorder=-1))

    buffer_sizes = text_data.index.astype(int).tolist()
    buffer_sizes.sort()
    buffer_labels = [human_readable_size(size) for size in buffer_sizes]
    plt.yticks(ticks=np.arange(len(buffer_labels)) + 0.5, labels=buffer_labels, fontsize=SMALL_FONT_SIZE)

    plt.xlabel("# Nodes", fontsize=BIG_FONT_SIZE)
    if not cfg.hide_y_labels:
        plt.ylabel("Vector Size", fontsize=BIG_FONT_SIZE)
    else:
        plt.ylabel("")
    plt.xticks(fontsize=SMALL_FONT_SIZE, rotation=0)

    outdir = cfg.output_dir
    if outdir is None:
        outdir = Path("plot") / cfg.system / "heatmaps" / cfg.collective
    target_dir = ensure_dir(outdir)

    args_str_parts = []
    for key, value in [
        ("nnodes", ",".join(cfg.sorted_nodes())),
        ("tasks", cfg.tasks_per_node),
        ("notes", cfg.notes),
        ("exclude", cfg.exclude),
        ("metric", cfg.metric),
    ]:
        if value is None:
            continue
        args_str_parts.append(f"{key}_{value}")
    args_str = "_".join(args_str_parts)
    outfile = target_dir / f"{args_str}.pdf"
    plt.savefig(outfile, bbox_inches="tight")
    plt.close()
    return outfile
