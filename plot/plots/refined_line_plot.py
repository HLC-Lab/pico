# Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
# Licensed under the MIT License

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..utils import apply_adaptive_legend, ensure_dir, style_axes
from .refined_loader import RefinedDataset


sns.set_style("whitegrid")
sns.set_context("paper")


def generate_refined_line_plot(
    dataset: RefinedDataset,
    name: str,
    *,
    output_dir: str | Path | None = None,
) -> Path:
    df = pd.DataFrame(dataset.data)
    if df.empty:
        raise ValueError("Dataset is empty; nothing to plot.")

    df['cluster_collective'] = df['Cluster'].astype(str) + '_' + df['collective'].astype(str)
    order = (
        df.sort_values('message_bytes')
          .drop_duplicates('Message')['Message']
          .tolist()
    )
    df['Message'] = pd.Categorical(df['Message'], categories=order, ordered=True)

    palette = sns.color_palette("tab10", n_colors=df['cluster_collective'].nunique())

    plt.figure(figsize=(12, 8))
    ax = sns.lineplot(
        data=df,
        x='Message',
        y='bandwidth',
        hue='cluster_collective',
        style='cluster_collective',
        markers=True,
        markersize=9,
        linewidth=2,
        estimator='mean',
        errorbar=('sd', 0.5),
        palette=palette,
    )

    ax.set_ylabel('Bandwidth (Gb/s)', fontsize=15)
    ax.set_xlabel('Message Size', fontsize=15)
    ax.set_title(name, fontsize=18)
    ax.tick_params(axis='both', which='major')

    handles, labels = ax.get_legend_handles_labels()
    new_labels = [' '.join(w.capitalize() for w in lbl.replace('_', ' ').split()) for lbl in labels]
    apply_adaptive_legend(ax, handles=handles, labels=new_labels, loc='upper left', frameon=True)
    style_axes(ax)

    plt.tight_layout()

    target_dir = ensure_dir(output_dir or Path('plots'))
    out_path = Path(target_dir) / f"{name.replace(',', '').replace(' ', '_').lower()}_sd_line.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    return out_path
