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


def generate_stacked_latency_bars(
    dataset: RefinedDataset,
    name: str,
    *,
    sizes: tuple[str, ...],
    output_dir: str | Path | None = None,
) -> Path:
    df = pd.DataFrame(dataset.data)
    if df.empty:
        raise ValueError("Dataset is empty; nothing to plot.")

    df = df[df['collective'] == 'allreduce'].copy()

    mean_lat = (
        df.groupby(['Message', 'message_bytes', 'Cluster'], observed=True)
          .agg(latency_mean=('latency', 'mean'))
          .reset_index()
    )

    want_clusters = ['baseline', 'op_null', 'no_memcpy', 'no_memcpy_op_null']
    piv = (
        mean_lat[mean_lat['Cluster'].isin(want_clusters)]
            .pivot_table(index=['Message', 'message_bytes'], columns='Cluster', values='latency_mean', aggfunc='mean')
            .reset_index()
    )

    sizes_present = [s for s in sizes if s in piv['Message'].values]
    if not sizes_present:
        raise ValueError("Requested message sizes not found in data.")

    piv = piv[piv['Message'].isin(sizes_present)].copy().sort_values('message_bytes')
    x_order = piv['Message'].tolist()

    for col in ['baseline', 'op_null', 'no_memcpy']:
        if col not in piv:
            piv[col] = pd.NA

    memcpy = (piv['baseline'] - piv['no_memcpy']).clip(lower=0)
    reduction = (piv['baseline'] - piv['op_null']).clip(lower=0)
    other = (piv['baseline'] - (memcpy + reduction)).clip(lower=0)

    base = piv['baseline']
    pct_memcpy = (memcpy / base) * 100.0
    pct_reduction = (reduction / base) * 100.0
    pct_other = (other / base) * 100.0

    bars = pd.DataFrame({
        'Message': piv['Message'],
        'pct_memcpy': pct_memcpy,
        'pct_reduction': pct_reduction,
        'pct_other': pct_other,
    }).set_index('Message').loc[x_order]

    fig, ax = plt.subplots(figsize=(12, 8))
    pal = sns.color_palette("tab10", 3)
    ax.bar(bars.index, bars['pct_other'].values, label='Network Communications', width=0.7, color=pal[0])
    ax.bar(bars.index, bars['pct_reduction'].values, bottom=bars['pct_other'].values,
           label='Reduction Computation', width=0.7, color=pal[1])
    ax.bar(
        bars.index,
        bars['pct_memcpy'].values,
        bottom=(bars['pct_other'].values + bars['pct_reduction'].values),
        label='Data Movements', width=0.7, color=pal[2]
    )

    ax.set_ylabel('Latency share (% of baseline)', fontsize=15)
    ax.set_xlabel('Message Size', fontsize=15)
    ax.set_title(f'{name} – Per step breakdown', fontsize=18)
    ax.set_ylim(0, 100)
    ax.tick_params(axis='x', rotation=0)
    apply_adaptive_legend(ax, loc='lower left', frameon=True)
    style_axes(ax)

    plt.tight_layout()
    target_dir = ensure_dir(output_dir or Path('plots'))
    out_path = Path(target_dir) / f"{name.replace(',', '').replace(' ', '_').lower()}_stacked_latency.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    return out_path
