# Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
# Licensed under the MIT License

import os, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import subprocess
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rcParams
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import gmean

from ..utils import ensure_dir, style_axes


matplotlib.rc("pdf", fonttype=42)  # To avoid issues with camera-ready submission
#rcParams['figure.figsize'] = 12,6.75
#rcParams['figure.figsize'] = 6.75,3.375
rcParams["figure.figsize"] = 3.375, 3.375
sns.set_style("whitegrid")
big_font_size = 18
small_font_size = 15
fmt = ".2f"
sbrn_palette = sns.color_palette("deep")  # ["#A6C8FF", ...]
sota_palette = [sbrn_palette[i] for i in range(len(sbrn_palette)) if sbrn_palette[i] != sns.xkcd_rgb["red"]]


metrics = ["mean", "median", "percentile_90"]


def human_readable_size(num_bytes):
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if num_bytes < 1024:
            return f"{int(num_bytes)} {unit}"
        num_bytes /= 1024
    return f"{int(num_bytes)} PiB"


@dataclass(slots=True)
class BoxplotConfig:
    system: str
    nnodes: Iterable[str]
    tasks_per_node: int = 1
    notes: str | None = None
    exclude: str | None = None
    metric: str = "mean"
    output_dir: str | Path | None = None

    def nodes_list(self) -> list[str]:
        if isinstance(self.nnodes, str):
            return [n.strip() for n in str(self.nnodes).split(",") if n.strip()]
        return [str(n) for n in self.nnodes]

def get_summaries(cfg, coll):
    metadata_file = f"results/{cfg.system}_metadata.csv"
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file {metadata_file} not found.")

    metadata = pd.read_csv(metadata_file)
    summaries: dict[str, str] = {}
    for nodes in cfg.nodes_list():
        if "tasks_per_node" in metadata.columns:
            filtered_metadata = metadata[
                (metadata["collective_type"].str.lower() == coll)
                & (metadata["nnodes"].astype(str) == str(nodes))
                & (metadata["tasks_per_node"].astype(int) == cfg.tasks_per_node)
            ]
        else:
            filtered_metadata = metadata[
                (metadata["collective_type"].str.lower() == coll)
                & (metadata["nnodes"].astype(str) == str(nodes))
            ]

        if cfg.notes:
            filtered_metadata = filtered_metadata[filtered_metadata["notes"].str.strip() == cfg.notes.strip()]
        else:
            filtered_metadata = filtered_metadata[filtered_metadata["notes"].isnull()]

        if filtered_metadata.empty:
            continue

        filtered_metadata = filtered_metadata.iloc[-1]
        summaries[nodes] = f"results/{cfg.system}/{filtered_metadata['timestamp']}/"
    return summaries


def get_summaries_df(cfg, coll):
    summaries = get_summaries(cfg, coll)
    df = pd.DataFrame()
    # Loop over the summaries
    for nodes, summary in summaries.items():
        # Create the summary, by calling the summarize_data.py script
        # Check if the summary already exists
        if not os.path.exists(summary + "/aggregated_results_summary.csv") or True:        
            subprocess.run([
                "python3",
                "./plot/summarize_data.py",
                "--result-dir",
                summary
            ],
            stdout=subprocess.DEVNULL)

        # Read the data
        s = pd.read_csv(summary + "/aggregated_results_summary.csv")        
        # Filter by collective type
        s = s[s["collective_type"].str.lower() == coll]      
        # Drop the rows where buffer_size is equal to 4 (we do not have them for all results :( )  
        s = s[s["buffer_size"] != 4]
        s["Nodes"] = nodes

        # Append s to df
        df = pd.concat([df, s], ignore_index=True)
    return df

def algo_name_to_family(algo_name, system):
    if algo_name.lower().startswith("bine"):
        return "Bine"    
    if system == "fugaku":
        if "recursive-doubling" in algo_name.lower():
            return "Binomial"
        elif "recursive_doubling" in algo_name.lower():
            return "Binomial"
        elif "nonoverlap" in algo_name.lower():
            return "Non Overlapping"
        elif "non-overlap" in algo_name.lower():
            return "Non Overlapping"
        elif "blacc" in algo_name.lower():
            return "Blacc"
        elif "doublespread" in algo_name.lower():
            return "Double Spread"
        elif "recursive-halving" in algo_name.lower():
            return "Binomial"
        elif "torus" in algo_name.lower():
            return "Ring" # Bucket-like
        elif "bruck" in algo_name.lower():
            return "Bruck"
        elif "default-default" == algo_name.lower():
            return "Default"
        elif "neighbor" in algo_name.lower():
            return "Neighbor"
        elif "ring" in algo_name.lower():
            return "Ring"
        elif "linear" in algo_name.lower():
            return "Linear"
        elif "gtbc" in algo_name.lower():
            return "GTBC"
        elif "trix" in algo_name.lower():
            return "Trix"
        elif "rdbc" in algo_name.lower():
            return "RDBC"
        elif "pairwise" in algo_name.lower():
            return "Pairwise"
        elif "knomial" in algo_name.lower():
            return "Knomial"
        elif "trinaryx" in algo_name.lower():
            return "Trix"
        elif "split-binary" in algo_name.lower():
            return "Binary"
        elif "binomial" in algo_name.lower():
            return "Binomial"
        elif "bruck" in algo_name.lower():
            return "Bruck"
        elif "binary" in algo_name.lower():
            return "Binary"            
        elif "bintree" in algo_name.lower():
            return "Binary"
        elif "crp" in algo_name.lower():
            return "CRP"
        elif "use-bcast" in algo_name.lower():
            return "Use Bcast"
        elif "simple" in algo_name.lower():
            return "Simple"
        elif "pipeline" in algo_name.lower():
            return "Pipeline"
        elif "chain" in algo_name.lower():
            return "Chain"
    elif system == "leonardo" or system == "mare_nostrum":
        if "default_ompi" == algo_name.lower():
            return "Default"
        elif "recursive_doubling" in algo_name.lower():
            return "Binomial"
        elif "ring" in algo_name.lower():
            return "Ring"
        elif "rabenseifner" in algo_name.lower():
            return "Binomial"
        elif "binary" in algo_name.lower():
            return "Binary"
        elif "binomial" in algo_name.lower():
            return "Binomial"
        elif "in_order" in algo_name.lower():
            return "In Order"
        elif "bruck" in algo_name.lower():
            return "Bruck"
        elif "knomial" in algo_name.lower():
            return "Knomial"
        elif "neighbor" in algo_name.lower():
            return "Neighbor"
        elif "linear" in algo_name.lower():
            return "Linear"
        elif "pairwise" in algo_name.lower():
            return "Pairwise"
        elif "recursive" in algo_name.lower():
            return "Binomial"
        elif "scatter_allgather" in algo_name.lower():
            return "Binomial"
        elif "sparbit" in algo_name.lower():
            return "Binomial"
        elif "allgather_reduce" in algo_name.lower():
            return "Allgather Reduce"
    elif system == "lumi":
        if "binomial_mpich" in algo_name.lower():
            return "Binomial"
        elif "default_mpich" in algo_name.lower():
            return "Default"
        elif "recursive_doubling" in algo_name.lower():
            return "Binomial"
        elif "ring" in algo_name.lower():
            return "Ring"
        elif "rabenseifner" in algo_name.lower():
            return "Binomial"
        elif "binary" in algo_name.lower():
            return "Binary"
        elif "binomial" in algo_name.lower():
            return "Binomial"
        elif "recursive_halving" in algo_name.lower():
            return "Binomial"
        elif "non_blocking" in algo_name.lower():
            return "Non Blocking"        
        elif "non_commutativ" in algo_name.lower():
            return "Non Commutative"
        elif "bruck" in algo_name.lower():
            return "Bruck"
        elif "scatter_allgather" in algo_name.lower():
            return "Binomial"
        elif "knomial" in algo_name.lower():
            return "Knomial"
        elif "distance_doubling" in algo_name.lower():
            return "Binomial"
        elif "neighbor" in algo_name.lower():
            return "Neighbor"
        elif "scattered_mpich" in algo_name.lower():
            return "Scattered"
        elif "pairwise" in algo_name.lower():
            return "Pairwise"
        elif "sparbit" in algo_name.lower():
            return "Binomial"

    # error
    raise ValueError(f"Unknown algorithm {algo_name} for system {system}")
    

def augment_df(df, metric):
    reference = "Bine"
    # Step 1: Create an empty list to hold the new rows
    new_data = []

    # For each (buffer_size, nodes) group the data so that for eacha algo_family we only keep the entry with the highest bandwidth_mean
    df = df.loc[df.groupby(['buffer_size', 'Nodes', 'algo_family'])['bandwidth_' + metric].idxmax()]
    total_cases = 0
    win_cases = 0
    # Step 2: Group by 'buffer_size' and 'Nodes'
    for (buffer_size, nodes), group in df.groupby(['buffer_size', 'Nodes']):        
        # Step 3: Get the best algorithm
        best_algo_row = group.loc[group['bandwidth_' + metric].idxmax()]
        best_algo = best_algo_row['algo_family']
        total_cases += 1
        # Step 4: Get the second best algorithm (excluding the best one)
        tmp = group[group['algo_family'] != best_algo]['bandwidth_' + metric]
        if tmp.empty:
            print(f"Warning: No second best algorithm found for buffer_size {buffer_size} and nodes {nodes}. Skipping.", file=sys.stderr)
            continue
        second_best_algo_row = group.loc[tmp.idxmax()]
        second_best_algo = second_best_algo_row['algo_family']

        # Get Bine bandwidth_mean for this group
        bine_row = group.loc[group['algo_family'] == reference]
        if bine_row.empty:
            print(f"Warning: No Bine algorithm found for buffer_size {buffer_size} and nodes {nodes}. Skipping.", file=sys.stderr)
            continue
        
        bine_bandwidth_mean = bine_row['bandwidth_' + metric].values[0]

        #print(f"Buffer size: {buffer_size}, Nodes: {nodes}, Best algo: {best_algo}, Second best algo: {second_best_algo}")
        #print(group)

        ratio = bine_bandwidth_mean / best_algo_row['bandwidth_' + metric]
        # Truncate to 1 decimal place
        ratio = round(ratio, 1)
        
        if best_algo == reference:
            cell = best_algo_row['bandwidth_' + metric] / second_best_algo_row['bandwidth_' + metric]  
            win_cases += 1
        elif ratio >= 1.0:
            cell = ratio  
            win_cases += 1       
        else:
            continue


        # Step 6: Append the data for this group (including old columns)
        new_data.append({
            'buffer_size': buffer_size,
            'Nodes': nodes,
            #'algo_family': best_algo,
            #'bandwidth_' + metric: best_algo_row['bandwidth_' + metric],
            'cell': cell,
        })

    # Step 7: Create a new DataFrame
    return (pd.DataFrame(new_data), (win_cases / float(total_cases)) * 100.0)

def algo_to_family(df, cfg):
    # Convert algo_name to algo_family
    df["algo_family"] = df["algo_name"].apply(lambda x: algo_name_to_family(x, cfg.system))
    # Drop algo_name
    df = df.drop(columns=["algo_name"])
    return df

def family_name_to_letter_color(family_name):
    if family_name == "Default":
        return ("D", sota_palette[0])
    elif family_name == "Binomial":
        return ("N", sota_palette[1])
    elif family_name == "Bruck":
        return ("K", sota_palette[2])
    elif family_name == "Ring":
        return ("R", sota_palette[3])
    elif family_name == "Neighbor":
        return ("H", sota_palette[4])
    elif family_name == "Linear":
        return ("L", sota_palette[5])
    elif family_name == "GTBC":
        return ("G", sota_palette[6])
    elif family_name == "Pairwise":
        return ("P", sota_palette[6])
    elif family_name == "In Order":
        return ("I", sota_palette[6])
    elif family_name == "Knomial":
        return ("O", sota_palette[6])    
    elif family_name == "Binary":
        return ("Y", sota_palette[6])    
    elif family_name == "Non Blocking":
        return ("B", sota_palette[6])    
    elif family_name == "Non Commutative":
        return ("C", sota_palette[6])
    elif family_name == "Scattered":
        return ("S", sota_palette[6])
    elif family_name == "Trix":
        return ("X", sota_palette[6])
    elif family_name == "Use Bcast":
        return ("U", sota_palette[6])
    elif family_name == "Simple":
        return ("M", sota_palette[6])
    elif family_name == "Blacc":
        return ("A", sota_palette[6])
    elif family_name == "CRP":
        return ("Z", sota_palette[6])
    else:
        # error
        raise ValueError(f"Unknown algorithm family {family_name}")

def get_data_coll(cfg, coll):
    df = get_summaries_df(cfg, coll)
          
    # Drop the columns I do not need
    df = df[["buffer_size", "Nodes", "algo_name", "mean", "median", "percentile_90"]]

    # If system name is "fugaku", drop all the algo_name starting with uppercase "RECDOUB"
    if cfg.system == "fugaku":
        df = df[~df["algo_name"].str.startswith("RECDOUB")]

    if cfg.exclude:
        df = df[~df["algo_name"].str.contains(cfg.exclude, case=False)]
    
    #df = df[~df["algo_name"].str.contains("default_mpich", case=False)]
    
    # Compute the bandwidth for each metric
    for m in metrics:
        if m == cfg.metric:
            df["bandwidth_" + m] = ((df["buffer_size"]*8.0)/(1000.0*1000*1000)) / (df[m].astype(float) / (1000.0*1000*1000))
    
    # drop all the metrics
    for m in metrics:
        df = df.drop(columns=[m])
    # print full df, no limts on cols or rows
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)

    df = algo_to_family(df, cfg)
    return augment_df(df, cfg.metric)

def generate_boxplot(cfg: BoxplotConfig) -> Path:
    df = pd.DataFrame()
    collectives = ["allreduce", "allgather", "reduce_scatter", "alltoall", "bcast", "reduce", "gather", "scatter"]
    for coll in collectives:
        df_coll, wins = get_data_coll(cfg, coll)
        label = "Red.-Scat." if coll.lower() == "reduce_scatter" else coll
        if df_coll.empty:
            continue
        df_coll["Collective"] = label.capitalize() + "\n(" + str(int(wins)) + "%)"
        df_coll = df_coll.drop(columns=["buffer_size", "Nodes"])
        df_coll = df_coll.rename(columns={"cell": "Improvement (%)"})
        df_coll["Improvement (%)"] = (df_coll["Improvement (%)"] - 1) * 100.0
        df = pd.concat([df, df_coll], ignore_index=True)

    if df.empty:
        raise RuntimeError("No data available to build the boxplot.")

    mean_props = {
        "marker": "o",
        "markerfacecolor": "black",
        "markeredgecolor": "black",
        "markersize": 8,
    }

    plt.figure()
    palette = sns.color_palette("deep", n_colors=df["Collective"].nunique())
    ax = sns.boxplot(
        data=df,
        y="Collective",
        x="Improvement (%)",
        hue="Collective",
        showfliers=True,
        showmeans=True,
        meanprops=mean_props,
        palette=palette,
        dodge=False,
    )
    if ax.legend_:
        ax.legend_.remove()
    plt.ylabel("")
    style_axes(ax)

    out_dir = cfg.output_dir or (Path("plot") / cfg.system)
    ensure_dir(out_dir)
    outfile = Path(out_dir) / "boxplot.pdf"
    plt.savefig(outfile, bbox_inches="tight")
    plt.close()
    return outfile
