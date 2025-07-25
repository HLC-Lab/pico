# Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
# Licensed under the MIT License

import os, sys, argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import subprocess
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rcParams
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

matplotlib.rc('pdf', fonttype=42) # To avoid issues with camera-ready submission
#rcParams['figure.figsize'] = 12,6.75
rcParams['figure.figsize'] = 6.75,6.75
big_font_size = 18
small_font_size = 15
fmt=".2f"
sbrn_palette = sns.color_palette("deep")# ["#A6C8FF", "#75D1D1", "#8EC6FF", "#FFBC9A", "#C8A6FF", "#F2AFA1", "#A6F0A6"]
sota_palette = [sbrn_palette[i] for i in range(len(sbrn_palette)) if sbrn_palette[i] != sns.xkcd_rgb['red']]


metrics = ["mean", "median", "percentile_90"]


def human_readable_size(num_bytes):
    for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB']:
        if num_bytes < 1024:
            return f"{int(num_bytes)} {unit}"
        num_bytes /= 1024
    return f"{int(num_bytes)} PiB"

def get_summaries(args):
    # Read metadata file
    metadata_file = f"results/" + args.system + "_metadata.csv"
    if not os.path.exists(metadata_file):
        print(f"Metadata file {metadata_file} not found. Exiting.", file=sys.stderr)
        sys.exit(1)
    metadata = pd.read_csv(metadata_file)
    nnodes = [n for n in str(args.nnodes).split(",")]
    summaries = {} # Contain the folder for each node count
    # Search all the entries we might need
    for nodes in nnodes:
        if "tasks_per_node" in metadata.columns:
            filtered_metadata = metadata[(metadata["collective_type"].str.lower() == args.collective.lower()) & \
                                        (metadata["nnodes"].astype(str) == str(nodes)) & \
                                        (metadata["tasks_per_node"].astype(int) == args.tasks_per_node)]
        else:
            filtered_metadata = metadata[(metadata["collective_type"].str.lower() == args.collective.lower()) & \
                                        (metadata["nnodes"].astype(str) == str(nodes))]            
        if args.notes:
            filtered_metadata = filtered_metadata[(filtered_metadata["notes"].str.strip() == args.notes.strip())]
        else:
            # Keep only those without notes
            filtered_metadata = filtered_metadata[filtered_metadata["notes"].isnull()]

        if args.system == "leonardo":
            # Discard OMPI_BINE
            filtered_metadata = filtered_metadata[~filtered_metadata["mpi_lib"].str.contains("OMPI_BINE", case=False)]
            
        if filtered_metadata.empty:
            print(f"Metadata file {metadata_file} does not contain the requested data. Exiting.", file=sys.stderr)
            sys.exit(1)
    
        # Among the remaining ones, keep only tha last one
        filtered_metadata = filtered_metadata.iloc[-1]
        #summaries[nodes] = "results/" + args.system + "/" + filtered_metadata["timestamp"] + "/" + str(filtered_metadata["test_id"]) + "/aggregated_result_summary.csv"
        summaries[nodes] = "results/" + args.system + "/" + filtered_metadata["timestamp"] + "/"
    return summaries

def get_summaries_df(args):
    summaries = get_summaries(args)
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

        s = pd.read_csv(summary + "/aggregated_results_summary.csv")        
        # Filter by collective type
        s = s[s["collective_type"].str.lower() == args.collective.lower()]      
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
        elif "nccl_collnet" in algo_name.lower():
            return "CollNet"
        elif "nccl_nvls" in algo_name.lower():
            return "NVLS"
        elif "pat" in algo_name.lower():
            return "Binomial"        
        elif "allreduce_hier" in algo_name.lower():
            return "Bine"
        elif "nccl_tree" in algo_name.lower():
            return "Binary"
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
    # Step 1: Create an empty list to hold the new rows
    new_data = []
    
    # For each (buffer_size, nodes) group the data so that for eacha algo_family we only keep the entry with the highest bandwidth_mean
    df = df.loc[df.groupby(['buffer_size', 'Nodes', 'algo_family'])['bandwidth_' + metric].idxmax()]

    # Step 2: Group by 'buffer_size' and 'Nodes'
    for (buffer_size, nodes), group in df.groupby(['buffer_size', 'Nodes']):        
        # Step 3: Get the best algorithm        
        best_algo_row = group.loc[group['bandwidth_' + metric].idxmax()]
        best_algo = best_algo_row['algo_family']
        
        # Step 4: Get the second best algorithm (excluding the best one)
        tmp = group[group['algo_family'] != best_algo]['bandwidth_' + metric]
        if tmp.empty:
            print(f"Warning: No second best algorithm found for buffer_size {buffer_size} and nodes {nodes}. Skipping.", file=sys.stderr)
            continue
        second_best_algo_row = group.loc[tmp.idxmax()]
        second_best_algo = second_best_algo_row['algo_family']

        # Get Bine bandwidth_mean for this group
        bine_row = group.loc[group['algo_family'] == "Bine"]
        if bine_row.empty:
            print(f"Warning: No Bine algorithm found for buffer_size {buffer_size} and nodes {nodes}. Skipping.", file=sys.stderr)
            continue
        
        bine_bandwidth_mean = bine_row['bandwidth_' + metric].values[0]

        #print(f"Buffer size: {buffer_size}, Nodes: {nodes}, Best algo: {best_algo}, Second best algo: {second_best_algo}")
        #print(group)

        ratio = bine_bandwidth_mean / best_algo_row['bandwidth_' + metric]
        # Truncate to 1 decimal place
        ratio = round(ratio, 1)
        
        if best_algo == "Bine":
            cell = best_algo_row['bandwidth_' + metric] / second_best_algo_row['bandwidth_' + metric]  
        elif ratio >= 1.0:
            cell = ratio         
        else:
            #print(f"Losign on {buffer_size},{nodes} vs {best_algo} by {bine_bandwidth_mean / best_algo_row['bandwidth_' + metric]} ({bine_bandwidth_mean} vs {best_algo_row['bandwidth_' + metric]})")
            cell = best_algo
        
        # Step 6: Append the data for this group (including old columns)
        new_data.append({
            'buffer_size': buffer_size,
            'Nodes': nodes,
            #'algo_family': best_algo,
            #'bandwidth_' + metric: best_algo_row['bandwidth_' + metric],
            'cell': cell,
        })

    # Step 7: Create a new DataFrame
    return pd.DataFrame(new_data)

def algo_to_family(df, args):
    # Convert algo_name to algo_family
    df["algo_family"] = df["algo_name"].apply(lambda x: algo_name_to_family(x, args.system))
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
    elif family_name == "NVLS":
        return ("V", sota_palette[6])
    elif family_name == "CollNet":
        return ("W", sota_palette[6])
    else:
        # error
        raise ValueError(f"Unknown algorithm family {family_name}")

def main():
    parser = argparse.ArgumentParser(description="Generate graphs")
    parser.add_argument("--system", type=str, help="System name")
    parser.add_argument("--nnodes", type=str, help="Number of nodes (comma separated)")
    parser.add_argument("--tasks_per_node", type=int, help="Tasks per node", default=1)
    parser.add_argument("--collective", type=str, help="Collective")    
    parser.add_argument("--notes", type=str, help="Notes")   
    parser.add_argument("--exclude", type=str, help="Algos to exclude", default=None)   
    parser.add_argument("--metric", type=str, help="Metric to consider [mean|median|percentile_90]", default="mean")   
    parser.add_argument("--base", type=str, help="Compare against [all|binomial]", default="all")
    parser.add_argument("--y_no", help="Does not show ticks and labels on y-axis", action="store_true")
    args = parser.parse_args()

    #print("Called with args:")
    #print(args)

    df = get_summaries_df(args)    
          
    # Drop the columns I do not need
    df = df[["buffer_size", "Nodes", "algo_name", "mean", "median", "percentile_90"]]

    # If system name is "fugaku", drop all the algo_name starting with uppercase "RECDOUB"
    if args.system == "fugaku":
        df = df[~df["algo_name"].str.startswith("RECDOUB")]

    if args.exclude:
        df = df[~df["algo_name"].str.contains(args.exclude, case=False)]
    
    #df = df[~df["algo_name"].str.contains("default_mpich", case=False)]
    
    # Compute the bandwidth for each metric
    for m in metrics:
        if m == args.metric:
            df["bandwidth_" + m] = ((df["buffer_size"]*8.0)/(1000.0*1000*1000)) / (df[m].astype(float) / (1000.0*1000*1000))
    
    # drop all the metrics
    for m in metrics:
        df = df.drop(columns=[m])

    # print full df, no limts on cols or rows
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)

    df = algo_to_family(df, args)
    df = augment_df(df,args.metric)
    #print(df)

    # We need to separate numerical and string cells
    # Step 1: Create the 'numeric' version of the DataFrame, where strings are NaN
    df_numeric = df.copy()
    df_numeric['cell'] = pd.to_numeric(df['cell'], errors='coerce')

    # Step 2: Pivot the numeric data for heatmap plotting
    heatmap_data_numeric = df_numeric.pivot(index='buffer_size', columns='Nodes', values='cell')
    heatmap_data_numeric = heatmap_data_numeric[args.nnodes.split(",")]    

    # Step 3: Pivot the original data for string annotation
    heatmap_data_string = df.pivot(index='buffer_size', columns='Nodes', values='cell')
    heatmap_data_string = heatmap_data_string[args.nnodes.split(",")]

    # Set up the figure and axes
    plt.figure()

    # Create a matrix of colors for the heatmap cells based on the content of the dataframe
    # Create an empty matrix of the same shape as df for background colors
    cell_colors = np.full(heatmap_data_string.shape, 'white', dtype=object)  # Default white for numbers

    # Create the heatmap with numerical values for color
    red_green = LinearSegmentedColormap.from_list("RedGreen", ["darkred", "white", "darkgreen"])
    ax = sns.heatmap(heatmap_data_numeric, 
                    annot=True, 
                    cmap=red_green, 
                    fmt=fmt,
                    center=1, 
                    cbar=True, 
                    #square=True,
                    annot_kws={'size': big_font_size, 'weight': 'bold'},
                    cbar_kws={"orientation": "horizontal", "location" : "top", "aspect": 40},
                    )

    # Get the colorbar and set the font size
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=small_font_size)  # Adjust font size of ticks

    ###############
    # SET STRINGS #
    ###############
    for i in range(heatmap_data_string.shape[0]):
        for j in range(heatmap_data_string.shape[1]):
            val = heatmap_data_string.iloc[i, j]
            # Check if the value is a string
            if isinstance(val, str):
                val, col = family_name_to_letter_color(val)
                plt.text(j + 0.5, i + 0.5, val, ha='center', va='center', color=col, weight='bold', fontsize=big_font_size)
            # Check if the value is NaN (not a number)
            elif pd.isna(val):
                plt.text(j + 0.5, i + 0.5, "N/A", ha='center', va='center', color='black', weight='bold', fontsize=big_font_size)
    
    ################
    # SET BG COLOR #
    ################
    # Loop over each cell and change the background color
    for i in range(heatmap_data_string.shape[0]):
        for j in range(heatmap_data_string.shape[1]):
            if isinstance(heatmap_data_string.iloc[i, j], str):
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color='#f0f0f0', lw=0, zorder=-1))    

    # For each ror use the corresponding buffer_size_hr rather than buffer_size as labels
    # Get all the row names, sort them (numerically), and the apply to each of them the human_readable_size function
    # to get the human-readable size
    # Then set the x-ticks labels to these human-readable sizes
    # Get heatmap_data.rows and convert to a list of int
    buffer_sizes = heatmap_data_string.index.astype(int).tolist()
    buffer_sizes.sort()
    buffer_sizes = [human_readable_size(int(x)) for x in buffer_sizes]
    # Use buffer_sizes as labels
    plt.yticks(ticks=np.arange(len(buffer_sizes)) + 0.5, labels=buffer_sizes)

    plt.xlabel("# Nodes", fontsize=big_font_size)
    plt.ylabel("Vector Size", fontsize=big_font_size)
    plt.xticks(fontsize=small_font_size)
    plt.yticks(fontsize=small_font_size)
    # Do not rotate xticklabels
    plt.xticks(rotation=0)   

    # Make dir if it does not exist
    outdir = "plot/" + args.system + "/heatmaps/" + args.collective + "/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # in outfile name we should save all the infos in args
    # Convert args to a string with param name and param value
    args_str = "_".join([f"{k}_{v}" for k, v in vars(args).items() \
                        if k != "nnodes" and k != "system" and k != "collective" and (k != "notes" or v != None) and (k != "exclude" or v != None)])
    args_str = args_str.replace("|", "_")
    outfile = outdir + "/" + args_str + ".pdf"
    # Save as PDF
    plt.savefig(outfile, bbox_inches="tight")

if __name__ == "__main__":
    main()

