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
fmt=".1f"
sbrn_palette = sns.color_palette("deep")# ["#A6C8FF", "#75D1D1", "#8EC6FF", "#FFBC9A", "#C8A6FF", "#F2AFA1", "#A6F0A6"]
sota_palette = [sbrn_palette[i] for i in range(len(sbrn_palette)) if sbrn_palette[i] != sns.xkcd_rgb['red']]


metrics = ["mean", "median", "percentile_90"]

binomials = {}
##################################
# Leonardo's binomial algorithms #
##################################
binomials[("OMPI", "4.1.6", "1.0.0", "allreduce")]      = ["recursive_doubling_ompi", "recursive_doubling_over", "rabenseifner_ompi", "rabenseifner_over"]
binomials[("OMPI", "4.1.6", "1.0.0", "allgather")]      = ["recursive_doubling_ompi", "recursive_doubling_over", "k_bruck_over", "sparbit_over"]
binomials[("OMPI", "4.1.6", "1.0.0", "alltoall")]       = ["modified_bruck_ompi"]
binomials[("OMPI", "4.1.6", "1.0.0", "bcast")]          = ["binomial_ompi", "scatter_allgather_ompi", "scatter_allgather_over", "scatter_allgather_ring_ompi"]
binomials[("OMPI", "4.1.6", "1.0.0", "gather")]         = ["binomial_ompi"]
binomials[("OMPI", "4.1.6", "1.0.0", "reduce")]         = ["binomial_ompi", "rabenseifner_ompi"]
binomials[("OMPI", "4.1.6", "1.0.0", "reduce_scatter")] = ["recursive_distance_doubling_over", "recursive_halving_ompi", "recursive_halving_over"]
binomials[("OMPI", "4.1.6", "1.0.0", "scatter")]        = ["binomial_ompi"]

########################################
# Mare Nostrum 5's binomial algorithms #
########################################
binomials[("OMPI", "4.1.5", "1.0.0", "allreduce")]      = ["recursive_doubling_ompi", "recursive_doubling_over", "rabenseifner_ompi", "rabenseifner_over"]
binomials[("OMPI", "4.1.5", "1.0.0", "allgather")]      = ["recursive_doubling_ompi", "recursive_doubling_over", "k_bruck_over", "sparbit_over"]
binomials[("OMPI", "4.1.5", "1.0.0", "alltoall")]       = ["modified_bruck_ompi"]
binomials[("OMPI", "4.1.5", "1.0.0", "bcast")]          = ["binomial_ompi", "scatter_allgather_ompi", "scatter_allgather_over", "scatter_allgather_ring_ompi"]
binomials[("OMPI", "4.1.5", "1.0.0", "gather")]         = ["binomial_ompi"]
binomials[("OMPI", "4.1.5", "1.0.0", "reduce")]         = ["binomial_ompi", "rabenseifner_ompi"]
binomials[("OMPI", "4.1.5", "1.0.0", "reduce_scatter")] = ["recursive_distance_doubling_over", "recursive_halving_ompi", "recursive_halving_over"]
binomials[("OMPI", "4.1.5", "1.0.0", "scatter")]        = ["binomial_ompi"]

##############################
# LUMI's binomial algorithms #
##############################
binomials[("CRAY_MPICH", "8.1.29", "1.0.0", "allreduce")]      = ["rabenseifner_mpich", "rabenseifner_over", "recursive_doubling_mpich", "recursive_doubling_over"]
binomials[("CRAY_MPICH", "8.1.29", "1.0.0", "allgather")]      = ["brucks_mpich", "recursive_doubling_mpich", "recursive_doubling_over", "sparbit_over"]
binomials[("CRAY_MPICH", "8.1.29", "1.0.0", "alltoall")]       = ["brucks_mpich"]
binomials[("CRAY_MPICH", "8.1.29", "1.0.0", "bcast")]          = ["binomial_mpich", "scatter_allgather_over", "scatter_recursive_doubling_allgather_mpich"]
binomials[("CRAY_MPICH", "8.1.29", "1.0.0", "gather")]         = ["binomial_mpich"]
binomials[("CRAY_MPICH", "8.1.29", "1.0.0", "reduce")]         = ["binomial_mpich", "rabenseifner_mpich"]
binomials[("CRAY_MPICH", "8.1.29", "1.0.0", "reduce_scatter")] = ["recursive_distance_doubling_mpich", "recursive_halving_mpich", "recursive_halving_over"]
binomials[("CRAY_MPICH", "8.1.29", "1.0.0", "scatter")]        = ["binomial_mpich"]

################################
# Fugaku's binomial algorithms #
################################
binomials[("FJMPI", "x.x.x", "4.0.1", "allreduce")]      = ["default-recursive_doubling"]
binomials[("FJMPI", "x.x.x", "4.0.1", "reduce_scatter")] = ["default-recursive-halving"]
binomials[("FJMPI", "x.x.x", "4.0.1", "allgather")]      = ["default-bruck", "default-recursive-doubling"]
binomials[("FJMPI", "x.x.x", "4.0.1", "bcast")]          = ["default-binomial"]
binomials[("FJMPI", "x.x.x", "4.0.1", "alltoall")]       = ["default-modified-bruck"]
binomials[("FJMPI", "x.x.x", "4.0.1", "scatter")]        = ["default-binomial"]
binomials[("FJMPI", "x.x.x", "4.0.1", "gather")]         = ["default-binomial"]
binomials[("FJMPI", "x.x.x", "4.0.1", "reduce")]         = ["default-binomial"]

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
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)

        s = pd.read_csv(summary + "/aggregated_results_summary.csv")        
        # Filter by collective type
        s = s[s["collective_type"].str.lower() == args.collective.lower()]      
        # Drop the rows where buffer_size is equal to 4 (we do not have them for all results :( )  
        s = s[s["buffer_size"] != 4]
        s["Nodes"] = nodes
        # Append s to df
        df = pd.concat([df, s], ignore_index=True)
    return df

def get_best_binomial(df, args):
    # Get metadata
    meta = pd.read_csv("results/" + args.system + "_metadata.csv")
    # We assume that mpi_lib, mpi_lib_version, and libswing_version
    # are the same for all entries, so just read them for the last entry
    mpi_lib = meta["mpi_lib"].iloc[-1]
    mpi_lib_version = meta["mpi_lib_version"].iloc[-1]
    libswing_version = meta["libswing_version"].iloc[-1]

    bin = binomials.get((mpi_lib, mpi_lib_version, libswing_version, args.collective.lower()))
    if bin is None:
        print(f"No binomial algorithms found for {mpi_lib} {mpi_lib_version} {libswing_version} {args.collective.lower()}. Exiting.", file=sys.stderr)
        sys.exit(1)
    # Find the best algorithm for each buffer_size and "Nodes" among those in bin
    # Create masks
    bin_mask = df["algo_name"].str.lower().isin(bin)
    # Grouping keys
    group_keys = ["buffer_size", "Nodes"]
    # Find best binomial per group
    best_binomial = df[bin_mask].loc[
        df[bin_mask].groupby(group_keys)[args.metric].idxmin()
    ].copy()
    best_binomial["algo_name"] = "best_binomial"
    return best_binomial

def get_best_swing(df, args):
    # Create masks
    swing_mask = df["algo_name"].str.lower().str.startswith("swing")
    # Grouping keys
    group_keys = ["buffer_size", "Nodes"]
    # Find best swing per group
    best_swing = df[swing_mask].loc[
        df[swing_mask].groupby(group_keys)[args.metric].idxmin()
    ].copy()
    best_swing["algo_name"] = "best_swing"    
    return best_swing

def get_best_other(df, args):
    # Create masks
    swing_mask = df["algo_name"].str.lower().str.startswith("swing")
    non_swing_mask = ~swing_mask
    # Grouping keys
    group_keys = ["buffer_size", "Nodes"]
    # Find best non-swing per group
    best_other = df[non_swing_mask].loc[
        df[non_swing_mask].groupby(group_keys)[args.metric].idxmin()
    ].copy()
    best_other["algo_name"] = "best_other"    
    return best_other

def get_heatmap_data(df, base, args):
    # Generate plot
    # Keep only best_swing and best_other
    best_df = df[df["algo_name"].isin(["best_swing", base])]

    # Pivot to get one column per algo_name
    pivot = best_df.pivot_table(
        index=["buffer_size", "Nodes"],
        columns="algo_name",
        values="bandwidth_mean"
    ).reset_index()

    # Compute ratio
    pivot["bandwidth_ratio"] = pivot["best_swing"] / pivot[base]

    # Pivot for heatmap with sizes on the x-axis
    heatmap_data = pivot.pivot(
        columns="Nodes",
        index="buffer_size",
        values="bandwidth_ratio"
    )
    
    # Pivot again for heatmap (Nodes as rows, buffer_size as columns)
    heatmap_data = pivot.pivot(
        columns="Nodes",
        index="buffer_size",
        values="bandwidth_ratio"
    )

    # Reorder rows
    #heatmap_data = heatmap_data.loc[args.nnodes.split(",")]        
    heatmap_data = heatmap_data[args.nnodes.split(",")]

    return heatmap_data


def algo_name_to_family(algo_name, system):
    if system == "fugaku":
        if algo_name.lower().startswith("swing"):
            return "Bine"
        elif algo_name.lower().startswith("default"):
            if "recursive-doubling" in algo_name.lower():
                return "Binomial"
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
        elif algo_name.lower().startswith("ring"):
            return "Ring"
    # error
    raise ValueError(f"Unknown algorithm {algo_name} for system {system}")
    

def augment_df(df):
    # Step 1: Create an empty list to hold the new rows
    new_data = []

    # For each (buffer_size, nodes) group the data so that for eacha algo_family we only keep the entry with the highest bandwidth_mean
    df = df.loc[df.groupby(['buffer_size', 'Nodes', 'algo_family'])['bandwidth_mean'].idxmax()]

    # Step 2: Group by 'buffer_size' and 'Nodes'
    for (buffer_size, nodes), group in df.groupby(['buffer_size', 'Nodes']):        
        # Step 3: Get the best algorithm
        best_algo_row = group.loc[group['bandwidth_mean'].idxmax()]
        best_algo = best_algo_row['algo_family']
        
        # Step 4: Get the second best algorithm (excluding the best one)
        second_best_algo_row = group.loc[group[group['algo_family'] != best_algo]['bandwidth_mean'].idxmax()]
        second_best_algo = second_best_algo_row['algo_family']

        #print(f"Buffer size: {buffer_size}, Nodes: {nodes}, Best algo: {best_algo}, Second best algo: {second_best_algo}")
        #print(group)
        
        if best_algo == "Bine":
            cell = best_algo_row['bandwidth_mean'] / second_best_algo_row['bandwidth_mean']            
        else:
            cell = best_algo
        
        # Step 6: Append the data for this group (including old columns)
        new_data.append({
            'buffer_size': buffer_size,
            'Nodes': nodes,
            #'algo_family': best_algo,
            #'bandwidth_mean': best_algo_row['bandwidth_mean'],
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

    df = get_summaries_df(args)
          
    # Drop the columns I do not need
    df = df[["buffer_size", "Nodes", "algo_name", "mean", "median", "percentile_90"]]

    # If system name is "fugaku", drop all the algo_name starting with uppercase "RECDOUB"
    if args.system == "fugaku":
        df = df[~df["algo_name"].str.startswith("RECDOUB")]

    if args.exclude:
        # Drop those with "segmented" and "block" in the name
        df = df[~df["algo_name"].str.contains(args.exclude, case=False)]
    
    df = df[~df["algo_name"].str.contains("default_mpich", case=False)]
    
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

    df = algo_to_family(df, args)
    df = augment_df(df)
    print(df)


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
                    center=1, 
                    cbar=True, 
                    #square=True,
                    annot_kws={'size': big_font_size},
                    cbar_kws={"orientation": "horizontal", "location" : "top", "aspect": 40},
                    )

    ###############
    # SET STRINGS #
    ###############
    for i in range(heatmap_data_string.shape[0]):
        for j in range(heatmap_data_string.shape[1]):
            val = heatmap_data_string.iloc[i, j]
            # Check if the value is a string
            if isinstance(val, str):
                val, col = family_name_to_letter_color(val)
                plt.text(j + 0.5, i + 0.5, val, ha='center', va='center', color=col, fontsize=big_font_size)
            # Check if the value is NaN (not a number)
            elif pd.isna(val):
                plt.text(j + 0.5, i + 0.5, "N/A", ha='center', va='center', color='black', fontsize=big_font_size)
    
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
    # Do not rotate xticklabels
    plt.xticks(rotation=0)   

    # Make dir if it does not exist
    outdir = "plot/" + args.system + "_hm/" + args.collective + "/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # in outfile name we should save all the infos in args
    # Convert args to a string with param name and param value
    args_str = "_".join([f"{k}_{v}" for k, v in vars(args).items() \
                        if k != "nnodes" and k != "system" and k != "collective" and (k != "notes" or v != None) and (k != "exclude" or v != None)])
    outfile = outdir + "/" + args_str + ".pdf"
    # Save as PDF
    plt.savefig(outfile, bbox_inches="tight")

if __name__ == "__main__":
    main()

