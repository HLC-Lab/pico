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
    systems = [s for s in str(args.systems).split(",")]
    summaries = {} # Contain the folder for each node count
    i = 0
    # Search all the entries we might need
    for system in systems:
        # Read metadata file
        metadata_file = f"results/" + system + "_metadata.csv"
        if not os.path.exists(metadata_file):
            print(f"Metadata file {metadata_file} not found. Exiting.", file=sys.stderr)
            sys.exit(1)
        metadata = pd.read_csv(metadata_file)
       
        if "tasks_per_node" in metadata.columns:
            filtered_metadata = metadata[(metadata["collective_type"].str.lower() == args.collective.lower()) & \
                                        (metadata["tasks_per_node"].astype(int) == args.tasks_per_node)]
        else:
            filtered_metadata = metadata[(metadata["collective_type"].str.lower() == args.collective.lower())]            
        
        if args.notes and args.notes.strip().split(",")[i] != "null":
            filtered_metadata = filtered_metadata[(filtered_metadata["notes"].str.strip() == args.notes.strip().split(",")[i])]
        else:
            # Keep only those without notes
            filtered_metadata = filtered_metadata[filtered_metadata["notes"].isnull()]
            
        if filtered_metadata.empty:
            print(f"Metadata file {metadata_file} does not contain the requested data. Exiting.", file=sys.stderr)
            sys.exit(1)
    
        # Among the remaining ones, keep only tha last one
        # Get unique nnodes

        n = args.nnodes.split(",")[i]
        if "x" in str(n):
            # If there are multiple nodes, keep only the first one
            n = int(np.prod(n.split("x")))
        else:
            n = int(n)        
        print(f"Using {n} nodes for {system}")
        # Keep only the entry with nnodes equal to n
        filtered_metadata = filtered_metadata[filtered_metadata["nnodes"].astype(int) == n]
        # Keep only the last entry
        filtered_metadata = filtered_metadata.iloc[-1]
        # Check if anything remained
        if filtered_metadata.empty:
            print(f"Metadata file {metadata_file} does not contain the requested data. Exiting.", file=sys.stderr)
            sys.exit(1)
        i += 1
        summaries[system] = "results/" + system + "/" + filtered_metadata["timestamp"] + "/"
    return summaries

def get_summaries_df(args):
    summaries = get_summaries(args)
    df = pd.DataFrame()
    # Loop over the summaries
    for system, summary in summaries.items():
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
        s["System"] = system
        # Append s to df
        df = pd.concat([df, s], ignore_index=True)
    return df

def get_best_binomial(df, args):
    best_binomial = pd.DataFrame()
    for system in args.systems.split(","):
        # Get metadata
        meta = pd.read_csv("results/" + system + "_metadata.csv")
        # We assume that mpi_lib, mpi_lib_version, and libswing_version
        # are the same for all entries, so just read them for the last entry
        mpi_lib = meta["mpi_lib"].iloc[-1]
        mpi_lib_version = meta["mpi_lib_version"].iloc[-1]
        libswing_version = meta["libswing_version"].iloc[-1]

        bin = binomials.get((mpi_lib, mpi_lib_version, libswing_version, args.collective.lower()))
        if bin is None:
            print(f"No binomial algorithms found for {mpi_lib} {mpi_lib_version} {libswing_version} {args.collective.lower()}. Exiting.", file=sys.stderr)
            sys.exit(1)
        # Find the best algorithm for each buffer_size and "System" among those in bin
        # Create masks
        bin_mask = df["algo_name"].str.lower().isin(bin)
        # Grouping keys
        group_keys = ["buffer_size", "System"]
        # Find best binomial per group
        tmp_best_binomial = df[bin_mask].loc[
            df[bin_mask].groupby(group_keys)[args.metric].idxmin()
        ].copy()
        tmp_best_binomial["algo_name"] = "best_binomial"
        # Add to the best_binomial dataframe
        best_binomial = pd.concat([best_binomial, tmp_best_binomial], ignore_index=True)
    return best_binomial

def get_best_swing(df, args):
    # Create masks
    swing_mask = df["algo_name"].str.lower().str.startswith("swing")
    # Grouping keys
    group_keys = ["buffer_size", "System"]
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
    group_keys = ["buffer_size", "System"]
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
        index=["buffer_size", "System"],
        columns="algo_name",
        values="bandwidth_mean"
    ).reset_index()

    # Compute ratio
    pivot["bandwidth_ratio"] = pivot["best_swing"] / pivot[base]

    # Pivot for heatmap with sizes on the x-axis
    heatmap_data = pivot.pivot(
        columns="System",
        index="buffer_size",
        values="bandwidth_ratio"
    )
    
    # Pivot again for heatmap (Nodes as rows, buffer_size as columns)
    heatmap_data = pivot.pivot(
        columns="System",
        index="buffer_size",
        values="bandwidth_ratio"
    )

    # Reorder rows
    #heatmap_data = heatmap_data.loc[args.nnodes.split(",")]        
    heatmap_data = heatmap_data[args.systems.split(",")]

    return heatmap_data

system_name_hr = {
    "leonardo": "Leonardo",
    "mare_nostrum": "Mare Nostrum 5",
    "lumi": "LUMI",
    "fugaku": "Fugaku",
}

def main():
    parser = argparse.ArgumentParser(description="Generate graphs")
    parser.add_argument("--systems", type=str, help="Buffer size (comma separated)")
    parser.add_argument("--nnodes", type=str, help="Nodes on each system (comma separated)")
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
    df = df[["buffer_size", "System", "algo_name", "mean", "median", "percentile_90"]]


    df = df[~df["algo_name"].str.startswith("RECDOUB")]
    if args.exclude:
        # Drop those with "segmented" and "block" in the name
        df = df[~df["algo_name"].str.contains(args.exclude, case=False)]
    
    df = df[~df["algo_name"].str.contains("default_mpich", case=False)]

    best_swing = get_best_swing(df, args)
    best_other = get_best_other(df, args)
    best_binomial = get_best_binomial(df, args)
    # Combine everything
    augmented_df = pd.concat([df, best_swing, best_other, best_binomial], ignore_index=True)

    # Compute the bandwidth for each metric
    for m in metrics:
        augmented_df["bandwidth_" + m] = ((augmented_df["buffer_size"]*8.0)/(1000.0*1000*1000)) / (augmented_df[m].astype(float) / (1000.0*1000*1000))

    if args.base == "all":
        heatmap_data_primary = get_heatmap_data(augmented_df, "best_other", args)
        heatmap_data_secondary = get_heatmap_data(augmented_df, "best_binomial", args)
    else:
        heatmap_data_primary = get_heatmap_data(augmented_df, "best_binomial", args)
        heatmap_data_secondary = get_heatmap_data(augmented_df, "best_other", args)

    red_green = LinearSegmentedColormap.from_list("RedGreen", ["darkred", "white", "darkgreen"])
    #red_green = "RdYlGn"
    fig = plt.figure()

    # Create the structure
    ax = sns.heatmap(
        heatmap_data_primary,
        annot=False,
        cmap=red_green,
        cbar_kws={"orientation": "horizontal", "location" : "top", "aspect": 40},
        center=1.0,
        mask=heatmap_data_primary.isna(),
        linewidths=0.1, 
        linecolor="white",        
        cbar=True
    )

    # Get the colorbar and set the font size
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=small_font_size)  # Adjust font size of ticks


    # Plot the secondary first, so that the colors of the primary will overwrite them
    # Create the heatmap for the secondary values
    ax_secondary = sns.heatmap(
        heatmap_data_secondary,
        annot=True,
        annot_kws={'va':'top', 'size': str(small_font_size)}, # Attention, on heatmaps the coordinate system is inverted and thus 'top' actually goes to the bottom of the cell
        fmt=fmt,
        cmap=None, # I do not draw the colors of the secondary at all, since they are overwritten by the primary
        center=1.0,
        #cbar_kws={"label": "Speedup"},
        mask=heatmap_data_secondary.isna(),
        linewidths=0.1, 
        linecolor="white",
        cbar=False
    )

    ## Manually adjust vertical spacing (moving the annotations)
    #vertical_spacing = 0.05 
    #for text in ax_secondary.texts:
    #    # Get current y position and modify it
    #    x, y = text.get_position()  # Get the (x, y) position of the annotation
    #    text.set_position((x, y + vertical_spacing))  # Add vertical spacing    
    
    # Create the heatmap for the primary values
    ax_primary = sns.heatmap(
        heatmap_data_primary,
        annot=True,
        annot_kws={'va':'bottom', 'weight':'bold', 'size': str(big_font_size)}, # Attention, on heatmaps the coordinate system is inverted and thus 'bottom' actually goes to the top of the cell
        fmt=fmt,
        cmap=red_green,
        center=1.0,
        #cbar_kws={"label": "Speedup"},
        mask=heatmap_data_primary.isna(),
        linewidths=0, 
        linecolor="white",
        cbar=False
    )

    # Get the text annotations (this includes both secondary and primary annotations)
    annotations = ax_primary.texts

    # Number of annotations in the secondary heatmap
    num_secondary = int(len(annotations) / 2)

    # Separate the annotations into primary and secondary
    secondary_annotations = annotations[:num_secondary]
    primary_annotations = annotations[num_secondary:]

    # Manually adjust vertical spacing (moving the annotations)
    #vertical_spacing = 0.04 
    #i = 0
    #for text in primary_annotations:
    #    # Get current y position and modify it
    #    x, y = text.get_position()  # Get the (x, y) position of the annotation
    #    text.set_position((x, y - vertical_spacing))  # Add vertical spacing   

    # Manually adjust vertical spacing (moving the annotations)
    vertical_spacing = 0.08
    i = 0
    for text in secondary_annotations:
        # Get current y position and modify it
        x, y = text.get_position()  # Get the (x, y) position of the annotation
        text.set_position((x, y + vertical_spacing))  # Add vertical spacing   

    # Now get the color of the primary text annotations, and use them for the secondary annotations
    # Just take the colors of the annotation texts of the primary heatmap
    text_colors = [text.get_color() for text in primary_annotations]
    
    # Set the color of the secondary annotations to the same as the primary ones
    for i, text in enumerate(secondary_annotations):
        text.set_color(text_colors[i])


    # Now, add custom annotations where the placeholder value exists
    for i in range(heatmap_data_primary.shape[0]):
        for j in range(heatmap_data_primary.shape[1]):
            if np.isnan(heatmap_data_primary.iloc[i, j]): 
                plt.text(j + 0.5, i + 0.5, "N/A", ha='center', va='center', color='black', fontsize=big_font_size, weight='bold')        

    # For each ror use the corresponding buffer_size_hr rather than buffer_size as labels
    # Get all the row names, sort them (numerically), and the apply to each of them the human_readable_size function
    # to get the human-readable size
    # Then set the x-ticks labels to these human-readable sizes
    # Get heatmap_data.rows and convert to a list of int
    buffer_sizes = heatmap_data_primary.index.astype(int).tolist()
    buffer_sizes.sort()
    buffer_sizes = [human_readable_size(int(x)) for x in buffer_sizes]
    # Use buffer_sizes as labels
    plt.yticks(ticks=np.arange(len(buffer_sizes)) + 0.5, labels=buffer_sizes)

    #if args.base == "all":
    #    plt.title("Bine Trees/Butterflies Speedup over Best Overall Algorithm (top) and over Best Binomial Tree/Butterfly Algorithm (bottom)")
    #else:
    #    plt.title("Bine Trees/Butterflies Speedup over Best Binomial Tree/Butterfly (top) and over Best Overall Algorithm (bottom)")
    plt.ylabel("Vector Size", fontsize=big_font_size)
    plt.xlabel("")
    plt.yticks(fontsize=small_font_size)
    # For the x-ticks, use the system names in systems_name_hr
    # Get the system names and convert to a list
    systems = heatmap_data_primary.columns.tolist()
    systems_hr = []
    for s in systems:
        # Find s position in args.systems
        pos = args.systems.split(",").index(s)
        # Get the corresponding nnodes
        n = args.nnodes.split(",")[pos]
        if "x" in str(n):
            n = int(np.prod(n.split("x")))
        else:
            n = int(n)
        systems_hr.append(system_name_hr[s] + "\n(" + str(n) + " nodes)")

        
    # Use systems_hr as labels
    plt.xticks(ticks=np.arange(len(systems)) + 0.5, labels=systems_hr, fontsize=small_font_size)

    if args.y_no:
        plt.yticks([])
        plt.ylabel("")        

    plt.tight_layout()

    # Increase font size for xlabel and cbar



    # Make dir if it does not exist
    outdir = "plot/hm_compr/" + args.collective + "/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # in outfile name we should save all the infos in args
    # Convert args to a string with param name and param value
    args_str = "_".join([f"{v}" for k, v in vars(args).items() \
                        if k  == "nnodes"])
    outfile = outdir + "/" + args_str + ".pdf"
    # Save as PDF
    plt.savefig(outfile, bbox_inches="tight")

if __name__ == "__main__":
    main()

