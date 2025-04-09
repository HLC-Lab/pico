import os, sys, argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import subprocess
from matplotlib.colors import LinearSegmentedColormap

binomials = {}
binomials[("OMPI", "4.1.6", "1.0.0", "allreduce")]      = ["recursive_doubling_ompi", "recursive_doubling_over", "rabenseifner_ompi", "rabenseifner_over"]
binomials[("OMPI", "4.1.6", "1.0.0", "allgather")]      = ["recursive_doubling_ompi", "recursive_doubling_over", "k_bruck_over", "sparbit_over"]
binomials[("OMPI", "4.1.6", "1.0.0", "alltoall")]       = ["modified_bruck_ompi"]
binomials[("OMPI", "4.1.6", "1.0.0", "bcast")]          = ["binomial_ompi", "scatter_allgather_ompi", "scatter_allgather_over", "scatter_allgather_ring_ompi"]
binomials[("OMPI", "4.1.6", "1.0.0", "gather")]         = ["binomial_ompi"]
binomials[("OMPI", "4.1.6", "1.0.0", "reduce")]         = ["binomial_ompi", "rabenseifner_ompi"]
binomials[("OMPI", "4.1.6", "1.0.0", "reduce_scatter")] = ["recursive_distance_doubling_over", "recursive_halving_ompi", "recursive_halving_over"]
binomials[("OMPI", "4.1.6", "1.0.0", "scatter")]        = ["binomial_ompi"]



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
    nnodes = [int(n) for n in str(args.nnodes).split(",")]
    summaries = {} # Contain the folder for each node count
    # Search all the entries we might need
    for nodes in nnodes:
        filtered_metadata = metadata[(metadata["collective_type"].str.lower() == args.collective.lower()) & (metadata["nnodes"].astype(int) == nodes) & (metadata["tasks_per_node"].astype(int) == args.tasks_per_node)]
        if args.notes:
            filtered_metadata = filtered_metadata[(filtered_metadata["notes"].str == args.notes)]
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
        if not os.path.exists(summary + "/aggregated_results_summary.csv"):        
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
        df[bin_mask].groupby(group_keys)["mean"].idxmin()
    ].copy()
    best_binomial["algo_name"] = "best_other"
    return best_binomial

def get_best_swing(df):
    # Create masks
    swing_mask = df["algo_name"].str.lower().str.startswith("swing")
    # Grouping keys
    group_keys = ["buffer_size", "Nodes"]
    # Find best swing per group
    best_swing = df[swing_mask].loc[
        df[swing_mask].groupby(group_keys)["mean"].idxmin()
    ].copy()
    best_swing["algo_name"] = "best_swing"    
    return best_swing

def get_best_other(df):
    # Create masks
    swing_mask = df["algo_name"].str.lower().str.startswith("swing")
    non_swing_mask = ~swing_mask
    # Grouping keys
    group_keys = ["buffer_size", "Nodes"]
    # Find best non-swing per group
    best_other = df[non_swing_mask].loc[
        df[non_swing_mask].groupby(group_keys)["mean"].idxmin()
    ].copy()
    best_other["algo_name"] = "best_other"    
    return best_other

def main():
    parser = argparse.ArgumentParser(description="Generate graphs")
    parser.add_argument("--system", type=str, help="System name")
    parser.add_argument("--nnodes", type=str, help="Number of nodes (comma separated)")
    parser.add_argument("--tasks_per_node", type=int, help="Tasks per node", default=1)
    parser.add_argument("--collective", type=str, help="Collective")    
    parser.add_argument("--notes", type=str, help="Notes")   
    parser.add_argument("--exclude", type=str, help="Algos to exclude", default=None)   
    parser.add_argument("--vs", type=str, help="Compare against this algo_name [all|binomial]", required=True, default="all")
    args = parser.parse_args()

    df = get_summaries_df(args)
          
    # Drop the columns I do not need
    df = df[["buffer_size", "Nodes", "algo_name", "mean", "median"]]

    # If system name is "fugaku", drop all the algo_name starting with uppercase "RECDOUB"
    if args.system == "fugaku":
        df = df[~df["algo_name"].str.startswith("RECDOUB")]

    if args.exclude:
        # Drop those with "segmented" and "block" in the name
        df = df[~df["algo_name"].str.contains(args.exclude, case=False)]

    best_swing = get_best_swing(df)
    if args.vs == "all":
        best_other = get_best_other(df)
    else:
        best_other = get_best_binomial(df, args)
    # Combine everything
    augmented_df = pd.concat([df, best_swing, best_other], ignore_index=True)

    # Combine back
    for m in ["mean", "median"]:
        augmented_df["bandwidth_" + m] = ((augmented_df["buffer_size"]*8.0)/(1000.0*1000*1000)) / (augmented_df[m].astype(float) / (1000.0*1000*1000))

    # Generate plot
    # Keep only best_swing and best_other
    best_df = augmented_df[augmented_df["algo_name"].isin(["best_swing", "best_other"])]

    # Pivot to get one column per algo_name
    pivot = best_df.pivot_table(
        index=["buffer_size", "Nodes"],
        columns="algo_name",
        values="bandwidth_mean"
    ).reset_index()

    # Compute ratio
    pivot["bandwidth_ratio"] = pivot["best_swing"] / pivot["best_other"]

    # Pivot for heatmap with sizes on the x-axis
    heatmap_data = pivot.pivot(
        index="Nodes",
        columns="buffer_size",
        values="bandwidth_ratio"
    )
    
    # Pivot again for heatmap (Nodes as rows, buffer_size as columns)
    heatmap_data = pivot.pivot(
        index="Nodes",
        columns="buffer_size",
        values="bandwidth_ratio"
    )

    red_green = LinearSegmentedColormap.from_list("RedGreen", ["darkred", "white", "darkgreen"])
    #red_green = "RdYlGn"
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
        cmap=red_green,
        center=1.0,
        cbar_kws={"label": "Bandwidth Ratio (best_swing / best_other)"},
        mask=heatmap_data.isna()
    )
    # For each column use the corresponding buffer_size_hr rather than buffer_size as labels
    # Get all the column names, sort them (numerically), and the apply to each of them the human_readable_size function
    # to get the human-readable size
    # Then set the x-ticks labels to these human-readable sizes
    # Get heatmap_data.columns and convert to a list of int
    buffer_sizes = heatmap_data.columns.astype(int).tolist()
    buffer_sizes.sort()
    buffer_sizes = [human_readable_size(int(x)) for x in buffer_sizes]
    # Use buffer_sizes as labels
    plt.xticks(ticks=np.arange(len(buffer_sizes)) + 0.5, labels=buffer_sizes, rotation=45)

    plt.title("Bandwidth Ratio: best_swing vs best_other")
    plt.xlabel("Vector Size")
    plt.ylabel('')
    plt.tight_layout()

    # Make dir if it does not exist
    outdir = "plot/" + args.system + "/" + args.collective + "/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # in outfile name we should save all the infos in args
    # Convert args to a string with param name and param value (except nnodes)
    args_str = "_".join([f"{k}_{v}" for k, v in vars(args).items() \
                        if k != "nnodes" and k != "system" and k != "collective" and (k != "notes" or v != None) and (k != "exclude" or v != None)])
    outfile = outdir + "/" + args_str + ".pdf"
    # Save as PDF
    plt.savefig(outfile, bbox_inches="tight")

if __name__ == "__main__":
    main()

