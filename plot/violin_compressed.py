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
rcParams['figure.figsize'] = 12/2.0,6.75/2.0
#rcParams['figure.figsize'] = 6.75,6.75
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


nodes_dict = {}

nodes_dict["leonardo"] = ["128", "256", "512", "1024", "2048"]
nodes_dict["lumi"] = ["16", "32", "64", "128", "256", "512", "1024"]
nodes_dict["mare_nostrum"] = ["4", "8", "16"]
nodes_dict["fugaku"] = ["2x2x2", "8x8x8", "64x64", "32x256"]

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
    
        # Drop the number of nodes not in nodes_dict[system]
        if system in nodes_dict:
            filtered_metadata = filtered_metadata[filtered_metadata["nnodes"].astype(str).isin(nodes_dict[system])]
        else:
            print(f"System {system} not found in nodes_dict. Exiting.", file=sys.stderr)
            sys.exit(1)

        # For each nnodes, keep only the last entry
        filtered_metadata = filtered_metadata.sort_values("timestamp").groupby("nnodes").tail(1)

        #filtered_metadata = filtered_metadata.iloc[-1]
        # Check if anything remained
        if filtered_metadata.empty:
            print(f"Metadata file {metadata_file} does not contain the requested data. Exiting.", file=sys.stderr)
            sys.exit(1)
        i += 1
        # for each node count put in summaries
        for index, row in filtered_metadata.iterrows():
            summaries[(system, row["nnodes"])] = "results/" + system + "/" + row["timestamp"] + "/"
    return summaries

def get_summaries_df(args):
    summaries = get_summaries(args)
    df = pd.DataFrame()
    # Loop over the summaries
    for (system, nnodes), summary in summaries.items():
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
        s["Nodes"] = nnodes
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
    group_keys = ["buffer_size", "System", "Nodes"]
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
    group_keys = ["buffer_size", "System", "Nodes"]
    # Find best non-swing per group
    best_other = df[non_swing_mask].loc[
        df[non_swing_mask].groupby(group_keys)[args.metric].idxmin()
    ].copy()
    best_other["algo_name"] = "best_other"    
    return best_other

system_name_hr = {
    "leonardo": "Leonardo",
    "mare_nostrum": "Mare Nostrum 5",
    "lumi": "LUMI",
    "fugaku": "Fugaku",
}

def main():
    parser = argparse.ArgumentParser(description="Generate graphs")
    parser.add_argument("--systems", type=str, help="Buffer size (comma separated)")
    parser.add_argument("--tasks_per_node", type=int, help="Tasks per node", default=1)
    parser.add_argument("--collective", type=str, help="Collective")    
    parser.add_argument("--notes", type=str, help="Notes")   
    parser.add_argument("--exclude", type=str, help="Algos to exclude", default=None)   
    parser.add_argument("--metric", type=str, help="Metric to consider [mean|median|percentile_90]", default="mean")   
    parser.add_argument("--plot_type", type=str, help="Plot type [violin|box]", default="violin")   
    parser.add_argument("--y_no", help="Does not show ticks and labels on y-axis", action="store_true")
    args = parser.parse_args()

    df = get_summaries_df(args)
          
    # Drop the columns I do not need
    df = df[["buffer_size", "System", "Nodes", "algo_name", "mean", "median", "percentile_90"]]


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

    augmented_df = augmented_df[augmented_df["algo_name"].isin(["best_swing", "best_other", "best_binomial"])]

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    #print(augmented_df)    

   # Pivot to get variants side by side
    pivot = augmented_df.pivot_table(
        index=['System', 'Nodes', 'buffer_size'],
        columns='algo_name',
        values='bandwidth_mean'
    ).reset_index()

    # Compute the ratios
    pivot['Bine Performance Gain vs. Best SotA'] = pivot['best_swing'] / pivot['best_other']
    pivot['Bine Performance Gain vs. Best Binomial/Butterfly'] = pivot['best_swing'] / pivot['best_binomial']


    # Melt into long format
    melted = pivot.melt(
        id_vars=['System', 'Nodes', 'buffer_size'],
        value_vars=['Bine Performance Gain vs. Best SotA', 'Bine Performance Gain vs. Best Binomial/Butterfly'],
        var_name='Comparison',
        value_name='Ratio'
    )

    # Drop all the vs_binomial where system is fugaku
    melted = melted[~((melted["System"] == "fugaku") & (melted["Comparison"] == "Bine Performance Gain vs. Best Binomial/Butterfly"))]

    plt.figure()
    if args.plot_type == "violin":
        sns.violinplot(
            data=melted,
            x='System',
            y='Ratio',
            hue='Comparison',
            split=True,
            scale='width',
            inner='quartile'
        )
    elif args.plot_type == "box":
        ax= sns.boxplot(data=melted, x='System', y='Ratio', hue='Comparison', showmeans=True)
        plt.axhline(1.0, linestyle='--', color='gray')  # Optional baseline    
        # y log scale
        #plt.yscale('log')

        # Get current labels
        current_labels = [tick.get_text() for tick in ax.get_xticklabels()]
        # Map to new labels
        new_labels = [system_name_hr.get(label, label) for label in current_labels]
        # Set new labels
        tick_positions = ax.get_xticks()
        ax.set_xticks(tick_positions)  # explicitly set ticks        
        ax.set_xticklabels(new_labels)
        # Remove legend title
        ax.legend(title=None)
        ax.set_xlabel("")
    else:
        # Swarm plot
        ax = sns.swarmplot(data=melted, x='System', y='Ratio', hue='Comparison', dodge=True, alpha=0.5)
        plt.axhline(1.0, linestyle='--', color='gray')
        # y log scale
        #plt.yscale('log')
        # Get current labels
        current_labels = [tick.get_text() for tick in ax.get_xticklabels()]
        # Map to new labels
        new_labels = [system_name_hr.get(label, label) for label in current_labels]
        # Set new labels
        tick_positions = ax.get_xticks()
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(new_labels)
        # Remove legend title
        ax.legend(title=None)
        ax.set_xlabel("")

    plt.ylabel('Bine Tree Improvement Ratio')
    #plt.title('Performance Ratio by System and Node Count')
    plt.tight_layout()

    # Make dir if it does not exist
    outdir = "plot/violin_compr/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    outfile = outdir + "/" + args.collective + "_" + args.plot_type +  ".pdf"
    # Save as PDF
    plt.savefig(outfile, bbox_inches="tight")

if __name__ == "__main__":
    main()

