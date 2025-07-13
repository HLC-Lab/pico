# Copyright (c) 2025 Saverio Pasqualoni
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
from scipy.stats import gmean


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
            
        if filtered_metadata.empty:
            print(f"Metadata file {metadata_file} does not contain the requested data. Exiting.", file=sys.stderr)
            sys.exit(1)
    
        # Among the remaining ones, keep only tha last one
        filtered_metadata = filtered_metadata.iloc[-1]
        #summaries[nodes] = "results/" + args.system + "/" + filtered_metadata["timestamp"] + "/" + str(filtered_metadata["test_id"]) + "/aggregated_result_summary.csv"
        summaries[nodes] = "results/" + args.system + "/" + filtered_metadata["timestamp"] + "/"
    return summaries

def get_tracer_out(args, compute_max=False):
    summaries = get_summaries(args)
    if compute_max:        
        ratio_small_v = float("+inf")
        ratio_big_v = float("+inf")
    else:
        ratio_small_v = 0
        ratio_big_v = 0   
    samples_small = 0
    samples_big = 0
    # Loop over the summaries
    for nodes, summary in summaries.items():
        if not os.path.exists(summary + "/traced_alloc.csv"):        
            command = [
                "python3", "tracer/trace_communications.py",
                "--location", args.system,
                "--alloc", summary + "/alloc.csv",
                "--comm", "tracer/algo_patterns.json",
                "--save"
            ]

            subprocess.run(command, stdout=subprocess.DEVNULL)

        # Add
        t = pd.read_csv(summary + "/traced_alloc.csv")
        # Filter by collective type
        coll_to_search = args.collective.lower()
        if coll_to_search == "gather":
            coll_to_search = "scatter" # Same num bytes
        t = t[t["COLLECTIVE"].str.lower() == coll_to_search]

        small = 0
        big = 0
        if coll_to_search == "allreduce":
            bine = float(t[t["ALGORITHM"] == "bine_latency"]["EXTERNAL"].iloc[0])
            sota = float(t[t["ALGORITHM"] == "recursive_doubling"]["EXTERNAL"].iloc[0])
            if bine != 0 and sota != 0:
                small = bine / sota
                samples_small += 1
            bine = float(t[t["ALGORITHM"] == "bine_bandwidth"]["EXTERNAL"].iloc[0])
            sota = float(t[t["ALGORITHM"] == "rabenseifner"]["EXTERNAL"].iloc[0]) 
            if bine != 0 and sota != 0:
                big = bine / sota
                samples_big += 1
        elif coll_to_search == "allgather":
            bine = float(t[t["ALGORITHM"] == "bine_halving"]["EXTERNAL"].iloc[0])
            sota = float(t[t["ALGORITHM"] == "distance_halving"]["EXTERNAL"].iloc[0])            
            if bine != 0 and sota != 0:    
                big = bine / sota      
                samples_big += 1      
        elif coll_to_search == "reduce_scatter":
            bine = float(t[t["ALGORITHM"] == "bine_doubling"]["EXTERNAL"].iloc[0])
            sota = float(t[t["ALGORITHM"] == "distance_doubling"]["EXTERNAL"].iloc[0])            
            if bine != 0 and sota != 0:
                big = bine / sota
                samples_big += 1
        elif coll_to_search == "alltoall":
            bine = float(t[t["ALGORITHM"] == "bine"]["EXTERNAL"].iloc[0])
            sota = float(t[t["ALGORITHM"] == "bruck"]["EXTERNAL"].iloc[0])            
            if bine != 0 and sota != 0:
                big = bine / sota
                samples_big += 1
        elif coll_to_search == "bcast":
            bine = float(t[t["ALGORITHM"] == "bine_halving"]["EXTERNAL"].iloc[0])
            sota = float(t[t["ALGORITHM"] == "binomial_halving"]["EXTERNAL"].iloc[0])            
            if bine != 0 and sota != 0:
                small = bine / sota
                samples_small += 1
            bine = float(t[t["ALGORITHM"] == "bine_bdw_doubling_halving"]["EXTERNAL"].iloc[0])
            sota = float(t[t["ALGORITHM"] == "binomial_bdw_halving_doubling"]["EXTERNAL"].iloc[0])            
            if bine != 0 and sota != 0:
                big = bine / sota
                samples_big += 1
        elif coll_to_search == "reduce":
            bine = float(t[t["ALGORITHM"] == "bine_halving"]["EXTERNAL"].iloc[0])
            sota = float(t[t["ALGORITHM"] == "binomial_halving"]["EXTERNAL"].iloc[0])            
            if bine != 0 and sota != 0:
                small = bine / sota
                samples_small += 1
            bine = float(t[t["ALGORITHM"] == "bine_bdw_halving_doubling"]["EXTERNAL"].iloc[0])
            sota = float(t[t["ALGORITHM"] == "binomial_bdw_halving_doubling"]["EXTERNAL"].iloc[0])            
            if bine != 0 and sota != 0:
                big = bine / sota 
                samples_big += 1           
        elif coll_to_search == "reduce_scatter":
            bine = float(t[t["ALGORITHM"] == "distance_doubling"]["EXTERNAL"].iloc[0])
            sota = float(t[t["ALGORITHM"] == "bine_doubling"]["EXTERNAL"].iloc[0])            
            if bine != 0 and sota != 0:
                big = bine / sota   
                samples_big += 1         
        elif coll_to_search == "scatter":
            bine = float(t[t["ALGORITHM"] == "bine_halving"]["EXTERNAL"].iloc[0])
            sota = float(t[t["ALGORITHM"] == "binomial_halving"]["EXTERNAL"].iloc[0])            
            if bine != 0 and sota != 0:
                big = bine / sota   
                samples_big += 1         

        #print(f"nodes: {nodes} Small: {small}, Big: {big}")
        if compute_max:
            if small < ratio_small_v and small != 0:
                ratio_small_v = small
            if big < ratio_big_v and big != 0:
                ratio_big_v = big
        else:
            ratio_small_v += small
            ratio_big_v += big    

    if samples_small != 0:
        if compute_max:
            samples_small = 1
        ratio_small_v /= float(samples_small)
        ratio_small_v = 1.0 - ratio_small_v
    if samples_big != 0:
        if compute_max:
            samples_big = 1
        ratio_big_v /= float(samples_big)   
        ratio_big_v = 1.0 - ratio_big_v
    return (ratio_small_v, ratio_big_v)

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

        # Read the data
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
    

def augment_df(df, metric, keep_all_ratios=False, reference="Bine"):
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
        elif ratio >= 1.0:
            cell = ratio         
        else:
            if keep_all_ratios: # How much is bine faster/slower wrt. binomial
                if best_algo == reference:
                    cell = best_algo_row['bandwidth_' + metric] / second_best_algo_row['bandwidth_' + metric]  
                else:
                    cell = second_best_algo_row['bandwidth_' + metric] / best_algo_row['bandwidth_' + metric] 
            else:
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
    parser.add_argument("--bvb", help="Compare Bine only with Binomial", action="store_true")
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

    df_all = df.copy()
    df_all = augment_df(df_all, args.metric)
    df_numeric_all = df_all.copy()
    df_numeric_all['cell'] = pd.to_numeric(df_all['cell'], errors='coerce')
    
    coll = args.collective.capitalize().replace("_", "")
    if coll.lower() == "reducescatter":
        coll = "Red.-Scat."

    if args.bvb:
        df_bvb = df.copy()
        if args.collective.lower() == "alltoall":
            filter = ["Bruck", "Bine"]        
        else:
            filter = ["Binomial", "Bine"]
        df_bvb = df_bvb[df_bvb['algo_family'].isin(filter)]
        df_bvb = augment_df(df_bvb, args.metric, True)
        df_numeric_bvb = df_bvb.copy()
        df_numeric_bvb['cell'] = pd.to_numeric(df_bvb['cell'], errors='coerce')

        total_valid = df_numeric_bvb["cell"].notna().sum()    
        
        # Rows with cell > 1
        greater_than_1 = df_numeric_bvb[df_numeric_bvb["cell"] > 1]
        count_gt1 = (len(greater_than_1)/total_valid)*100.0
        gmean_gt1 = gmean(greater_than_1["cell"])
        max_gt1 = greater_than_1["cell"].max()
        gmean_gt1 = (gmean_gt1 - 1) * 100.0
        max_gt1 = (max_gt1 - 1) * 100.0

        # Rows with cell < 1
        less_than_1 = df_numeric_bvb[df_numeric_bvb["cell"] < 1]
        if less_than_1["cell"].empty:
            count_lt1 = 0.0
            gmean_lt1 = 0.0
            min_lt1 = 0.0
        else:        
            count_lt1 = (len(less_than_1)/total_valid)*100.0
            gmean_lt1 = gmean(less_than_1["cell"])
            min_lt1 = less_than_1["cell"].min()
            gmean_lt1 = (gmean_lt1 - 1) * 100.0
            min_lt1 = (min_lt1 - 1) * 100.0
            gmean_lt1 = -gmean_lt1 
            min_lt1 = -min_lt1

        #print(f"Bine wins: {count_gt1:.2f}% ({gmean_gt1:.2f}%/{max_gt1:.2f}%) Binomial wins: {count_lt1:.2f}% ({gmean_lt1:.2f}%/{min_lt1:.2f}%)")
        """
        if coll.lower() == "alltoall":
            coll = "A2A"
        elif coll.lower() == "allgather":
            coll = "AG"
        elif coll.lower() == "allreduce":
            coll = "AR"
        elif coll.lower() == "scatter":
            coll = "SCT"
        elif coll.lower() == "gather":
            coll = "GAT"
        elif coll.lower() == "bcast":
            coll = "BCST"
        elif coll.lower() == "reduce":
            coll = "RED"
        elif coll.lower() == "reducescatter":
            coll = "RDSC"
        """

        small_red, big_red = get_tracer_out(args, False)
        small_red *= 100.0
        big_red *= 100.0

        small_red_max, big_red_max = get_tracer_out(args, True)
        small_red_max *= 100.0
        big_red_max *= 100.0        

        red = f"{big_red:.0f}\\%/{big_red_max:.0f}\\%"

        """
        if small_red != 0:
            red = f"{small_red:.0f}\\%/{big_red:.0f}\\%"
        else:
            red = f"{big_red:.0f}\\%"
        """

        #print(f"{coll} & {count_gt1:.2f}\\% & {gmean_gt1:.2f}\\%/{max_gt1:.2f}\\% & {count_lt1:.2f}\\% & {gmean_lt1:.2f}\\%/{min_lt1:.2f}\\% & \\\\")
        print(f"{coll} & {count_gt1:.0f}\\% & {gmean_gt1:.0f}\\%/{max_gt1:.0f}\\% & {count_lt1:.0f}\\% & {gmean_lt1:.0f}\\%/{min_lt1:.0f}\\% & {red} \\\\")
        
    else:
        wins_sota = 100 * (df_numeric_all['cell'].notna().sum() / df_numeric_all.shape[0])
        mean_impr_sota = (df_numeric_all['cell'].mean() - 1)*100.0
        median_impr_sota = (df_numeric_all['cell'].median() - 1)*100.0
        max_impr_sota = (df_numeric_all['cell'].max() - 1)*100.0 
        # Replace numbers < 1.0 with nan
        #df_numeric_bvb.loc[df_numeric_bvb['cell'] < 1.0, 'cell'] = np.nan
        #wins_bvb = 100 * (df_numeric_bvb['cell'].notna().sum() / df_numeric_bvb.shape[0])
        #mean_impr_bvb = (df_numeric_bvb['cell'].mean() - 1)*100.0
        #max_impr_bvb = (df_numeric_bvb['cell'].max() - 1)*100.0    
        print(f"{coll} & {wins_sota:.0f}% & {mean_impr_sota:.0f}% & {median_impr_sota:.0f}% & {max_impr_sota:.0f}%  \\\\")
        #print(df_numeric_all)
    
if __name__ == "__main__":
    main()

