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
#rcParams['figure.figsize'] = 6.75,0.421875
rcParams['figure.figsize'] = 6.75,0.84375
big_font_size = 18
small_font_size = 15


# Parse command line arguments
parser = argparse.ArgumentParser(description='Plot sinfo summary')
parser.add_argument('--min_ranks', type=int, help='Minimum number of ranks to consider')
args = parser.parse_args()

df_leo = pd.read_csv('sinfo_summary_leonardo.csv')
df_leo["System"] = "Leonardo"
df_lumi = pd.read_csv('sinfo_summary_lumi.csv')
df_lumi["System"] = "LUMI"
df = pd.concat([df_leo, df_lumi], ignore_index=True)

# Discard all the rows where "Nodes" < args.min_ranks
if args.min_ranks:
    df = df[df["Nodes"] >= args.min_ranks]


###########
# Boxplot #
###########
# Custom mean marker style
mean_props = {
    "marker": "o",
    "markerfacecolor": "black",  # Fill color
    "markeredgecolor": "black",  # Border color
    "markersize": 8
}


# Make a boxplot, using "Collective" as x-axis and "Improvement" as y-axis
# Set the figure size
plt.figure()
#sns.scatterplot(data=df, x='Nodes', y='Reduction')
sns.boxplot(data=df, x='Reduction', y="System", showmeans=True, meanprops=mean_props)
# remove y-title
plt.ylabel("")
plt.xlabel("Global Links Traffic Reduction (%)")

# Save as PDF
plt.savefig("sinfo_summary_min_" + str(args.min_ranks) + ".pdf", bbox_inches="tight")

###############
# Scatterplot #
###############
# Randomze df rows order
df = df.sample(frac=1).reset_index(drop=True)
rcParams['figure.figsize'] = 13,13
plt.clf()
plt.figure()
sns.scatterplot(data=df, x='Nodes', y='Reduction', hue="System", style="System", markers=["o", "s"], s=100)
# x logscale
plt.xscale("log")
plt.savefig("sinfo_summary_scatter_min_" + str(args.min_ranks) + ".pdf", bbox_inches="tight")