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
rcParams['figure.figsize'] = 6.75,0.421875
big_font_size = 18
small_font_size = 15


df = pd.read_csv('sinfo_summary.csv')

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
sns.boxplot(data=df, x='Reduction', showmeans=True, meanprops=mean_props)
# remove y-title
plt.ylabel("")
plt.xlabel("Global Links Traffic Reduction (%)")


# Save as PDF
plt.savefig("sinfo_summary.pdf", bbox_inches="tight")