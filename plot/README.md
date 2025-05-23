 # Introduction
 This directory contains the code for plotting the results of the experiments (data stored in the `results` directory). All the scripts should be called from the repository root directory.

- `summarize_data.py` postprocesses the data in `results`, generating summaries that are used in plots.
- `create_graphs.py` original Saverio's script for creating lineplots/barplots
- `heatmaps.sh` produces all the heatmaps. It relies on the `heatmap.py` script.
- `boxplots.sh` produces the boxplots shown in the paper. It calls the `boxplot.py` script. 
- `tables.sh` generates the tables used in the paper. It calls the `table.py` script.