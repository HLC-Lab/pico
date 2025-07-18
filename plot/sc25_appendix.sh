#!/bin/bash
# A list of summary files to process
summary_files=(
    "results/fugaku/2025_03_26___19_49_06/aggregated_results_summary.csv"
    "results/fugaku/2025_03_27___04_01_47/aggregated_results_summary.csv"
    "results/fugaku/2025_04_06___02_35_52/aggregated_results_summary.csv"
    "results/fugaku/2025_04_09___17_37_37/aggregated_results_summary.csv"
    "results/fugaku/2025_04_12___03_23_58/aggregated_results_summary.csv"
    "results/leonardo/2025_03_28___16_48_20/aggregated_results_summary.csv"
    "results/leonardo/2025_03_28___17_42_08/aggregated_results_summary.csv"
    "results/leonardo/2025_03_28___18_19_43/aggregated_results_summary.csv"
    "results/leonardo/2025_04_05___23_20_55/aggregated_results_summary.csv"
    "results/leonardo/2025_04_06___13_24_12/aggregated_results_summary.csv"
    "results/leonardo/2025_05_01___17_13_22/aggregated_results_summary.csv"
    "results/leonardo/2025_05_01___17_13_53/aggregated_results_summary.csv"
    "results/leonardo/2025_05_01___17_14_29/aggregated_results_summary.csv"
    "results/leonardo/2025_06_04___14_26_23/aggregated_results_summary.csv"
    "results/lumi/2025_04_09___00_15_52/aggregated_results_summary.csv"
    "results/lumi/2025_04_09___16_00_41/aggregated_results_summary.csv"
    "results/lumi/2025_04_10___14_26_48/aggregated_results_summary.csv"
    "results/lumi/2025_04_10___15_46_57/aggregated_results_summary.csv"
    "results/lumi/2025_04_10___16_24_38/aggregated_results_summary.csv"
    "results/lumi/2025_04_10___16_25_22/aggregated_results_summary.csv"
    "results/lumi/2025_04_10___18_47_49/aggregated_results_summary.csv"
    "results/mare_nostrum/2025_04_10___17_40_37/aggregated_results_summary.csv"
    "results/mare_nostrum/2025_04_10___19_18_47/aggregated_results_summary.csv"
    "results/mare_nostrum/2025_04_10___19_19_35/aggregated_results_summary.csv"
    "results/mare_nostrum/2025_04_10___20_36_45/aggregated_results_summary.csv"
    "results/mare_nostrum/2025_04_12___14_04_47/aggregated_results_summary.csv"
    "results/mare_nostrum/2025_04_12___14_07_21/aggregated_results_summary.csv"
    "results/mare_nostrum/2025_04_14___03_44_45/aggregated_results_summary.csv"
    "results/mare_nostrum/2025_04_14___12_48_39/aggregated_results_summary.csv"
    "results/mare_nostrum/2025_06_05___15_11_38/aggregated_results_summary.csv"
)

# Loop through each summary file and create graphs
for summary_file in "${summary_files[@]}"; do
    python3 plot/create_graphs.py --summary-file "$summary_file"
done

# Move them to the paper_appendix directory
for SYSTEM in "leonardo" "fugaku" "lumi" "mare_nostrum"; do
    mkdir -p plot/paper_appendix/$SYSTEM
    mv plot/$SYSTEM/*.png plot/paper_appendix/$SYSTEM/
done