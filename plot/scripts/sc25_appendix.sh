#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

summarize_and_plot() {
    local dir="$1"
    python3 "${REPO_ROOT}/plot/summarize_data.py" --result-dir "$dir" &> /dev/null
    (cd "$REPO_ROOT" && python3 -m plot summary --summary-file "${dir}/aggregated_results_summary.csv")
}

#############################
# Non-power of 2 node count #
#############################
for dir in \
    "${REPO_ROOT}/results/leonardo/2025_06_04___15_12_34" \
    "${REPO_ROOT}/results/lumi/2025_04_12___00_29_36" \
    "${REPO_ROOT}/results/mare_nostrum/2025_06_05___20_58_44"
do
    summarize_and_plot "$dir"
done

########
# GPUs #
########
for dir in "${REPO_ROOT}/results/mare_nostrum/2025_04_14___04_39_29"
do
    summarize_and_plot "$dir"
    last_path="$(basename "$dir")"
    mkdir -p "${REPO_ROOT}/plot/paper_appendix/mare_nostrum/gpu"
    mv "${REPO_ROOT}/plot/mare_nostrum/"*"$last_path"* "${REPO_ROOT}/plot/paper_appendix/mare_nostrum/gpu/"
done

###############################################################
# Detailed plots expanding the heatmaps/boxplots in the paper #
###############################################################
summary_files=(
    "${REPO_ROOT}/results/fugaku/2025_03_26___19_49_06/aggregated_results_summary.csv"
    "${REPO_ROOT}/results/fugaku/2025_03_27___04_01_47/aggregated_results_summary.csv"
    "${REPO_ROOT}/results/fugaku/2025_04_06___02_35_52/aggregated_results_summary.csv"
    "${REPO_ROOT}/results/fugaku/2025_04_09___17_37_37/aggregated_results_summary.csv"
    "${REPO_ROOT}/results/fugaku/2025_04_12___03_23_58/aggregated_results_summary.csv"
    "${REPO_ROOT}/results/leonardo/2025_03_28___16_48_20/aggregated_results_summary.csv"
    "${REPO_ROOT}/results/leonardo/2025_03_28___17_42_08/aggregated_results_summary.csv"
    "${REPO_ROOT}/results/leonardo/2025_03_28___18_19_43/aggregated_results_summary.csv"
    "${REPO_ROOT}/results/leonardo/2025_04_05___23_20_55/aggregated_results_summary.csv"
    "${REPO_ROOT}/results/leonardo/2025_04_06___13_24_12/aggregated_results_summary.csv"
    "${REPO_ROOT}/results/leonardo/2025_05_01___17_13_22/aggregated_results_summary.csv"
    "${REPO_ROOT}/results/leonardo/2025_05_01___17_13_53/aggregated_results_summary.csv"
    "${REPO_ROOT}/results/leonardo/2025_05_01___17_14_29/aggregated_results_summary.csv"
    "${REPO_ROOT}/results/leonardo/2025_06_04___14_26_23/aggregated_results_summary.csv"
    "${REPO_ROOT}/results/lumi/2025_04_09___00_15_52/aggregated_results_summary.csv"
    "${REPO_ROOT}/results/lumi/2025_04_09___16_00_41/aggregated_results_summary.csv"
    "${REPO_ROOT}/results/lumi/2025_04_10___14_26_48/aggregated_results_summary.csv"
    "${REPO_ROOT}/results/lumi/2025_04_10___15_46_57/aggregated_results_summary.csv"
    "${REPO_ROOT}/results/lumi/2025_04_10___16_24_38/aggregated_results_summary.csv"
    "${REPO_ROOT}/results/lumi/2025_04_10___16_25_22/aggregated_results_summary.csv"
    "${REPO_ROOT}/results/lumi/2025_04_10___18_47_49/aggregated_results_summary.csv"
    "${REPO_ROOT}/results/mare_nostrum/2025_04_10___17_40_37/aggregated_results_summary.csv"
    "${REPO_ROOT}/results/mare_nostrum/2025_04_10___19_18_47/aggregated_results_summary.csv"
    "${REPO_ROOT}/results/mare_nostrum/2025_04_10___19_19_35/aggregated_results_summary.csv"
    "${REPO_ROOT}/results/mare_nostrum/2025_04_10___20_36_45/aggregated_results_summary.csv"
    "${REPO_ROOT}/results/mare_nostrum/2025_04_12___14_04_47/aggregated_results_summary.csv"
    "${REPO_ROOT}/results/mare_nostrum/2025_04_12___14_07_21/aggregated_results_summary.csv"
    "${REPO_ROOT}/results/mare_nostrum/2025_04_14___03_44_45/aggregated_results_summary.csv"
    "${REPO_ROOT}/results/mare_nostrum/2025_04_14___12_48_39/aggregated_results_summary.csv"
    "${REPO_ROOT}/results/mare_nostrum/2025_06_05___15_11_38/aggregated_results_summary.csv"
)

for summary_file in "${summary_files[@]}"; do
    (cd "$REPO_ROOT" && python3 -m plot summary --summary-file "$summary_file")
done

##################################################
# Move all plots to the paper_appendix directory #
##################################################
for SYSTEM in "leonardo" "fugaku" "lumi" "mare_nostrum"; do
    mkdir -p "${REPO_ROOT}/plot/paper_appendix/${SYSTEM}"
    mv "${REPO_ROOT}/plot/${SYSTEM}/"*.png "${REPO_ROOT}/plot/paper_appendix/${SYSTEM}/"
done
