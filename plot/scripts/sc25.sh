#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "Generating tables ..."
"${SCRIPT_DIR}/tables.sh" &> /dev/null

echo "Generating boxplots ..."
"${SCRIPT_DIR}/boxplots.sh" &> /dev/null

echo "Generating heatmaps ..."
"${SCRIPT_DIR}/heatmaps.sh" &> /dev/null

echo "Generating fig. 5 ..."
pushd "${REPO_ROOT}/tracer/sinfo" >/dev/null
python3 plot.py &> /dev/null
popd >/dev/null

mkdir -p "${REPO_ROOT}/plot/paper"

# Fig. 5
cp "${REPO_ROOT}/tracer/sinfo/multi_box_min_None_allreduce_rabenseifner_vs_bine_bandwidth.pdf" "${REPO_ROOT}/plot/paper/fig5.pdf"

# Fig. 8a
cp "${REPO_ROOT}/plot/lumi/heatmaps/allreduce/tasks_per_node_1_metric_mean_base_all_y_no_False.pdf" "${REPO_ROOT}/plot/paper/fig8a.pdf"

# Fig. 8b
cp "${REPO_ROOT}/plot/lumi/boxplot.pdf" "${REPO_ROOT}/plot/paper/fig8b.pdf"

# Fig. 9a
cp "${REPO_ROOT}/plot/leonardo/heatmaps/allreduce/tasks_per_node_1_metric_mean_base_all_y_no_False.pdf" "${REPO_ROOT}/plot/paper/fig9a.pdf"

# Fig. 9b
cp "${REPO_ROOT}/plot/leonardo/boxplot.pdf" "${REPO_ROOT}/plot/paper/fig9b.pdf"

# Fig. 10a
cp "${REPO_ROOT}/plot/mare_nostrum/boxplot.pdf" "${REPO_ROOT}/plot/paper/fig10a.pdf"

# Fig. 10b
cp "${REPO_ROOT}/plot/fugaku/boxplot.pdf" "${REPO_ROOT}/plot/paper/fig10b.pdf"

# Table 3
cp "${REPO_ROOT}/plot/lumi/bine_vs_binomial.txt" "${REPO_ROOT}/plot/paper/table3.txt"

# Table 4
cp "${REPO_ROOT}/plot/leonardo/bine_vs_binomial.txt" "${REPO_ROOT}/plot/paper/table4.txt"

# Table 5
cp "${REPO_ROOT}/plot/mare_nostrum/bine_vs_binomial.txt" "${REPO_ROOT}/plot/paper/table5.txt"
