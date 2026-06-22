#!/usr/bin/env bash
set -euo pipefail

METRIC="${METRIC:-mean}"
SYSTEM="${SYSTEM:-leonardo}"
RUNS_STR="${RUNS:-}"


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

# Default datasets per system (timestamp:nodes pairs).
declare -A BASE_RUNS=(
  [leonardo]="2025_04_05___23_20_55:256 2025_04_06___13_24_12:128 2025_04_06___13_24_31:64 2025_04_06___13_24_51:32 2025_04_06___13_25_12:16 2025_04_06___13_25_25:8 2025_04_06___13_25_39:4"
  [mare_nostrum]="2025_04_14___03_44_45:64 2025_04_12___14_04_47:32 2025_04_10___19_19_35:16 2025_04_10___19_18_47:8 2025_04_10___17_40_37:4"
  [lumi]="2025_04_07___18_42_37:8 2025_04_10___16_25_22:16 2025_04_10___16_24_38:32 2025_04_09___00_15_52:64 2025_04_10___14_26_48:128 2025_04_10___15_46_57:256 2025_04_10___18_47_49:512 2025_04_09___16_00_41:1024"
)

declare -A STANDARD_PLOTS_ENABLED=(
  [leonardo]=1
)

collect_summary_timestamps() {
  local ordered=()
  declare -A seen=()
  for entry in "$@"; do
    [[ -z "$entry" ]] && continue
    for pair in $entry; do
      local ts="${pair%%:*}"
      if [[ -z "${seen[$ts]:-}" ]]; then
        seen["$ts"]=1
        ordered+=("$ts")
      fi
    done
  done
  echo "${ordered[*]}"
}

merge_run_lists() {
  local ordered=()
  declare -A seen=()
  for entry in "$@"; do
    [[ -z "$entry" ]] && continue
    for pair in $entry; do
      if [[ -z "${seen[$pair]:-}" ]]; then
        seen["$pair"]=1
        ordered+=("$pair")
      fi
    done
  done
  echo "${ordered[*]}"
}

filter_run_list_by_max_nodes() {
  local run_list="$1"
  local max_nodes="$2"
  local filtered=()
  for pair in $run_list; do
    local nodes="${pair##*:}"
    [[ -z "$nodes" ]] && continue
    if (( nodes <= max_nodes )); then
      filtered+=("$pair")
    fi
  done
  echo "${filtered[*]}"
}

declare -A BEST_BINE_ALLGATHER_RUNS=()
BEST_BINE_ALLGATHER_RUNS[leonardo]="$(merge_run_lists \
  "2025_03_28___16_48_20:2048 2025_03_28___17_42_08:1024 2025_03_28___18_19_43:512" \
  "${BASE_RUNS[leonardo]}")"
BEST_BINE_ALLGATHER_RUNS[mare_nostrum]="${BASE_RUNS[mare_nostrum]}"
BEST_BINE_ALLGATHER_RUNS[lumi]="${BASE_RUNS[lumi]}"

declare -A BEST_BINE_ALLGATHER_OUTPUTS=(
  [leonardo]="plot/leonardo/heatmaps/allgather/leonardo_allgather_best_bine_variant_2048.pdf"
  [mare_nostrum]=""
  [lumi]=""
)

declare -A BEST_BINE_ALLGATHER_RUNS_64=()
if [[ -n "${BEST_BINE_ALLGATHER_RUNS[leonardo]:-}" ]]; then
  BEST_BINE_ALLGATHER_RUNS_64[leonardo]="$(filter_run_list_by_max_nodes "${BEST_BINE_ALLGATHER_RUNS[leonardo]}" 64)"
fi

declare -A BEST_BINE_ALLGATHER_OUTPUTS_64=(
  [leonardo]="plot/leonardo/heatmaps/allgather/leonardo_allgather_best_bine_variant_64.pdf"
)

declare -A BEST_BINE_REDUCE_SCATTER_RUNS=()
BEST_BINE_REDUCE_SCATTER_RUNS[leonardo]=""
BEST_BINE_REDUCE_SCATTER_RUNS[mare_nostrum]="${BASE_RUNS[mare_nostrum]}"
BEST_BINE_REDUCE_SCATTER_RUNS[lumi]="${BASE_RUNS[lumi]}"

declare -A BEST_BINE_REDUCE_SCATTER_OUTPUTS=(
  [leonardo]=""
  [mare_nostrum]=""
  [lumi]=""
)

run_best_bine_plot() {
  local collective="$1"
  local run_list="$2"
  local output_path="$3"

  [[ -z "$run_list" ]] && return

  local -a run_array=()
  split_run_list "$run_list" run_array
  if (( ${#run_array[@]} == 0 )); then
    return
  fi

  local -a cmd=(python3 -m plot bine-heatmap --system "${SYSTEM}" --collective "${collective}" --metric "${METRIC}")
  cmd+=(--runs "${run_array[@]}")
  if [[ -n "$output_path" ]]; then
    cmd+=(--output "$output_path")
  fi
  "${cmd[@]}"
}


# Convert RUNS string into --runs args if provided
split_run_list() {
  local input="$1"
  local -n out_ref="$2"
  out_ref=()
  if [[ -n "$input" ]]; then
    # shellcheck disable=SC2206
    out_ref=($input)
  fi
}

RUNS_ARRAY=()
split_run_list "${RUNS_STR}" RUNS_ARRAY

RUN_STANDARD_PLOTS=0
if (( ${#RUNS_ARRAY[@]} )); then
  RUN_STANDARD_PLOTS=1
elif [[ "${STANDARD_PLOTS_ENABLED[$SYSTEM]:-0}" == "1" ]]; then
  RUN_STANDARD_PLOTS=1
fi


# Pre-summarize data, skipping if summary file already exists
DEFAULT_SUMMARY_RUNS="$(collect_summary_timestamps \
  "${BASE_RUNS[$SYSTEM]:-}" \
  "${BEST_BINE_ALLGATHER_RUNS[$SYSTEM]:-}" \
  "${BEST_BINE_REDUCE_SCATTER_RUNS[$SYSTEM]:-}")"

if [[ -n "${RUNS_STR}" ]]; then
  # Only summarize those explicitly requested
  for pair in "${RUNS_ARRAY[@]}"; do
    ts="${pair%%:*}"
    result_dir="results/${SYSTEM}/${ts}"
    summary_file="$result_dir/aggregated_results_summary.csv"

    if [[ -d "$result_dir" ]]; then
      if [[ ! -f "$summary_file" ]]; then
        echo "Summarizing data for ${ts}..."
        python3 ./plot/summarize_data.py --result-dir "$result_dir"
      else
        echo "Skipping summarization for ${ts} (already exists)."
      fi
    else
      echo "Missing results directory for ${ts}" >&2
      exit 1
    fi
  done
else
  if [[ -z "${DEFAULT_SUMMARY_RUNS}" ]]; then
    echo "No default run list for system ${SYSTEM}. Provide RUNS to summarize." >&2
    exit 1
  fi
  # Summarize every run needed for the default plots
  for ts in ${DEFAULT_SUMMARY_RUNS}; do
    result_dir="results/${SYSTEM}/${ts}"
    summary_file="$result_dir/aggregated_results_summary.csv"
    if [[ -d "$result_dir" ]]; then
      if [[ ! -f "$summary_file" ]]; then
        echo "Summarizing data for ${ts}..."
        python3 ./plot/summarize_data.py --result-dir "$result_dir"
      else
        echo "Skipping summarization for ${ts} (already exists)."
      fi
    else
      echo "Missing results directory for ${ts}" >&2
      exit 1
    fi
  done
fi


if (( RUN_STANDARD_PLOTS )); then
  # Call the single Python script with the chosen options
  ALLGATHER_CMD=(python3 -m plot bine-heatmap --system "${SYSTEM}" --collective ALLGATHER --metric "${METRIC}")
  if (( ${#RUNS_ARRAY[@]} )); then
    ALLGATHER_CMD+=(--runs "${RUNS_ARRAY[@]}")
  elif [[ -n "${BASE_RUNS[$SYSTEM]:-}" ]]; then
    DEFAULT_RUNS_ARRAY=()
    split_run_list "${BASE_RUNS[$SYSTEM]}" DEFAULT_RUNS_ARRAY
    ALLGATHER_CMD+=(--runs "${DEFAULT_RUNS_ARRAY[@]}")
  fi
  "${ALLGATHER_CMD[@]}"

  REDUCE_SCATTER_CMD=(python3 -m plot bine-heatmap --system "${SYSTEM}" --collective REDUCE_SCATTER --metric "${METRIC}")
  if (( ${#RUNS_ARRAY[@]} )); then
    REDUCE_SCATTER_CMD+=(--runs "${RUNS_ARRAY[@]}")
  elif [[ -n "${BASE_RUNS[$SYSTEM]:-}" ]]; then
    DEFAULT_RUNS_ARRAY=()
    split_run_list "${BASE_RUNS[$SYSTEM]}" DEFAULT_RUNS_ARRAY
    REDUCE_SCATTER_CMD+=(--runs "${DEFAULT_RUNS_ARRAY[@]}")
  fi
  "${REDUCE_SCATTER_CMD[@]}"
else
  echo "Skipping standard ALLGATHER/REDUCE_SCATTER heatmaps for ${SYSTEM} (only best-Bine plots requested)."
fi

run_best_bine_plot "ALLGATHER" "${BEST_BINE_ALLGATHER_RUNS[$SYSTEM]:-}" "${BEST_BINE_ALLGATHER_OUTPUTS[$SYSTEM]:-}"
run_best_bine_plot "ALLGATHER" "${BEST_BINE_ALLGATHER_RUNS_64[$SYSTEM]:-}" "${BEST_BINE_ALLGATHER_OUTPUTS_64[$SYSTEM]:-}"
run_best_bine_plot "REDUCE_SCATTER" "${BEST_BINE_REDUCE_SCATTER_RUNS[$SYSTEM]:-}" "${BEST_BINE_REDUCE_SCATTER_OUTPUTS[$SYSTEM]:-}"
