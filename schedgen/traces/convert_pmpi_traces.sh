#!/usr/bin/env bash
# Convert all PMPI traces under the local traces subtree into GOAL schedules
# using schedgen's trace replay mode with dynamic collective selection.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCHEDGEN_BIN="${SCRIPT_DIR}/../bin/schedgen"
RULES_FILE="${SCRIPT_DIR}/dynamic_selector_rules.txt"
TRACE_ROOT="${SCRIPT_DIR}/hpc/lulesh"

if [[ ! -x "${SCHEDGEN_BIN}" ]]; then
  echo "error: schedgen binary not found or not executable at ${SCHEDGEN_BIN}" >&2
  exit 1
fi

if [[ ! -f "${RULES_FILE}" ]]; then
  echo "error: selector rules file missing at ${RULES_FILE}" >&2
  exit 1
fi

if [[ ! -d "${TRACE_ROOT}" ]]; then
  echo "error: trace root directory missing at ${TRACE_ROOT}" >&2
  exit 1
fi

extra_args=("$@")

mapfile -t trace_dirs < <(find "${TRACE_ROOT}" -type d -name "mpi_traces" | sort)

if [[ ${#trace_dirs[@]} -eq 0 ]]; then
  echo "warning: no mpi_traces directories found under ${TRACE_ROOT}" >&2
  exit 0
fi

for trace_dir in "${trace_dirs[@]}"; do
  dataset_dir=$(dirname "${trace_dir}")
  dataset_name=$(basename "${dataset_dir}")
  rank0_trace="${trace_dir}/pmpi-trace-rank-0.txt"

  if [[ ! -f "${rank0_trace}" ]]; then
    echo "skipping ${dataset_name}: missing ${rank0_trace}" >&2
    continue
  fi

  goal_output="${dataset_dir}/${dataset_name}_opt_27_ar.goal"

  echo "Generating ${goal_output} from ${rank0_trace}"
  if ! "${SCHEDGEN_BIN}" \
    --ptrn=trace \
    --traces "${rank0_trace}" \
    --filename "${goal_output}" \
    --selector-rules "${RULES_FILE}" \
    "${extra_args[@]}"; then
    status=$?
    echo "error: schedgen failed with exit code ${status} for ${rank0_trace}" >&2
    continue
  fi
done
