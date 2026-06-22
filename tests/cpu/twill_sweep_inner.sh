#!/bin/bash
# Runs as the body of the sbatch job submitted by tests/cpu/twill_sweep.sh, i.e.
# ON A COMPUTE NODE inside the job's allocation. Iterates the (window x cache x
# group) grid; for each combo it invokes the stock submit_wrapper in LOCATION=local
# mode (via tests/cpu/twill_sweep_local.sh), which runs the stock orchestrator
# in-process and `srun`s onto THIS job's allocated nodes -- because we are inside
# the allocation, each srun attaches as a job STEP (not a new job). So every combo
# runs on the same nodes => comparable, and the whole sweep is a single squeue job.
# Reads the grid + config path from the environment exported by twill_sweep.sh
# (propagated into the job via sbatch --export=ALL).

set -u
# Repo root: prefer TWILL_REPO (exported by twill_sweep.sh, propagated via sbatch
# --export=ALL). Under sbatch the script body is a spool-dir copy, so BASH_SOURCE
# would NOT resolve to the repo. Fall back to SLURM_SUBMIT_DIR, then BASH_SOURCE.
HERE="${TWILL_REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
cd "$HERE" || { echo "cannot cd to repo root $HERE" >&2; exit 1; }
[[ -f scripts/submit_wrapper.sh ]] || { echo "repo root '$HERE' has no scripts/submit_wrapper.sh (set TWILL_REPO)" >&2; exit 1; }

CONFIG="${TWILL_SWEEP_CONFIG:-tests/cpu/twill_sweep_local.sh}"
n=0
for w in $TWILL_WINDOWS; do for c in $TWILL_CACHES; do for g in $TWILL_GROUPS; do n=$((n + 1)); done; done; done

i=0
for w in $TWILL_WINDOWS; do
  for c in $TWILL_CACHES; do
    for g in $TWILL_GROUPS; do
      i=$((i + 1))
      echo "=================================================================="
      echo " [$i/$n] combo: TWILL_WINDOW=$w TWILL_CACHE=$c TWILL_GROUP=$g"
      echo "=================================================================="
      # orchestrator.sh exits 1 when LOCATION=local (its last line is a
      # short-circuit `[[ $LOCATION != local ]] && squeue` which returns 1
      # when the condition is false). Don't rely on the exit code; instead
      # verify success by checking that a new entry appeared in results/local/.
      _before=$(ls -1 results/local/ 2>/dev/null | wc -l)
      TWILL_WINDOW="$w" TWILL_CACHE="$c" TWILL_GROUP="$g" \
        bash scripts/submit_wrapper.sh --file "$CONFIG"
      _after=$(ls -1 results/local/ 2>/dev/null | wc -l)
      if [[ "$_after" -le "$_before" ]]; then
        echo "combo failed (no new results written): W=$w cache=$c group=$g" >&2
        exit 1
      fi
    done
  done
done
echo "TWILL sweep: all $n combos done (same allocation)."
