#!/bin/bash
# TWILL sweep — ONE batch job, reusing the stock submit_wrapper/orchestrator
# UNMODIFIED. This script submits a SINGLE `sbatch` job whose body
# (tests/cpu/twill_sweep_inner.sh) iterates the whole (window x cache x group)
# grid INSIDE the job's allocation. Every combo's `srun` therefore attaches to
# that one allocation as a job STEP (not a new job), so:
#   * all combos run on the SAME nodes  => timings are directly comparable
#     (separate jobs would land on different dragonfly placement -> not comparable);
#   * `squeue` shows exactly ONE job for the whole sweep.
# (An earlier version used `salloc <command>`, which runs the body on the LOGIN
# node and relies on srun inheriting the allocation via SLURM_JOB_ID; on Leonardo
# that inheritance does not happen, so every srun self-submitted its own job.
# sbatch runs the body on a compute node inside the allocation -> reliable steps.)
#
#   export PICO_ACCOUNT=<your_slurm_project>
#   bash tests/cpu/twill_sweep.sh                      # returns immediately with a job id
#
# Knobs via env (space-separated lists):
#   TWILL_WINDOWS="8 32" TWILL_CACHES="1" TWILL_GROUPS="4 8 16 32 node" bash tests/cpu/twill_sweep.sh
# Defaults: {2 8 32 128} x {0 1} x {4 8 16 32}. GROUPS accepts integers (synthetic
# uniform group size) or "node". TWILL_ONLY=yes (default) sweeps only the twill
# variants (vendor/bine baselines are knob-invariant; get them from one full-field
# run: bash scripts/submit_wrapper.sh --file tests/cpu/twill_alltoall.sh).
#
# Each combo lands in its own results/local/<ts>/ run (LOCATION=local; the combo
# is recorded in the metadata NOTES). Job stdout/stderr -> twill_sweep_<jobid>.out/err.
# Compare with utils/twill_compare.py (--cross).
#
# NOTE: the sbatch parameter line below mirrors submit_wrapper's Leonardo/QOS
# handling; verify it on your first run. Everything else reuses the stock scripts.

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$HERE" || { echo "cannot cd to repo root $HERE" >&2; exit 1; }
# Pass the repo root to the batch body explicitly: under sbatch the script is
# copied to Slurm's spool dir, so BASH_SOURCE there does NOT point at the repo.
# --export=ALL propagates this into the job.
export TWILL_REPO="$HERE"

export TWILL_WINDOWS="${TWILL_WINDOWS:-2 8 32 128}"
export TWILL_CACHES="${TWILL_CACHES:-0 1}"
export TWILL_GROUPS="${TWILL_GROUPS:-4 8 16 32}"
export TWILL_ONLY="${TWILL_ONLY:-yes}"
export TWILL_SWEEP_CONFIG="${TWILL_SWEEP_CONFIG:-tests/cpu/twill_sweep_local.sh}"
BASE_CONFIG="${TWILL_BASE_CONFIG:-tests/cpu/twill_alltoall.sh}"

# Pull SLURM/run parameters from the base config (N_NODES, PARTITION, QOS, ...).
# shellcheck disable=SC1090
source "$BASE_CONFIG"
: "${PICO_ACCOUNT:?set PICO_ACCOUNT to your SLURM project before running the sweep}"

n=0
for w in $TWILL_WINDOWS; do for c in $TWILL_CACHES; do for g in $TWILL_GROUPS; do n=$((n + 1)); done; done; done
echo "TWILL single-allocation sweep: $n combo(s) on $N_NODES node(s) in ONE sbatch job"
echo "  windows=[$TWILL_WINDOWS]  caches=[$TWILL_CACHES]  groups=[$TWILL_GROUPS]  (TWILL_ONLY=$TWILL_ONLY)"

# Compile once on the login node (cached no-op inside the allocation afterwards),
# so we don't burn allocation time on the first build. Reuses submit_wrapper.
# Route through the LOCAL config (LOCATION=local => the sbatch branch is
# unreachable) and gate compile-only on TWILL_COMPILE_ONLY (submit_wrapper resets
# COMPILE_ONLY before sourcing the config, so a cmdline COMPILE_ONLY=yes would be
# lost and it would otherwise submit a full benchmark job).
echo "Compiling (once) ..."
TWILL_COMPILE_ONLY=yes bash scripts/submit_wrapper.sh --file "$TWILL_SWEEP_CONFIG" || { echo "compile failed" >&2; exit 1; }

# Build the sbatch parameters (mirrors submit_wrapper's SLURM_PARAMS for the
# Leonardo/QOS case). Adjust here if your site differs. --export=ALL carries the
# grid env (TWILL_WINDOWS/CACHES/GROUPS, TWILL_SWEEP_CONFIG, PICO_ACCOUNT, ...)
# into the job so tests/cpu/twill_sweep_inner.sh sees the same values.
SBATCH=( --account="$PICO_ACCOUNT" --nodes="$N_NODES" --time="$TEST_TIME" --partition="$PARTITION" --exclusive
         --job-name=twill_sweep --export=ALL
         --output="$HERE/twill_sweep_%j.out" --error="$HERE/twill_sweep_%j.err" )
[[ -n "${QOS:-}" ]]      && SBATCH+=( --qos="$QOS" )
[[ -n "${QOS_GRES:-}" ]] && SBATCH+=( --gres="$QOS_GRES" )
_tpn="${QOS_TASKS_PER_NODE:-${LIB_0_TASKS_PER_NODE:-}}"
[[ -n "$_tpn" ]]         && SBATCH+=( --ntasks-per-node="$_tpn" )
[[ -n "${OTHER_SLURM_PARAMS:-}" ]] && SBATCH+=( ${OTHER_SLURM_PARAMS} )

echo "sbatch ${SBATCH[*]}"
echo "  -> ONE job; tests/cpu/twill_sweep_inner.sh iterates the $n combo(s) as srun steps inside it"
exec sbatch "${SBATCH[@]}" "$HERE/tests/cpu/twill_sweep_inner.sh"
