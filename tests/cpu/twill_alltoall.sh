#!/bin/bash
# TWILL alltoall evaluation (TUI run-config for scripts/submit_wrapper.sh --file).
#
#   bash scripts/submit_wrapper.sh --file tests/cpu/twill_alltoall.sh
#
# Headline metric: completion time vs message size for the twill variants against
# all Open MPI vendor alltoall algorithms and the libpico pairwise reference.
# Modeled on tests/cpu/all_512.sh; adjust N_NODES / LIB_0_TASKS_PER_NODE for your
# allocation.
#
# Group map: TWILL_GROUP=node (default) uses node=group. For a real dragonfly
# group map, generate one and export TWILL_MAP (see utils/twill_gen_map.py),
# e.g.:  TWILL_MAP=twill.map bash scripts/submit_wrapper.sh --file tests/cpu/twill_alltoall.sh
#
# A window/cache/group sweep is driven by tests/cpu/twill_sweep.sh, which grabs
# ONE allocation (salloc) and iterates the grid inside it (same nodes =>
# comparable), reusing the stock submit_wrapper/orchestrator unmodified. All
# TWILL_* knobs below respect the ambient environment so the sweep can set them
# per grid point.

# Run-control knobs respect the ambient env so the sweep (tests/cpu/twill_sweep.sh)
# can reuse this config inside its salloc (LOCATION=local, RUN=srun) and pre-compile.
export LOCATION="${LOCATION:-leonardo}"
export RUN="${RUN:-srun}"
export UCX_IB_SL=1
export PARTITION="boost_usr_prod"
export QOS="qos_special"
export QOS_TASKS_PER_NODE=32
export QOS_GRES="gpu:4"
export GENERAL_MODULES="python/3.11.7"
export COMPILE_ONLY="${COMPILE_ONLY:-no}"
# submit_wrapper.sh resets COMPILE_ONLY to its default BEFORE sourcing this file,
# so a cmdline `COMPILE_ONLY=yes` is lost. Honor a TWILL_* var (never reset) so the
# sweep's pre-compile step can compile-and-exit instead of submitting a real job.
[[ "${TWILL_COMPILE_ONLY:-no}" == "yes" ]] && export COMPILE_ONLY=yes
export DEBUG_MODE="${DEBUG_MODE:-no}"
export DRY_RUN="${DRY_RUN:-no}"
export DELETE="yes"
export COMPRESS="yes"
export N_NODES="${N_NODES:-16}"
export OUTPUT_LEVEL="all"
export TEST_TIME="${TEST_TIME:-02:00:00}"
export TYPES="int32"
# Total element counts (per-pair count = SIZE / comm_sz). Small sizes below
# comm_sz are skipped automatically for count>=comm_sz algorithms (see SKIP).
export SIZES="8,64,512,4096,32768,262144,2097152,16777216,134217728"
export SEGMENT_SIZES="0"

# ---- TWILL knobs (forwarded to all ranks; override-able by the sweep wrapper) ----
export TWILL_WINDOW="${TWILL_WINDOW:-32}"
# TWILL_GROUP: "node" (node=group) OR an integer N for synthetic uniform groups of
# size N (rank i -> group i/N), set manually and independent of tasks-per-node,
# e.g. TWILL_GROUP=7. A TWILL_MAP file (below) overrides this.
export TWILL_GROUP="${TWILL_GROUP:-node}"
export TWILL_CACHE="${TWILL_CACHE:-1}"
# export TWILL_MAP="${TWILL_MAP:-/path/to/twill.map}"   # uncomment for a real map

# Record the twill config in each run's metadata (NOTES column of
# results/<system>_metadata.csv). For a single (non-sweep) run this tags it with
# the knob values; in a sweep, tests/cpu/twill_sweep_local.sh overrides NOTES per combo.
# Set here because the TUI file is sourced after submit_wrapper's NOTES default.
export NOTES="twill W=${TWILL_WINDOW} group=${TWILL_GROUP} cache=${TWILL_CACHE}"
# export TWILL_SEED="${TWILL_SEED:-0}"                  # twill_random determinism

export LIB_COUNT=1
export LIB_0_NAME="Open MPI 4.1.6"
export LIB_0_VERSION="4.1.6"
export LIB_0_STANDARD="MPI"
export LIB_0_MPI_LIB="OMPI"
export LIB_0_PICOCC="mpicc"
export LIB_0_MPI_LIB_VERSION="4.1.6"
export LIB_0_TASKS_PER_NODE="4"
export LIB_0_LOAD_TYPE="module"
export LIB_0_MODULES="openmpi/4.1.6--gcc--12.2.0"
export LIB_0_COLLECTIVES="alltoall"
# Algorithm set. Default is the full field: all Open MPI vendor algorithms
# (default, linear, pairwise, modified bruck, linear_sync) + the libpico "over"
# algorithms (bine_over, pairwise_ompi_over) + the three twill variants.
# TWILL_ONLY=yes restricts to just the twill variants -- handy for the sweep,
# since the vendor/bine baselines are invariant to the TWILL_* knobs and only
# need measuring once.
# Notes: two_proc_ompi is omitted (hardwired for exactly 2 ranks; would be
# incorrect at any other comm size). bine_over needs a power-of-two comm size and
# pairwise_ompi needs an even comm size -- both fine at 64 ranks; revisit if you
# change the rank count.
if [[ "${TWILL_ONLY:-no}" == "yes" ]]; then
  _TWILL_ALGS="twill_shift_over,twill_random_over,twill_group_over"
else
  _TWILL_ALGS="default_ompi,linear_ompi,pairwise_ompi,modified_bruck_ompi,linear_sync_ompi,bine_over,pairwise_ompi_over,twill_shift_over,twill_random_over,twill_group_over"
fi
export LIB_0_ALLTOALL_ALGORITHMS="$_TWILL_ALGS"
# Every alltoall algorithm needs count >= comm_sz (the buffer splits into comm_sz
# per-pair blocks), so SKIP mirrors the list (sizes below comm_sz are dropped --
# with N_NODES=16 x 4 = 64 ranks, that is just the size=8 entry). SKIP and
# IS_SEGMENTED are derived from the list so their lengths always match it.
export LIB_0_ALLTOALL_ALGORITHMS_SKIP="$_TWILL_ALGS"
export LIB_0_ALLTOALL_ALGORITHMS_IS_SEGMENTED="$(printf '%s' "$_TWILL_ALGS" | sed -E 's/[^,]+/no/g')"
