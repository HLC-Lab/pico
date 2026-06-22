#!/bin/bash
# Per-combo config used by tests/cpu/twill_sweep.sh INSIDE its salloc.
#
# It reuses tests/cpu/twill_alltoall.sh (so the algorithm list, sizes, node count,
# etc. are defined in one place) but flips LOCATION to "local". That makes the
# stock submit_wrapper run the stock orchestrator *in-process* (no nested sbatch)
# while RUN=srun launches each rank onto the surrounding salloc allocation -- so
# every combo runs on the SAME nodes. submit_wrapper + orchestrator are unchanged.
#
# The (window,cache,group) knobs arrive as ambient env from twill_sweep_inner.sh;
# here we just stamp them into a per-combo results dir name (TIMESTAMP) and the
# metadata NOTES so utils/twill_compare.py can tell the combos apart.

# Base config (defines LIB_0_*, SIZES, N_NODES, QOS/partition, the twill knobs...).
source "$(dirname "${BASH_SOURCE[0]}")/twill_alltoall.sh"

export LOCATION="local"            # -> submit_wrapper runs the orchestrator in-process
export RUN="${RUN:-srun}"          # srun lands on the salloc nodes (override to mpiexec for a laptop dry test)
export RUNFLAGS="${RUNFLAGS:-}"
export TWILL_ONLY="${TWILL_ONLY:-yes}"

# Each combo is a separate submit_wrapper call, so it already gets its own fresh
# results/local/<timestamp>/ dir (standard PICO naming). The combo is recorded in
# the metadata NOTES below -- utils/twill_compare.py labels runs by NOTES, not by
# the dir name. Combos run sequentially (each a full benchmark), so their
# second-granularity timestamps differ naturally.
export NOTES="twill W=${TWILL_WINDOW} group=${TWILL_GROUP} cache=${TWILL_CACHE}"
