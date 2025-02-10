#!/bin/bash

source scripts/utils.sh

# Trap SIGINT (Ctrl+C) and call cleanup function
trap cleanup SIGINT

# Validate and initialize N_NODES
N_NODES=$1
if [[ -z "$N_NODES" ]] || [[ ! "$N_NODES" =~ ^[0-9]+$ ]] || [ "$N_NODES" -lt 2 ]; then
    error "N_NODES is not given or not set correctly."
    exit 1
fi

# Sanity checks
success "==========================================================\n\t\t SANITY CHECKS"
echo "Running tests in: $LOCATION"
echo "Debug mode: $DEBUG_MODE"
echo "Number of nodes: $N_NODES"
echo "Saving results in: $OUTPUT_DIR"
echo "Running benchmarks for collective: $COLLECTIVE_TYPE"
echo -e "For algorithms: \n $ALGOS"
echo -e "With sizes: \n $ARR_SIZES"
echo -e "And data types: \n $TYPES"
echo "MPI Library: $MPI_LIB, $MPI_LIB_VERSION"
echo "Libswing Version: $LIBSWING_VERSION"
echo "CUDA Enabled: $CUDA"
echo "NOTES: $NOTES"
success "=========================================================="

#Run tests for all configurations
run_all_tests || exit 1

if [ $DEBUG_MODE == "no" ]; then
    $SWING_DIR/results/compress_results.sh
fi
