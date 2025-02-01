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

# Default parameters
DEBUG_MODE=${2:-no}                             # Enable debug mode (yes/no)
TIMESTAMP=${3:-$(date +"%Y_%m_%d___%H:%M:%S")}  # Timestamp for result directory
LOCATION=${4:-local}                            # Environment location
ENABLE_CUDA=${5:-no}                            # Enable CUDA support (yes/no)


# Run the Python script to set environment variables
python3 scripts/select_test/parse_test.py $N_NODES 
source scripts/select_test/env_vars.sh

# Use the environment variables
echo "Running benchmarks for collective: $COLLECTIVE_TYPE"
echo "Algorithms to run: $ALGO"
echo "Algorithm names: $NAMES"
echo "Algorithms to skip: $SKIP"
echo "MPI Library: $MPI_LIB, Version: $MPI_LIB_VERSION"
echo "Libswing Version: $LIBSWING_VERSION"

# declare -A SIZES
# SIZES[ALLREDUCE]="8 64 512" # 2048 16384 131072 1048576 8388608 67108864"
# SIZES[ALLGATHER]="8 64 512 2048 16384 131072 1048576 8388608 67108864"
# SIZES[REDUCE_SCATTER]="8 64 512 2048 16384 131072 1048576 8388608 67108864"
#
#
# ARR_SIZES=${SIZES[$COLLECTIVE_TYPE]}
#
# # Supported types are "int8 int16 int32 int64 int float double char unsigned_char"
# TYPES="int64"
#
# # Select here what to do in debug mode
# if [ "$DEBUG_MODE" == yes ]; then
#     ALGOS="102"
#     ARR_SIZES="8"
#     TYPES="int" # For now only int,int32,int64 are supported in debug mode 
# fi
#
# # Load configuration for the specified environment
# if ! source_environment "$LOCATION"; then
#     error "Environment script for '${LOCATION}' not found!"
#     exit 1
# else 
#     success "Environment script for '${LOCATION}' loaded successfully."
#
#     # Define paths and output directories
#     RES_DIR=$SWING_DIR/results/
#     TEST_EXEC=$SWING_DIR/bin/test
#     RULE_UPDATER_EXEC=$SWING_DIR/bin/change_dynamic_rules
#     RULE_FILE_PATH=$SWING_DIR/ompi_rules/dynamic_rules.txt
#     OUTPUT_DIR="$RES_DIR/$LOCATION/$TIMESTAMP/"
#     DATA_DIR="$OUTPUT_DIR/data/"
#     ALGO_NAMES_FILE=$SWING_DIR/scripts/algo_names.csv
#
#     # Set OpenMPI environment variables
#     export OMPI_MCA_coll_hcoll_enable=0
#     export OMPI_MCA_coll_tuned_use_dynamic_rules=1
# fi
#
# # Clean and compile the codebase
# compile_code || exit 1
#
# # Create necessary output directories
# if [ $DEBUG_MODE == no ]; then
#     success "📂 Creating output directories..."
#     mkdir -p "$RES_DIR"
#     mkdir -p "$OUTPUT_DIR"
#     mkdir -p "$DATA_DIR"
# fi
#
#
# # Run tests for all configurations
# run_all_tests "$N_NODES" "$ALGOS" "$SKIP" "$ARR_SIZES" "$TYPES" "$OUTPUT_DIR" "$DEBUG_MODE"
#
#
# if [ "$DEBUG_MODE" == no ]; then
#     # TODO: this is only a proof of concept, to be modified later
#     # FIX: Ensure python venv and modules are correctly set up
#     OPERATOR=MPI_SUM
#     OTHER=none
#
#     python $RES_DIR/generate_metadata.py "$LOCATION" "$TIMESTAMP" \
#           "$N_NODES" "$COLLECTIVE_TYPE" "$ALGOS" "$NAMES" \
#           "$MPI_LIB" "$MPI_LIB_VERSION" "$LIBSWING_VERSION" "$ENABLE_CUDA" \
#           "$TYPES" "$OPERATOR" "$OTHER"
#
# fi
