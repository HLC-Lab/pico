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
export COLLECTIVE_TYPE=${2:-ALLREDUCE}          # Type of collective operation
DEBUG_MODE=${3:-no}                             # Enable debug mode (yes/no)
TIMESTAMP=${4:-$(date +"%Y_%m_%d___%H:%M:%S")}  # Timestamp for result directory
LOCATION=${5:-local}                            # Environment location
ENABLE_CUDA=${6:-no}                            # Enable CUDA support (yes/no)
ENABLE_OMPI_TEST=${7:-yes}                      # Enable OpenMPI tests (yes/no)

# Define supported algorithms for each collective type
# WARNING:
# ALLREDUCE_ALLGATHER_REDUCE (7) not included since can crash, wasting compute hours. 
# ALLGATHER_K_BRUCK (2) does not work.
# ALLGATHER_K_BRUCK_OVER (7) logic not implemented: requires an additional parameter.
# ALLGATHER_TWO_PROC (6) not included since it is only for 2 processes.
declare -A COLLECTIVE_ALGOS
COLLECTIVE_ALGOS[ALLREDUCE]="0 1 2 3 4 5 6 8 9 10 11 12 13 14 15 16"
COLLECTIVE_ALGOS[ALLGATHER]="0 1 3 4 5 8 9 10"
COLLECTIVE_ALGOS[REDUCE_SCATTER]="0 1 2 3 4 5 6 7"

# Modify algorithms if OMPI_TEST (open mpi with swing allreduce) is not used
if [ "$ENABLE_OMPI_TEST" == "no" ]; then
    COLLECTIVE_ALGOS[ALLREDUCE]="0 1 2 3 4 5 6 14 15 16"
fi

# Algorithms to skip if $N_NODES > $ARR_SIZE
declare -A COLLECTIVE_SKIPS
COLLECTIVE_SKIPS[ALLREDUCE]="4 5 6 9 10 11 12 13 16"
COLLECTIVE_SKIPS[ALLGATHER]=""
COLLECTIVE_SKIPS[REDUCE_SCATTER]=""

declare -A SIZES
SIZES[ALLREDUCE]="8 64 512 2048 16384 131072 1048576 8388608 67108864"
SIZES[ALLGATHER]="8 64 512 2048 16384 131072 1048576 8388608 67108864"
SIZES[REDUCE_SCATTER]="8 64 512 2048 16384 131072 1048576 8388608 67108864"


ALGOS=${COLLECTIVE_ALGOS[$COLLECTIVE_TYPE]}
SKIP=${COLLECTIVE_SKIPS[$COLLECTIVE_TYPE]}
ARR_SIZES=${SIZES[$COLLECTIVE_TYPE]}
# Supported types are "int8 int16 int32 int64 int float double char unsigned_char"
TYPES="int64 int32"

# Select here what to do in debug mode
if [ "$DEBUG_MODE" == yes ]; then
    ALGOS="10"
    ARR_SIZES="8"
    TYPES="int" # For now only int,int32,int64 are supported in debug mode 
fi

# Load configuration for the specified environment
if ! source_environment "$LOCATION"; then
    error "Environment script for '${LOCATION}' not found!"
    exit 1
else 
    success "Environment script for '${LOCATION}' loaded successfully."

    # Define paths and output directories
    RES_DIR=$SWING_DIR/results/
    TEST_EXEC=$SWING_DIR/bin/test
    RULE_UPDATER_EXEC=$SWING_DIR/bin/change_dynamic_rules
    RULE_FILE_PATH=$SWING_DIR/ompi_rules/dynamic_rules.txt
    OUTPUT_DIR="$RES_DIR/$LOCATION/$TIMESTAMP/"
    DATA_DIR="$OUTPUT_DIR/data/"

    # Set OpenMPI environment variables
    export OMPI_MCA_coll_hcoll_enable=0
    export OMPI_MCA_coll_tuned_use_dynamic_rules=1
fi

# Clean and compile the codebase
compile_code || exit 1

# Create necessary output directories
if [ $DEBUG_MODE == no ]; then
    success "📂 Creating output directories..."
    mkdir -p "$RES_DIR"
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$DATA_DIR"
fi


# Run tests for all configurations
run_all_tests "$N_NODES" "$ALGOS" "$ARR_SIZES" "$TYPES" "$OUTPUT_DIR" "$DEBUG_MODE"


# TODO: this is only a proof of concept,
# to be modified later
# FIX: Ensure python venv and modules are correctly set up
if [ "$DEBUG_MODE" == no ]; then
MPI_TYPE=OMPI_TEST
MPI_VERSION=5.0.1a1
LIBSWING_VERSION=0.0.1
OPERATOR=MPI_SUM
OTHER=none

python $RES_DIR/generate_metadata.py "$LOCATION" "$TIMESTAMP" \
      "$N_NODES" "$COLLECTIVE_TYPE" "$ALGOS" "$MPI_TYPE" "$MPI_VERSION" \
      "$LIBSWING_VERSION" "$ENABLE_CUDA" "$TYPES" "$OPERATOR" "$OTHER"

fi
