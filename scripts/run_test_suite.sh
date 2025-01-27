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
TIMESTAMP=${3:-$(date +"%Y_%m_%d___%H:%M:%S")}  # Timestamp for result directory
LOCATION=${4:-local}                            # Environment location
ENABLE_CUDA=${5:-no}                            # Enable CUDA support (yes/no)
ENABLE_OMPI_TEST=${6:-yes}                      # Enable OpenMPI tests (yes/no)
DEBUG_MODE=${7:-no}                             # Enable debug mode (yes/no)

# Define supported algorithms for each collective type
declare -A COLLECTIVE_ALGOS
COLLECTIVE_ALGOS[ALLREDUCE]="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16"
COLLECTIVE_ALGOS[ALLGATHER]="0 2 3 4 5 8 9 10"

# Modify algorithms if OMPI_TEST (open mpi with swing allreduce) is not used
if [ "$ENABLE_OMPI_TEST" == "no" ]; then
    COLLECTIVE_ALGOS[ALLREDUCE]="0 1 2 3 4 5 6 7 14 15 16"
fi

# Algorithms to skip if $N_NODES > $ARR_SIZE
declare -A COLLECTIVE_SKIPS
COLLECTIVE_SKIPS[ALLREDUCE]="4 5 6 9 10 11 12 13 16"
COLLECTIVE_SKIPS[ALLGATHER]=""

ALGOS=${COLLECTIVE_ALGOS[$COLLECTIVE_TYPE]}
SKIP=${COLLECTIVE_SKIPS[$COLLECTIVE_TYPE]}

# Define array sizes for testing
ARR_SIZES=(8 64 512 2048 16384 131072 1048576 8388608 67108864)
if [ "$DEBUG_MODE" == yes ]; then
    ARR_SIZES=(8 64 512)                             # Smaller sizes for debug mode
fi

TYPES=('int64')                          # Data types to test
# TYPES=('int32' 'int64' 'float' 'double' 'char' 'int8' 'int16') # Uncomment for all types 
# NOTE: problems with char, int8, int16



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
run_all_tests "$N_NODES" "$ALGOS" "$ARR_SIZES" "$TYPES" "$OUTPUT_DIR"


# # TODO: this is only a proof of concept,
# # to be modified later
#
# # FIX: Ensure python venv and modules are correctly set up
#
# MPI_TYPE=OpenMPI
# MPI_VERSION=0
# LIBSWING_VERSION=0
# OPERATOR=MPI_Sum
# OTHER=None
#
# python $RES_DIR/update_metadata.py "$LOCATION" "$TIMESTAMP" \
#       "$N_NODES" "$COLLECTIVE_TYPE" "$ALGOS" "$MPI_TYPE" "$MPI_VERSION" \
#       "$LIBSWING_VERSION" "$ENABLE_CUDA" "$TYPES" "$OPERATOR" "$OTHER"
