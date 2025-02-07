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
TEST_CONFIG=${3:-"scripts/config/test.json"}
TEST_ENV=${TEST_CONFIG}_env.sh
TIMESTAMP=${4:-$(date +"%Y_%m_%d___%H:%M:%S")}  # Timestamp for result directory
LOCATION=${5:-local}                            # Environment location
NOTES=${6:-""}                                  # Additional notes


if ! source_environment "$LOCATION"; then
    error "Environment script for '${LOCATION}' not found!"
    exit 1
else 
    success "Environment script for '${LOCATION}' loaded successfully."
fi

# Load the python module and the python venv with the correct packages installed
load_python

# Parse the test configuration file and source the test specific environment variables
python3 scripts/parse_test.py $TEST_CONFIG $TEST_ENV $N_NODES || exit 1
source $TEST_ENV
# Source the environment variables dependant on the MPI library
load_other_env_var

# Select here what to do in debug mode
if [ "$DEBUG_MODE" == yes ]; then
    ALGOS="default_ompi"
    ARR_SIZES="8"
    TYPES="int" # For now only int,int32,int64 are supported in debug mode 
fi

# Define paths and output directories
RES_DIR=$SWING_DIR/results/
TEST_EXEC=$SWING_DIR/bin/test

RULE_UPDATER=$SWING_DIR/ompi_rules/change_dynamic_rules.py
DYNAMIC_RULE_FILE=$SWING_DIR/ompi_rules/dynamic_rules.txt
ALGORITHM_CONFIG="$SWING_DIR/scripts/algorithm_config.json"

OUTPUT_DIR="$RES_DIR/$LOCATION/$TIMESTAMP/"
DATA_DIR="$OUTPUT_DIR/data/"

# Clean and compile the codebase
compile_code || exit 1

# Create necessary output directories
if [ $DEBUG_MODE == no ]; then
    success "📂 Creating output directories..."
    mkdir -p "$RES_DIR"
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$DATA_DIR"
    
    # Generate test metadata
    python3 results/generate_metadata.py "$LOCATION" "$TIMESTAMP" "$N_NODES" "${NOTES[@]}" || exit 1
fi

# Sanity checks
success "==========================================================\n\t\t SANITY CHECKS"
echo "Saving results in: $OUTPUT_DIR"
echo "Running benchmarks for collective: $COLLECTIVE_TYPE"
echo -e "For algorithms: \n $ALGOS"
echo -e "With sizes: \n $ARR_SIZES"
echo -e "And data types: \n $TYPES"
echo "MPI Library: $MPI_LIB, $MPI_LIB_VERSION"
echo "Libswing Version: $LIBSWING_VERSION"
echo "CUDA Enabled: $CUDA"
echo "NOTES: ${NOTES[@]}"
success "=========================================================="

# Run tests for all configurations
run_all_tests "$N_NODES" "$ALGOS" "$SKIP" "$ARR_SIZES" "$TYPES" "$OUTPUT_DIR" "$DEBUG_MODE" || exit 1

# Compress the results and add uncompressed to gitignore
chmod +x $RES_DIR/compress_results.sh
$RES_DIR/compress_results.sh
