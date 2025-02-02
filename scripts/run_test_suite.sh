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

if [ "$LOCATION" != "local" ]; then
    module load python
fi

# Run the Python script to set all test environment variables
python3 scripts/select_test/parse_test.py $N_NODES || exit 1
source scripts/select_test/test_env_vars.sh

# Set array sizes to test
ARR_SIZES="8 64 512 2048 16384 131072 1048576 8388608 67108864"
# Supported types are "int8 int16 int32 int64 int float double char unsigned_char"
TYPES="int64"

# Select here what to do in debug mode
if [ "$DEBUG_MODE" == yes ]; then
    ALGOS="102"
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

# Use the environment variables
echo "Running benchmarks for collective: $COLLECTIVE_TYPE"
echo "Algorithms to run: $ALGOS"
if [ "$DEBUG_MODE" == no ]; then
    echo "Algorithms to skip: $SKIP"
    echo "Algorithm names: $NAMES"
    echo "Output directory: $OUTPUT_DIR"
    echo "Notes for this test: $NOTES"
fi
echo "MPI Library: $MPI_LIB, Version: $MPI_LIB_VERSION"
echo "Libswing Version: $LIBSWING_VERSION"
echo "CUDA Enabled: $CUDA"

# Run tests for all configurations
run_all_tests "$N_NODES" "$ALGOS" "$SKIP" "$ARR_SIZES" "$TYPES" "$OUTPUT_DIR" "$DEBUG_MODE"

# Generate test metadata
if [ "$DEBUG_MODE" == no ]; then
    python3 results/generate_metadata.py "$LOCATION" "$TIMESTAMP" \
          "$N_NODES" "$COLLECTIVE_TYPE" "${ALGOS[@]}" "${NAMES[@]}" \
          "$MPI_LIB" "$MPI_LIB_VERSION" "$LIBSWING_VERSION" "$CUDA" \
          "${TYPES[@]}" "$MPI_OP" "$NOTES"
fi
