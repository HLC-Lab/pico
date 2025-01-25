#!/bin/bash

# Function to handle cleanup on script termination
# (e.g., Ctrl+C)
cleanup() {
    echo "Caught Ctrl+C! Stopping the script and \
          killing all child processes..."
    pkill -P $$
    exit 1
}

# Trap SIGINT (Ctrl+C) and call cleanup function
trap cleanup SIGINT

# Validate and initialize N_NODES
N_NODES=$1
if [[ -z "$N_NODES" ]] || [[ ! "$N_NODES" =~ ^[0-9]+$ ]] \
                       || [ "$N_NODES" -lt 2 ]; then
    echo "Error: N_NODES is not given or not set correctly. \
          Please provide a valid number of nodes as FIRST argument."
    exit 1
fi

# Default values for other parameters if not provided as arguments
TIMESTAMP=${2:-$(date +"%Y_%m_%d___%H:%M:%S")}
LOCATION=${3:-local}
COLLECTIVE_TO_TEST=${4:-ALLREDUCE}
export COLLECTIVE_TYPE=$COLLECTIVE_TO_TEST
ENABLE_CUDA=${5:-no}
ENABLE_OMPI_TEST=${6:-yes}

ALGOS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
ARR_SIZES=(8 64 512 2048 16384 131072 1048576 8388608 67108864)
TYPES=('int64')                          # Data types to test
# Algorithms to skip if $N_NODES > $ARR_SIZE
SKIP=(4 5 6 9 10 11 12 13 16)

# Uncomment for all types 
# NOTE: problems with char, int8, int16
# TYPES=('int32' 'int64' 'float' 'double' 'char' 'int8' 'int16')

# Remove all ompi_test algorithms from algo list
# if `ENABLE_OMPI_TEST` is set to `no`
if [ "$ENABLE_OMPI_TEST" == "no" ]; then
    ALGOS=("${ALGOS[@]/8}")
    ALGOS=("${ALGOS[@]/9}")
    ALGOS=("${ALGOS[@]/10}")
    ALGOS=("${ALGOS[@]/11}")
    ALGOS=("${ALGOS[@]/12}")
    ALGOS=("${ALGOS[@]/13}")
fi

# Load the environment-specific configuration based on the LOCATION
if [ -f scripts/environments/${LOCATION}.sh ]; then
    # FIX: do the MPI module load if OMPI_TEST is not used
    source scripts/environments/${LOCATION}.sh
    RES_DIR=$SWING_DIR/results/
    TEST_EXEC=$SWING_DIR/bin/test
    RULE_UPDATER_EXEC=$SWING_DIR/bin/change_collective_rules
    RULE_FILE_PATH=$SWING_DIR/ompi_rules/collective_rules.txt

    export OMPI_MCA_coll_hcoll_enable=0
    export OMPI_MCA_coll_tuned_use_dynamic_rules=1
else
    echo "ERROR: Environment script for '${LOCATION}' system not found!"
    exit 1
fi


# Output directories for results
OUTPUT_DIR="$RES_DIR/$LOCATION/$TIMESTAMP/"
DATA_DIR="$OUTPUT_DIR/data/"

# Function to determine the number of iterations
# based on array size
get_iterations() {
    size=$1
    if [ $size -le 512 ]; then
        echo 20000
    elif [ $size -le 1048576 ]; then
        echo 2000
    elif [ $size -le 8388608 ]; then
        echo 200
    elif [ $size -le 67108864 ]; then
        echo 20
    else
        echo 5
    fi
}

# Function to run a single test case
# Arguments: array size, iterations, data type, algorithm index
run_test() {
    local size=$1
    local iter=$2
    local type=$3
    local algo=$4

    echo "Running -> $N_NODES processes, $size array size, $type datatype (Algo: $algo)"
    $RUN $RUNFLAGS -n $N_NODES $TEST_EXEC $size $iter $type $algo $OUTPUT_DIR
}

# Create necessary output directories
mkdir -p "$RES_DIR"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$DATA_DIR"

# Clean and compile the codebase.
# if `ENABLE_OMPI_TEST` is `yes`, the makefile
# will inject it into `tests\...` files.
make clean
if [ "$ENABLE_OMPI_TEST" == "yes" ]; then
    make all OMPI_TEST=1
else
    make all
fi

# Test algorithms here, loop through:
# - algorithms:
#     - 8 swing latency
#     - 9 swing bandwidt memcp
#     - 10 swing bandwidth datatype
#     - 11 swing bandwidth datatype + memcp
#     - 12 swing bandwidth segmented
#     - 13 swing bandwidth static
#     - 14 swing latency OVER MPI
#     - 15 recursive doubling OVER MPI
#     - 16 swing bandwidth static OVER MPI
# - number of mpi processes
# - size of the array in number of elements (of type = TYPE) 
#
# note that algo 14, 15 and 16 are not defined inside Open MPI so
# rule file will be set to 0 (i.e. automatic default algorithm selection)
for algo in ${ALGOS[@]}; do
    # Update dynamic rules for the current algorithm
    $RULE_UPDATER_EXEC $RULE_FILE_PATH $algo
    export OMPI_MCA_coll_tuned_dynamic_rules_filename=${RULE_FILE_PATH}
    for size in "${ARR_SIZES[@]}"; do
        # Skip specific algorithms for certain conditions
        if [[ size -lt $N_NODES && " ${SKIP[@]} " =~ " ${algo} " ]]; then
            echo "Skipping algorithm $algo: is in SKIP and size=$size < N_NODES=$N_NODES"
            continue
        fi

        # Get the number of iterations for the given array size
        iter=$(get_iterations $size)
        for type in "${TYPES[@]}"; do
            run_test $size $iter $type $algo
        done
    done
done

# # FIX: Ensure python venv and modules are correctly set up
#
# # TODO: this is only a proof of concept,
# # to be modified later
#
# MPI_TYPE=OpenMPI
# MPI_VERSION=0
# LIBSWING_VERSION=0
# OPERATOR=MPI_Sum
# OTHER=None
#
# python $RES_DIR/update_metadata.py "$LOCATION" "$TIMESTAMP" \
#       "$N_NODES" "$COLLECTIVE_TO_TEST" "$ALGOS" "$MPI_TYPE" "$MPI_VERSION" \
#       "$LIBSWING_VERSION" "$ENABLE_CUDA" "$TYPES" "$OPERATOR" "$OTHER"
