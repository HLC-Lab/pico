#!/bin/bash

# Colors for styling output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

# Print error messages in red
error() {
    echo -e "\n${RED}ERROR: $1${NC}\n"
}

# Print success messages in green
success() {
    echo -e "\n${GREEN}$1${NC}\n"
}

# Cleanup function for SIGINT
cleanup() {
    error "Caught Ctrl+C! Killing all child processes..."
    pkill -P $$
    exit 1
}

# Source the environment configuration
source_environment() {
    local env_file="scripts/environments/$1.sh"
    if [ -f "$env_file" ]; then
        source "$env_file"
        return 0
    else
        return 1
    fi
}


# Compile the codebase
compile_code() {
    make clean
    make_command="make all"
    [ "$ENABLE_OMPI_TEST" == "yes" ] && make_command="$make_command OMPI_TEST=1"
    [ "$DEBUG_MODE" == "yes" ] && make_command="$make_command DEBUG=1"

    if ! $make_command; then
        error "Compilation failed. Exiting."
        return 1
    fi

    success "✅ Compilation succeeded."
    return 0
}

# Determine the number of iterations based on array size
get_iterations() {
    local size=$1
    if [ "$DEBUG_MODE" == "yes" ]; then
        echo 1
    elif [ $size -le 512 ]; then
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

# Function to get all algorithm names for a given test
get_all_algorithm_names() {
    local algo_numbers=($1)
    local algo_names=""

    for algo in "${algo_numbers[@]}"; do
        algo_name=$(awk -F, -v type="$COLLECTIVE_TYPE" -v algo="$algo" \
            '$1 == type && $2 == algo {print $3}' "$ALGO_NAMES_FILE")
        algo_names+="$algo_name "
    done

    echo "$algo_names"
}

# Function to get the algorithm name for a given number
get_algorithm_name() {
    local algo_number=$1
    awk -F, -v type="$COLLECTIVE_TYPE" -v algo="$algo_number" \
        '$1 == type && $2 == algo {print $3}' "$ALGO_NAMES_FILE"
}

# Function to run a single test case
# Arguments: array size, iterations, data type, algorithm index
run_test() {
    local size=$1
    local iter=$2
    local type=$3
    local algo=$4
    local debug_mode=$5

    local algo_name=$(get_algorithm_name "$algo")

    if [ "$debug_mode" == "yes" ]; then
      echo "DEBUG: $COLLECTIVE_TYPE -> $N_NODES processes, $size array size, $type datatype ($algo: $algo_name)"
    else
      echo "Benchmarking $COLLECTIVE_TYPE -> $N_NODES processes, $size array size, $type datatype ($algo: $algo_name Iter: $iter)"
    fi

    $RUN $RUNFLAGS -n $N_NODES $TEST_EXEC $size $iter $type $algo $OUTPUT_DIR
}

# Test algorithms here, loop through:
# - algorithms
# - number of mpi processes
# - size of the array in number of elements (of type = TYPE) 
#
# note that, if an algorithm is not internal to Open MPI, the dynamic
# rule file will be set to 0 (i.e. automatic default algorithm selection)
run_all_tests() {
    local nodes=$1
    local algos=($2)
    local sizes=($3)
    local types=($4)
    local output_dir=$5
    local debug_mode=$6

    for algo in ${algos[@]}; do
        # Update dynamic rule file for the algorithm
        $RULE_UPDATER_EXEC $RULE_FILE_PATH $algo
        export OMPI_MCA_coll_tuned_dynamic_rules_filename=${RULE_FILE_PATH}

        for size in ${sizes[@]}; do
            # Skip specific algorithms if conditions are met
            if [[ size -lt $nodes && " ${COLLECTIVE_SKIPS[$COLLECTIVE_TYPE]} " =~ " ${algo} " ]]; then
                echo "Skipping algorithm $algo for size=$size < N_NODES=$nodes"
                continue
            fi

            # Get the number of iterations for the size
            local iter=$(get_iterations $size)
            for type in ${types[@]}; do
                # Run the test for the given configuration
                run_test $size $iter $type $algo $debug_mode
            done
        done
    done
}
