# Colors for styling output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

# Print error messages in red
error() {
    echo -e "\n${RED}❌❌❌ ERROR: $1 ❌❌❌${NC}\n" >&2
}

# Print success messages in green
success() {
    echo -e "\n${GREEN}$1${NC}\n"
}

# Print warning messages in yellow
warning() {
    echo -e "\n${YELLOW}WARNING: $1 ${NC}\n"
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

# Activate the virtual environment, if it exists
# If not create it and install the required Python packages
activate_virtualenv() {
    if [ -f "$HOME/.swing_venv/bin/activate" ]; then
        source "$HOME/.swing_venv/bin/activate" || { error "Failed to activate virtual environment." ; return 1; }
    else
        warning "Virtual environment 'swing_venv' does not exist. Creating it..."
        
        # Create the virtual environment
        python3 -m venv "$HOME/.swing_venv" || { error "Failed to create virtual environment." ; return 1; }
        source "$HOME/.swing_venv/bin/activate" || { error "Failed to activate virtual environment after creation." ; return 1; }

        success "Virtual environment 'swing_venv' created and activated."

        pip install --upgrade pip || { error "Failed to upgrade pip." ; return 1; }
        pip install jsonschema || { error "Failed to install Python packages." ; return 1; }

        success "Python packages installed."
    fi

    return 0
}

# Compile the codebase
compile_code() {
    make clean -s
    make_command="make all -s"
    [ "$DEBUG_MODE" == "yes" ] && make_command="$make_command DEBUG=1"

    if ! $make_command; then
        error "Compilation failed. Exiting."
        return 1
    fi

    success "Compilation succeeded."
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


# Function to run a single test case
# Arguments: array size, iterations, data type, algorithm index
run_test() {
    local size=$1
    local type=$2
    local algo=$3
    local iter=$(get_iterations $size)

    if [ "$DEBUG_MODE" == "yes" ]; then
        echo "DEBUG: $COLLECTIVE_TYPE -> $N_NODES processes, $size array size, $type datatype ($algo)"
    else
        echo "Benchmarking $COLLECTIVE_TYPE -> $N_NODES processes, $size array size, $type datatype ($algo. Iter: $iter)"
    fi

    $RUN $RUNFLAGS -n $N_NODES $TEST_EXEC $size $iter $algo $type
}

# Test algorithms here, loop through:
# - algorithms
# - number of mpi processes
# - size of the array in number of elements (of type = TYPE) 
#
# note that, if an algorithm is not internal to Open MPI, the dynamic
# rule file will be set to 0 (i.e. automatic default algorithm selection)
run_all_tests() {
    for algo in ${ALGOS[@]}; do
        # Update dynamic rule file for the algorithm
        if [[ "$MPI_LIB" == "OMPI_SWING" ]] || [[ "$MPI_LIB" == "OMPI" ]]; then
            echo "Updating dynamic rule file for algorithm $algo..."
            python3 $RULE_UPDATER_EXEC $algo || exit 1
            export OMPI_MCA_coll_tuned_dynamic_rules_filename=${DYNAMIC_RULE_FILE}
        fi

        for size in ${ARR_SIZES[@]}; do
            # Skip specific algorithms if conditions are met
            if [[ size -lt $N_NODES && " ${SKIP} " =~ " ${algo} " ]]; then
                echo "Skipping algorithm $algo for size=$size < N_NODES=$N_NODES"
                continue
            fi

            # Get the number of iterations for the size
            for type in ${TYPES[@]}; do
                # Run the test for the given configuration
                run_test $size $type $algo
            done
        done
    done
}
