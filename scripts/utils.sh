# Colors for styling output, otherwise utils needs to be sourced at every make
export RED='\033[0;31m'
export GREEN='\033[0;32m'
export YELLOW='\033[0;33m'
export BLUE='\033[1;34m'
export NC='\033[0m'

# Print error messages in red
error() {
    echo -e "\n${RED}❌❌❌ ERROR: $1 ❌❌❌${NC}\n" >&2
}
export -f error

# Print success messages in green
success() {
    echo -e "\n${GREEN}$1${NC}\n"
}
export -f success

# Print warning messages in yellow
warning() {
    echo -e "\n${YELLOW}WARNING: $1 ${NC}\n"
}
export -f warning

# Cleanup function for SIGINT
cleanup() {
    error "Caught Ctrl+C! Killing all child processes..."
    pkill -P $$
    exit 1
}
export -f cleanup

usage() {
    echo "Usage: $0 --location <LOCATION> --nodes <N_NODES> [options...]"
    echo "Options:"
    echo "  --location          Location (required)"
    echo "  --nodes             Number of nodes (required, integer >=2)"
    echo "  --output-dir        Output dir of test (default: current date/time)"
    echo "  --types             Data types, comma separated [defaults to all]"
    echo "  --sizes             Array sizes, comma separated [defaults to all]"
    echo "  --test-config       Relative paths to config files, comma separated [default: 'config/test/*.json']"
    echo "  --interactive       Interactive mode (use salloc instead of sbatch) [default: no]"
    echo "  --debug-mode        Debug mode [default: no]"
    echo "  --notes             Notes [default: 'Def notes']"
    echo "  --task-per-node     Sbatch asks per node [default: 1]"
    echo "  --time              Sbatch time [default: 01:00:00]"
    echo "  --help              Show this help message"
}

# Parse the command line arguments
parse_cli_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --location)
                export LOCATION="$2"
                shift 2
                ;;
            --nodes)
                export N_NODES="$2"
                shift 2
                ;;
            --output-dir)
                export TIMESTAMP="$2"
                shift 2
                ;;
            --types)
                export TYPES_OVERRIDE="$2"
                shift 2
                ;;
            --sizes)
                export ARR_SIZES_OVERRIDE="$2"
                shift 2
                ;;
            --test-config)
                TEST_CONFIG_OVERRIDE="$2"
                shift 2
                ;;
            --interactive)
                INTERACTIVE="$2"
                shift 2
                ;;
            --debug-mode)
                export DEBUG_MODE="$2"
                shift 2
                ;;
            --notes)
                export NOTES="$2"
                shift 2
                ;;
            --task-per-node)
                export TASK_PER_NODE="$2"
                shift 2
                ;;
            --time)
                export TEST_TIME="$2"
                shift 2
                ;;
            --help)
                usage
                exit 0
                ;;
            *)
                error "Error: Unknown option $1" >&2
                usage
                exit 1
                ;;
        esac
    done
}

validate_args() {
    if [[ -z "$N_NODES" ]] || [[ ! "$N_NODES" =~ ^[0-9]+$ ]] || [ "$N_NODES" -lt 2 ]; then
        error "N_NODES must be a numeric value and at least 2."
        return 1
    elif [[ "$INTERACTIVE" != "yes" ]] && [[ "$INTERACTIVE" != "no" ]]; then
        error "INTERACTIVE must be either 'yes' or 'no'."
        return 1
    elif [[ "$DEBUG_MODE" != "yes" ]] && [[ "$DEBUG_MODE" != "no" ]]; then
        error "DEBUG_MODE must be either 'yes' or 'no'."
        return 1
    fi

    if [[ -n "$ARR_SIZES_OVERRIDE" ]]; then
        for size in ${ARR_SIZES_OVERRIDE//,/ }; do
            if [[ ! "$size" =~ ^[0-9]+$ ]]; then
                error "--sizes must be a comma-separated list of numeric values."
                return 1
            fi
        done
    fi

    if [[ -n "$TYPES_OVERRIDE" ]]; then
        for type in ${TYPES_OVERRIDE//,/ }; do
            if [[ ! "$type" =~ ^(int|int8|int16|int32|int64|float|double|char)$ ]]; then
                error " --types must be a comma-separated list. Allowed types: int, int8, int16, int32, int64, float, double, char"
                return 1
            fi
        done
    fi

    return 0
}

# Source the environment configuration
source_environment() {
    local env_file="config/environments/$1.sh"
    if [ -f "$env_file" ]; then
        source "$env_file"
        return 0
    else
        return 1
    fi
}

# Load the required modules
load_modules(){
    if [ -n "$MODULES" ]; then
        for module in ${MODULES//,/ }; do
            module load $module || { error "Failed to load module $module." ; return 1; }
        done
    fi

    return 0
}

# Activate the virtual environment, if it exists, if not create it
# Also checks and install the required Python packages
export required_python_packages="jsonschema packaging"
activate_virtualenv() {
    if [ -f "$HOME/.swing_venv/bin/activate" ]; then
        source "$HOME/.swing_venv/bin/activate" || { error "Failed to activate virtual environment." ; return 1; }
    else
        warning "Virtual environment 'swing_venv' does not exist. Creating it..."
        
        # Create the virtual environment
        python3 -m venv "$HOME/.swing_venv" || { error "Failed to create virtual environment." ; return 1; }
        source "$HOME/.swing_venv/bin/activate" || { error "Failed to activate virtual environment after creation." ; return 1; }

        success "Virtual environment 'swing_venv' created and activated."
    fi

    # Check and install missing packages
    pip install --upgrade pip || { error "Failed to upgrade pip." ; return 1; }
    for package in $required_python_packages; do
        if ! pip show "$package" > /dev/null 2>&1; then
            warning "Package '$package' not found. Installing..."
            pip install "$package" || { error "Failed to install $package." ; return 1; }
        fi
        success "Packages already installed."
    done

    return 0
}

# Compile the codebase
compile_code() {
    if [ "$DEBUG_MODE" == "yes" ]; then
      make_command="make all DEBUG=1"
    else
      make_command="make all -s"
    fi

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
export -f get_iterations


# Function to run a single test case
# Arguments: array size, iterations, data type, algorithm index
run_bench() {
    local size=$1
    local type=$2
    local algo=$3
    local iter=$(get_iterations $size)

    if [ "$DEBUG_MODE" == "yes" ]; then
        echo "DEBUG: $COLLECTIVE_TYPE -> $N_NODES processes, $size array size, $type datatype ($algo)"
        $RUN $RUNFLAGS -n $N_NODES $BENCH_EXEC $size $iter $algo $type
    else
        echo "Benchmarking $COLLECTIVE_TYPE -> $N_NODES processes, $size array size, $type datatype ($algo. Iter: $iter)"
        $RUN $RUNFLAGS -n $N_NODES $BENCH_EXEC $size $iter $algo $type || { error "Failed to run bench for coll=$COLLECTIVE_TYPE, algo=$algo, size=$size, dtype=$type" ; exit 1; }
    fi
}
export -f run_bench

# Function to update the dynamic rule file for the given algorithm
update_algorithm() {
    local algo=$1
    local cvar_indx=$2
    if [[ "$MPI_LIB" == "OMPI_SWING" ]] || [[ "$MPI_LIB" == "OMPI" ]]; then
        echo "Updating dynamic rule file for algorithm $algo..."
        python3 $ALGO_CHANGE_SCRIPT $algo || exit 1
        export OMPI_MCA_coll_tuned_dynamic_rules_filename=${DYNAMIC_RULE_FILE}
    elif [[ $MPI_LIB == "MPICH" ]] || [[ $MPI_LIB == "CRAY_MPI" ]]; then
        local cvar=${CVARS[$cvar_indx]}
        echo "Setting CVAR $cvar for algorithm $algo..."
        export "MPIR_CVAR_${COLLECTIVE_TYPE}_INTRA_ALGORITHM"=$cvar
    fi
}
export -f update_algorithm

# Test algorithms here, loop through:
# - algorithms
# - number of mpi processes
# - size of the array in number of elements (of type = TYPE) 
#
# note that, if an algorithm is not internal to Open MPI, the dynamic
# rule file will be set to 0 (i.e. automatic default algorithm selection)
run_all_tests() {
    local i=0
    for algo in ${ALGOS[@]}; do
        # Update dynamic rule file for the algorithm
        update_algorithm $algo $i

        for size in ${ARR_SIZES[@]}; do
            # Skip specific algorithms if conditions are met
            if [[ $size -lt $N_NODES && " ${SKIP} " =~ " ${algo} " ]]; then
                echo "Skipping algorithm $algo for size=$size < N_NODES=$N_NODES"
                continue
            fi

            # Get the number of iterations for the size
            for type in ${TYPES[@]}; do
                # Run the bench for the given configuration
                run_bench $size $type $algo
            done
        done
        ((i++))
    done
}
export -f run_all_tests
