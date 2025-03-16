# Colors for styling output, otherwise utils needs to be sourced at every make
export RED='\033[0;31m'
export GREEN='\033[0;32m'
export YELLOW='\033[0;33m'
export BLUE='\033[1;34m'
export NC='\033[0m'


# Set the default values
export DEFAULT_TIMESTAMP=$(date +"%Y_%m_%d___%H_%M_%S")
export DEFAULT_OUTPUT_LEVEL="summarized"
export DEFAULT_TYPES="int32"
export DEFAULT_SIZES="8,64,512,2048,16384,131072,1048576,8388608,67108864"
export DEFAULT_DEBUG_MODE="no"
export DEFAULT_INTERACTIVE="no"
export DEFAULT_COMPRESS="yes"
export DEFAULT_DELETE="no"
export DEFAULT_NOTES=""
export DEFAULT_TASK_PER_NODE=1
export DEFAULT_TEST_TIME="01:00:00"

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

# Info messages in blue
inform() {
    echo -e "${BLUE}$1${NC}"
}
export -f inform

# Cleanup function for SIGINT
cleanup() {
    error "Cleanup called! Killing all child processes and aborting..."
    pkill -P $$
    exit 1
}
export -f cleanup

check_arg() {
    if [[ -z "$2" || "$2" =~ ^-- ]]; then
        error "If given, option '$1' requires an argument."
        usage
        cleanup
    fi
}

# Show the usage message
usage() {
    cat <<EOF
Usage: $0 --location <LOCATION> --nodes <N_NODES> [options...]

Options:
  --location          Location (required)
  --nodes             Number of nodes (required, integer >=2)
  --output-dir        Output dir of test.
                      [default: "${DEFAULT_TIMESTAMP}" (current timestamp)]
  --types             Data types, comma separated.
                      Allowed types: int,int8,int16,int32,int64,float,double,char.
                      [default: "${DEFAULT_TYPES}"]
  --sizes             Array sizes, comma separated.
                      [default: "${DEFAULT_SIZES}"]
  --output-level      Specify which test data to save.
                      Allowed values:
                        summarized  - Save summarized test data only.
                        all         - Save all test data.
                      [default: "${DEFAULT_OUTPUT_LEVEL}"]
  --test-config       Relative paths to config files, comma separated.
                      [default: "config/test/*.json"]
  --interactive       Interactive mode (use salloc instead of sbatch).
                      [default: "${DEFAULT_INTERACTIVE}"]
  --compress          Compress result dir into a tar.gz.
                      [default: "${DEFAULT_COMPRESS}"]
  --delete            Delete result dir after compression.
                      If --compress is 'no', this will be ignored.
                      [default: "${DEFAULT_DELETE}"]
  --debug             Debug mode:
                        - Compile with -g -DDEBUG without optimization.
                        - Use int32 data type.
                        - Do not save results (--compress and --delete are ignored).
                        - Do not exit after error.
                      [default: "${DEFAULT_DEBUG_MODE}"]
  --notes             Notes for metadata entry.
                      [default: "${DEFAULT_NOTES}"]
  --task-per-node     Sbatch tasks per node.
                      [default: "${DEFAULT_TASK_PER_NODE}"]
  --time              Sbatch time, in format HH:MM:SS.
                      [default: "${DEFAULT_TEST_TIME}"]
  --help              Show this help message
EOF
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
                check_arg "$1" "$2"
                export TIMESTAMP="$2"
                shift 2
                ;;
            --types)
                check_arg "$1" "$2"
                export TYPES="$2"
                shift 2
                ;;
            --sizes)
                check_arg "$1" "$2"
                export SIZES="$2"
                shift 2
                ;;
            --test-config)
                check_arg "$1" "$2"
                TEST_CONFIG_OVERRIDE="$2"
                shift 2
                ;;
            --output-level)
                check_arg "$1" "$2"
                export OUTPUT_LEVEL="$2"
                shift 2
                ;;
            --interactive)
                check_arg "$1" "$2"
                export INTERACTIVE="$2"
                shift 2
                ;;
            --compress)
                check_arg "$1" "$2"
                export COMPRESS="$2"
                shift 2
                ;;
            --delete)
                check_arg "$1" "$2"
                export DELETE="$2"
                shift 2
                ;;
            --debug)
                check_arg "$1" "$2"
                export DEBUG_MODE="$2"
                shift 2
                ;;
            --notes)
                check_arg "$1" "$2"
                export NOTES="$2"
                shift 2
                ;;
            --task-per-node)
                check_arg "$1" "$2"
                export TASK_PER_NODE="$2"
                shift 2
                ;;
            --time)
                check_arg "$1" "$2"
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
                cleanup
                ;;
        esac
    done
}


# Validate the command line arguments
validate_args() {
    if [[ -z "$N_NODES" ]] || [[ ! "$N_NODES" =~ ^[0-9]+$ ]] || [ "$N_NODES" -lt 2 ]; then
        error "--nodes must be a numeric value and at least 2."
        usage
        return 1
    elif [[ "$OUTPUT_LEVEL" != "summarized" ]] && [[ "$OUTPUT_LEVEL" != "all" ]]; then
        error "--output-level must be either 'summarized' or 'all'."
        usage
        return 1
    elif [[ "$INTERACTIVE" != "yes" ]] && [[ "$INTERACTIVE" != "no" ]]; then
        error "--interactive must be either 'yes' or 'no'."
        usage
        return 1
    elif [[ "$DEBUG_MODE" != "yes" ]] && [[ "$DEBUG_MODE" != "no" ]]; then
        error "--debug must be either 'yes' or 'no'."
        usage
        return 1
    elif [[ "$COMPRESS" != "yes" ]] && [[ "$COMPRESS" != "no" ]]; then
        error "--compress must be either 'yes' or 'no'."
        usage
        return 1
    elif [[ "$DELETE" != "yes" ]] && [[ "$DELETE" != "no" ]]; then
        error "--delete must be either 'yes' or 'no'."
        usage
        return 1
    elif [[ ! "$TEST_TIME" =~ ^[0-9]{2}:[0-5][0-9]:[0-5][0-9]$ ]]; then
        error "--time must be in the format 'HH:MM:SS' with minutes and seconds between 00 and 59."
        usage
        return 1
    fi

    if [[ "$COMPRESS" == "no" ]] && [[ "$DELETE" == "yes" ]]; then
        warning "--compress is 'no', hence --delete will be ignored"
        export $DELETE="no"
    fi

    if [[ "$DEBUG_MODE" == "yes" ]]; then
        warning "Debug mode enabled. No results will be saved."
        warning "Overriding --types to 'int32' regardless of the configuration."
        export TYPES="int32"
    fi

    for size in ${SIZES//,/ }; do
        if [[ ! "$size" =~ ^[0-9]+$ ]] || [[ $size -lt 1 ]]; then
            error "--sizes must be a comma-separated list of positive integers."
            usage
            return 1
        fi
    done

    for type in ${TYPES//,/ }; do
        if [[ ! "$type" =~ ^(int|int8|int16|int32|int64|float|double|char)$ ]]; then
            error " --types must be a comma-separated list. Allowed types: int,int8,int16,int32,int64,float,double,char"
            usage
            return 1
        fi
    done


    return 0
}


# Source the environment configuration
source_environment() {
    if [ -z "$1" ]; then
        error "--location not provided."
        usage
        return 1
    fi

    local env_file="config/environments/$1.sh"
    if [ -f "$env_file" ]; then
        source "$env_file"
        return 0
    else
        error "Environment script for '${LOCATION}' not found!"
        usage
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
activate_virtualenv() {
    if [ -f "$HOME/.swing_venv/bin/activate" ]; then
        source "$HOME/.swing_venv/bin/activate" || { error "Failed to activate virtual environment." ; return 1; }
        success "Virtual environment 'swing_venv' activated."
    else
        warning "Virtual environment 'swing_venv' does not exist. Creating it..."
        
        # Create the virtual environment
        python3 -m venv "$HOME/.swing_venv" || { error "Failed to create virtual environment." ; return 1; }
        source "$HOME/.swing_venv/bin/activate" || { error "Failed to activate virtual environment after creation." ; return 1; }

        success "Virtual environment 'swing_venv' created and activated."
    fi

    # Check and install missing packages
    pip install --upgrade pip || { error "Failed to upgrade pip." ; return 1; }

    local required_python_packages="jsonschema packaging numpy pandas"
    for package in $required_python_packages; do
        if ! pip show "$package" > /dev/null 2>&1; then
            warning "Package '$package' not found. Installing..."
            pip install "$package" || { error "Failed to install $package." ; return 1; }
        fi
        success "Package $package already installed."
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
run_bench() {
    local size=$1 algo=$2 type=$3
    local iter=$(get_iterations $size)

    if [ "$DEBUG_MODE" == "yes" ]; then
        inform "DEBUG: $COLLECTIVE_TYPE -> $N_NODES processes, $size array size, $type datatype ($algo)"
        $RUN $RUNFLAGS -n $N_NODES $BENCH_EXEC $size $iter $algo $type
    else
        $RUN $RUNFLAGS -n $N_NODES $BENCH_EXEC $size $iter $algo $type || { error "Failed to run bench for coll=$COLLECTIVE_TYPE, algo=$algo, size=$size, dtype=$type" ; cleanup; }
    fi
}
export -f run_bench


# Function to select the algorithm by exporting the cvar or updating dynamic rules
update_algorithm() {
    local algo=$1
    local cvar_indx=$2
    if [[ "$MPI_LIB" == "OMPI_SWING" ]] || [[ "$MPI_LIB" == "OMPI" ]]; then
        success "Updating dynamic rule file for algorithm $algo..."
        python3 $ALGO_CHANGE_SCRIPT $algo || cleanup
        export OMPI_MCA_coll_tuned_dynamic_rules_filename=${DYNAMIC_RULE_FILE}
    elif [[ $MPI_LIB == "MPICH" ]] || [[ $MPI_LIB == "CRAY_MPICH" ]]; then
        local cvar=${CVARS[$cvar_indx]}
        # Disable optimized collectives for MPICH that can override algorithm selection
        if [[ $MPI_LIB == "CRAY_MPICH" ]]; then
            export MPICH_ALLREDUCE_NO_SMP=1
            if [[ "$cvar" == "auto" ]]; then
                export MPICH_COLL_OPT_OFF=0
                export MPICH_SHARED_MEM_COLL_OPT=0
            else 
                export MPICH_COLL_OPT_OFF=1
                export MPICH_SHARED_MEM_COLL_OPT=1
            fi
        fi
        success "Setting 'MPIR_CVAR_${COLLECTIVE_TYPE}_INTRA_ALGORITHM=$cvar' for algorithm $algo..."
        export "MPIR_CVAR_${COLLECTIVE_TYPE}_INTRA_ALGORITHM"=$cvar

    fi
}
export -f update_algorithm


# Test algorithms here, loop through:
# - algorithms
# - number of mpi processes
# - size of the array in number of elements (of type = TYPE) 
run_all_tests() {
    local i=0
    for algo in ${ALGOS[@]}; do
        update_algorithm $algo $i
        if [ "$DEBUG_MODE" == "no" ]; then
            inform "BENCH: $COLLECTIVE_TYPE -> $N_NODES processes"
        fi

        for size in ${SIZES//,/ }; do
            if [[ $size -lt $N_NODES && " ${SKIP} " =~ " ${algo} " ]]; then
                echo "Skipping algorithm $algo for size=$size < N_NODES=$N_NODES"
                continue
            fi

            for type in ${TYPES//,/ }; do
                run_bench $size $algo $type
            done
        done
        ((i++))
    done
}
export -f run_all_tests
