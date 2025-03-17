###############################################################################
# Colors for styling output, otherwise utils needs to be sourced at every make
###############################################################################
export RED='\033[0;31m'
export GREEN='\033[0;32m'
export YELLOW='\033[0;33m'
export BLUE='\033[1;34m'
export CYAN='\033[1;36m'
export NC='\033[0m'
export SEPARATOR="============================================================================================"

###############################################################################
# Default values
###############################################################################
export DEFAULT_COMPILE_ONLY="no"
export DEFAULT_TIMESTAMP=$(date +"%Y_%m_%d___%H_%M_%S")
export DEFAULT_TYPES="int32"
export DEFAULT_SIZES="8,64,512,2048,16384,131072,1048576,8388608,67108864"
export DEFAULT_COLLECTIVES="allreduce,allgather,bcast,reduce_scatter"
export DEFAULT_TEST_TIME="01:00:00"
export DEFAULT_OUTPUT_LEVEL="summarized"
export DEFAULT_COMPRESS="yes"
export DEFAULT_DELETE="no"
export DEFAULT_DEBUG_MODE="no"
export DEFAULT_DRY_RUN="no"
export DEFAULT_INTERACTIVE="no"
export DEFAULT_SHOW_MPICH_ENV="no"
export DEFAULT_NOTES=""
export DEFAULT_TASK_PER_NODE=1

###############################################################################
# Utility functions for logging
###############################################################################
error() {
    echo -e "\n${RED}❌❌❌ ERROR: $1 ❌❌❌${NC}\n" >&2
}
export -f error

success() {
    echo -e "\n${GREEN}$1${NC}\n"
}
export -f success

warning() {
    echo -e "\n${YELLOW}WARNING: ${1}${NC}"

    if [[ $# -gt 1 ]]; then
        shift  # Remove the first argument
        for msg in "$@"; do
            echo -e "${YELLOW}  • $msg ${NC}"
        done
    fi
    echo ""
}
export -f warning

inform() {
    echo -e "${BLUE}$1${NC}"
}
export -f inform

###############################################################################
# Cleanup function for SIGINT/SIGTERM
###############################################################################
cleanup() {
    error "Cleanup called! Killing all child processes and aborting..."
    pkill -P $$
    exit 1
}
export -f cleanup

###############################################################################
# Usage function: prints short or full help message
###############################################################################
usage() {
    local help_verbosity=$1
    if [ "$help_verbosity" == "full" ]; then
      cat <<EOF
Usage: $0 --location <LOCATION> --nodes <N_NODES> [options...]

Options:
--location          Location (required)
--nodes             Number of nodes (required if not in --compile-only)
--compile-only      Compile only.
                    [default: "${DEFAULT_COMPILE_ONLY}"]
--output-dir        Output dir of test.
                    [default: "${DEFAULT_TIMESTAMP}" (current timestamp)]
--types             Data types, comma separated.
                    Allowed types: int,int8,int16,int32,int64,float,double,char.
                    [default: "${DEFAULT_TYPES}"]
--sizes             Array sizes, comma separated.
                    [default: "${DEFAULT_SIZES}"]
--collectives       Comma separated list of collectives to test.
                    To each collective, it must correspond a JSON file in `config/test/`.
                    [default: "${DEFAULT_COLLECTIVES}"]
--time              Sbatch time, in format HH:MM:SS.
                    [default: "${DEFAULT_TEST_TIME}"]
--output-level      Specify which test data to save.
                    Allowed values:
                    summarized  - Save summarized test data only.
                    all         - Save all test data.
                    [default: "${DEFAULT_OUTPUT_LEVEL}"]
--compress          Compress result dir into a tar.gz.
                    [default: "${DEFAULT_COMPRESS}"]
--delete            Delete result dir after compression.
                    If --compress is 'no', this will be ignored.
                    [default: "${DEFAULT_DELETE}"]
--debug             Debug mode:
                      - --time is set to 00:10:00
                      - Run only one iteration for each test instance.
                      - Compile with -g -DDEBUG without optimization.
                      - Use int32 data type.
                      - Do not save results (--compress and --delete are ignored).
                      - Do not exit after error.
                    [default: "${DEFAULT_DEBUG_MODE}"]
--dry-run           Dry run mode. Test the script without running the actual bench tests.
                    It differs from debug mode as it will not compile and run code,
                    apart from python scripts to check the configuration and update dynamic rules.
                    [default: "${DEFAULT_DRY_RUN}"]
--interactive       Interactive mode (use salloc instead of sbatch).
                    [default: "${DEFAULT_INTERACTIVE}"]
--show-mpich-env    Show MPICH environment variables.
                    Will only apply if --debug is 'yes' and MPI_LIB is either MPICH or CRAY_MPICH.
                    Otherwise it won't have any effect.
                    [default: "${DEFAULT_SHOW_MPICH_ENV}"]
--notes             Notes for metadata entry.
                    [default: "${DEFAULT_NOTES}"]
--task-per-node     Sbatch tasks per node. As of now, even if you set this to a value greater than 1,
                    srun will run with just -n=--nodes. This behaviour will be updated in future.
                    [default: "${DEFAULT_TASK_PER_NODE}"]
--help              Show short help message
--help-full         Show full help message
EOF
    else
        cat <<EOF
Usage: $0 --location <LOCATION> --nodes <N_NODES> [options...]

Options:
--location          Location (required)
--nodes             Number of nodes (required if not in --compile-only)
--compile-only      Compile only.
                    [default: "${DEFAULT_COMPILE_ONLY}"]

For full help, run: $0 --help-full
EOF
    fi
}

###############################################################################
# Command-line argument parsing
###############################################################################
check_arg() {
    if [[ -z "$2" || "$2" =~ ^-- ]]; then
        error "If given, option '$1' requires an argument."
        usage
        cleanup
    fi
}

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
            --collectives)
                check_arg "$1" "$2"
                export COLLECTIVES="$2"
                shift 2
                ;;
            --time)
                check_arg "$1" "$2"
                export TEST_TIME="$2"
                shift 2
                ;;
            --output-level)
                check_arg "$1" "$2"
                export OUTPUT_LEVEL="$2"
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
            --compile-only)
                check_arg "$1" "$2"
                export COMPILE_ONLY="$2"
                shift 2
                ;;
            --debug)
                check_arg "$1" "$2"
                export DEBUG_MODE="$2"
                shift 2
                ;;
            --dry-run)
                check_arg "$1" "$2"
                export DRY_RUN="$2"
                shift 2
                ;;
            --interactive)
                check_arg "$1" "$2"
                export INTERACTIVE="$2"
                shift 2
                ;;
            --show-mpich-env)
                check_arg "$1" "$2"
                export SHOW_MPICH_ENV="$2"
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
            --help)
                usage
                exit 0
                ;;
            --help-full)
                usage "full"
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

###############################################################################
# Validate required arguments and options
###############################################################################
check_yes_no() {
    if [[ "$1" != "yes" && "$1" != "no" ]]; then
        error "$2 must be either 'yes' or 'no'."
        usage
        return 1
    fi
}

validate_args() {
    check_yes_no "$COMPILE_ONLY" "--compile-only" || return 1
    check_yes_no "$DEBUG_MODE" "--debug" || return 1
    if [[ "$COMPILE_ONLY" == "yes" ]]; then
        success "Compile only mode. Skipping validation."
        return 0
    fi

    if [[ -z "$N_NODES" || ! "$N_NODES" =~ ^[0-9]+$ || "$N_NODES" -lt 2 ]]; then
        error "--nodes must be a numeric value and at least 2."
        usage
        return 1
    elif [[ "$OUTPUT_LEVEL" != "summarized" && "$OUTPUT_LEVEL" != "all" ]]; then
        error "--output-level must be either 'summarized' or 'all'."
        usage
        return 1
    elif [[ ! "$TEST_TIME" =~ ^[0-9]{2}:[0-5][0-9]:[0-5][0-9]$ ]]; then
        error "--time must be in the format 'HH:MM:SS' with minutes and seconds between 00 and 59."
        usage
        return 1
    fi

    check_yes_no "$COMPRESS" "--compress" || return 1
    check_yes_no "$DELETE" "--delete" || return 1
    check_yes_no "$DRY_RUN" "--dry-run" || return 1
    check_yes_no "$INTERACTIVE" "--interactive" || return 1
    check_yes_no "$SHOW_MPICH_ENV" "--show-mpich-env" || return 1

    [[ "$DRY_RUN" == "yes" ]] && warning "DRY RUN MODE: Commands will be printed but not executed"

    if [[ "$COMPRESS" == "no" ]] && [[ "$DELETE" == "yes" ]]; then
        warning "--compress is 'no', hence --delete will be ignored"
        export DELETE="no"
    fi

    if [[ "$DEBUG_MODE" == "yes" ]]; then
        local messages=()
        messages+=("No results will be saved")
        messages+=("Types overridden to 'int32'")
        messages+=("Test time set to 00:10:00")
        [[ "$OUTPUT_LEVEL" != "$DEFAULT_OUTPUT_LEVEL" ]] && messages+=("Output level is set but it will be ignored")
        [[ "$COMPRESS" != "$DEFAULT_COMPRESS" ]] && messages+=("Compress option is set but it will be ignored")
        [[ "$DELETE" != "$DEFAULT_DELETE" ]] && messages+=("Delete option is set but it will be ignored")

        warning "Debug mode enabled" "${messages[@]}"
        export TYPES="int32"
        export TEST_TIME="00:10:00"
    fi

    for type in ${TYPES//,/ }; do
        if [[ ! "$type" =~ ^(int|int8|int16|int32|int64|float|double|char)$ ]]; then
            error " --types must be a comma-separated list. Allowed types: int,int8,int16,int32,int64,float,double,char"
            usage
            return 1
        fi
    done

    for size in ${SIZES//,/ }; do
        if [[ ! "$size" =~ ^[0-9]+$ ]] || [[ $size -lt 1 ]]; then
            error "--sizes must be a comma-separated list of positive integers."
            usage
            return 1
        fi
    done

    local test_conf_files=()
    for collective in ${COLLECTIVES//,/ }; do
        local file_path="$SWING_DIR/config/test/${collective}.json"
        if [ ! -f "$file_path" ]; then
            error "--collectives must be a comma-separated list. No '${collective}.json' file found in config/test/"
            usage
            return 1
        fi
        test_conf_files+=( "$file_path" )
    done
    export TEST_CONFIG_FILES=$(IFS=','; echo "${test_conf_files[*]}")

    if [[ "$SHOW_MPICH_ENV" == "yes" && "$DEBUG_MODE" == "yes" && "$MPI_LIB" == "CRAY_MPICH" ]]; then
        export MPICH_ENV_DISPLAY=1
    fi

    return 0
}

###############################################################################
# Source the environment script for the given location
###############################################################################
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

###############################################################################
# Load required modules
###############################################################################
load_modules(){
    if [ -n "$MODULES" ]; then
        for module in ${MODULES//,/ }; do
            module load $module || { error "Failed to load module $module." ; return 1; }
        done
    fi

    return 0
}

###############################################################################
# Activate virtual environment and install required packages
###############################################################################
activate_virtualenv() {
    if [ -f "$HOME/.swing_venv/bin/activate" ]; then
        source "$HOME/.swing_venv/bin/activate" || { error "Failed to activate virtual environment." ; return 1; }
        success "Virtual environment 'swing_venv' activated."
    else
        warning "Virtual environment 'swing_venv' does not exist. Creating it..."

        python3 -m venv "$HOME/.swing_venv" || { error "Failed to create virtual environment." ; return 1; }
        source "$HOME/.swing_venv/bin/activate" || { error "Failed to activate virtual environment after creation." ; return 1; }

        success "Virtual environment 'swing_venv' created and activated."
    fi

    pip install --upgrade pip > /dev/null || { error "Failed to upgrade pip." ; return 1; }

    local required_python_packages="jsonschema packaging numpy pandas"
    echo "Checking for packages: $required_python_packages"
    for package in $required_python_packages; do
        if ! pip show "$package" > /dev/null 2>&1; then
            warning "Package '$package' not found. Installing..."
            pip install "$package" || { error "Failed to install $package." ; return 1; }
        fi
    done
    success "All Python required packages are already installed."

    return 0
}

###############################################################################
# Compile the codebase
###############################################################################
compile_code() {
    if [ "$DRY_RUN" == "yes" ]; then
        if [ "$DEBUG_MODE" == "yes" ]; then
            inform "Would run: make all DEBUG=1"
        else
            inform "Would run: make all -s"
        fi
        success "Compilation would be attempted (dry run)."
        return 0
    else
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
    fi
}

###############################################################################
# Sanity checks
###############################################################################
print_formatted_list() {
    local list_name="$1"
    local list_items="$2"
    local items_per_line="${3:-5}"  # Default to 5 items per line
    local formatting="${4:-normal}" # Options: normal, numeric, size
    
    echo "  • $list_name:"
    if [[ -z "$list_items" ]]; then
        echo "      None specified"
        return
    fi
    
    case "$formatting" in
        "numeric")
            local i=1
            for item in ${list_items//,/ }; do
                echo "      ${i}. $item"
                ((i++))
            done
            ;;
        *)
            echo -n "      "
            local k=1
            local total_items=$(echo ${list_items//,/ } | wc -w)
            for item in ${list_items//,/ }; do
                if (( k < total_items )); then
                    echo -n "$item, "
                    if (( k % items_per_line == 0 )); then
                        echo
                        echo -n "      "
                    fi
                else
                    echo "$item"
                fi
                ((k++))
            done
            ;;
    esac
}
export -f print_formatted_list

print_section_header() {
    echo -e "\n\n"
    success "${SEPARATOR}\n\t\t\t\t${1}\n${SEPARATOR}"
}
export -f print_section_header

print_sanity_checks() {
    print_section_header "📊 CONFIGURATION SUMMARY"

    inform "Test Configuration:"
    echo "  • Config File:           $TEST_CONFIG"
    echo "  • Location:              $LOCATION"
    echo "  • Debug Mode:            $DEBUG_MODE"
    echo "  • Number of Nodes:       $N_NODES"

    inform "Output Settings:"
    echo "  • Output Level:          $OUTPUT_LEVEL"
    if [ "$DEBUG_MODE" == "no" ]; then
        echo "  • Results Directory:     $DATA_DIR"
        echo "  • Compress Results:      $COMPRESS"
        [ "$COMPRESS" == "yes" ] && echo "  • Delete After Compress: $DELETE"
    else
        echo "  • Results:               Not saving (Debug Mode)"
    fi

    inform "Test Parameters:"
    echo "  • Collective Type:       $COLLECTIVE_TYPE"
    
    print_formatted_list "Algorithms" "${ALGOS[*]}" 1 "numeric"
    print_formatted_list "Array Sizes" "$SIZES" 5 "normal"
    print_formatted_list "Data Types" "$TYPES" 5 "normal"

    inform "System Information:"
    echo "  • MPI Library:           $MPI_LIB $MPI_LIB_VERSION"
    echo "  • Libswing Version:      $LIBSWING_VERSION"
    echo "  • CUDA Enabled:          $CUDA"
    [ -n "$NOTES" ] && echo -e "\nNotes: $NOTES"

    success "${SEPARATOR}"
}
export -f print_sanity_checks

###############################################################################
# Determine the number of iterations based on array size
###############################################################################
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

###############################################################################
# Function to run a single test case
###############################################################################
run_bench() {
    local size=$1 algo=$2 type=$3
    local iter=$(get_iterations $size)
    local command="$RUN $RUNFLAGS -n $N_NODES $BENCH_EXEC $size $iter $algo $type"
    
    [[ "$DEBUG_MODE" == "yes" ]] && inform "DEBUG: $COLLECTIVE_TYPE -> $N_NODES processes, $size array size, $type datatype ($algo)"
    
    if [[ "$DRY_RUN" == "yes" ]]; then
        inform "Would run: $command"
    else
        if [ "$DEBUG_MODE" == "yes" ]; then
            $command
        else
            $command || { error "Failed to run bench for coll=$COLLECTIVE_TYPE, algo=$algo, size=$size, dtype=$type" ; cleanup; }
        fi
    fi
}
export -f run_bench

###############################################################################
# Function to update/select algorithm
###############################################################################
update_algorithm() {
    local algo=$1
    local cvar_indx=$2
    if [[ "$MPI_LIB" == "OMPI_SWING" ]] || [[ "$MPI_LIB" == "OMPI" ]]; then
        success "Updating dynamic rule file for algorithm $algo..."
        python3 $ALGO_CHANGE_SCRIPT $algo || cleanup
        export OMPI_MCA_coll_tuned_dynamic_rules_filename=${DYNAMIC_RULE_FILE}
    elif [[ $MPI_LIB == "MPICH" ]] || [[ $MPI_LIB == "CRAY_MPICH" ]]; then
        local cvar=${CVARS[$cvar_indx]}
        if [[ $MPI_LIB == "CRAY_MPICH" ]]; then
            warning "CRAY_MPICH may not support algorithm selection via CVARs. Results may vary."
        fi
        # FIX: CRAY_MPICH does not support algo selection
        #
        # Disable optimized collectives for MPICH that can override algorithm selection
        #     export MPICH_ALLREDUCE_NO_SMP=1
        #     if [[ "$cvar" == "auto" ]]; then
        #         export MPICH_COLL_OPT_OFF=0
        #         export MPICH_SHARED_MEM_COLL_OPT=0
        #     else 
        #         export MPICH_COLL_OPT_OFF=1
        #         export MPICH_SHARED_MEM_COLL_OPT=1
        #     fi
        success "Setting 'MPIR_CVAR_${COLLECTIVE_TYPE}_INTRA_ALGORITHM=$cvar' for algorithm $algo..."
        export "MPIR_CVAR_${COLLECTIVE_TYPE}_INTRA_ALGORITHM"=$cvar

    fi
}
export -f update_algorithm

###############################################################################
# Loop through algorithms, sizes, and types to run all tests
###############################################################################
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
