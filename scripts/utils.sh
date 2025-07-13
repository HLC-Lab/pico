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

# General options
export DEFAULT_TASKS_PER_NODE="1"
export DEFAULT_COMPILE_ONLY="no"
export DEFAULT_TIMESTAMP=$(date +"%Y_%m_%d___%H_%M_%S")
export DEFAULT_TYPES="int32"
export DEFAULT_SIZES="8,64,512,4096,32768,262144,2097152,16777216,134217728"
export DEFAULT_SEGMENT_SIZES="0,16384,131072,1048576"
export DEFAULT_COLLECTIVES="allreduce,allgather,alltoall,bcast,gather,reduce,reduce_scatter,scatter"

# GPU options
export DEFAULT_GPU_AWARENESS="no"
export DEFAULT_GPU_PER_NODE="0"

# Data saving options
export DEFAULT_OUTPUT_LEVEL="summarized"
export DEFAULT_COMPRESS="yes"
export DEFAULT_DELETE="no"
export DEFAULT_NOTES=""

# Various SLURM options
export DEFAULT_TEST_TIME="01:00:00"
export DEFAULT_EXCLUDE_NODES=""
export DEFAULT_JOB_DEP=""
export DEFAULT_OTHER_SLURM_PARAMS=""
export DEFAULT_INTERACTIVE="no"

# Debug options
export DEFAULT_DEBUG_MODE="no"
export DEFAULT_DRY_RUN="no"
export DEFAULT_SHOW_ENV="no"

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
        shift
        for msg in "$@"; do
            echo -e "${YELLOW}  • $msg ${NC}"
        done
    fi
    echo ""
}
export -f warning

inform() {
    echo -e "${BLUE}$1${NC}"

    if [[ $# -gt 1 ]]; then
        shift
        for msg in "$@"; do
            echo -e "  • $msg "
        done
    fi
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
usage_required() {
inform "Required arguments:"
      cat <<EOF
  --location          Location
  --nodes             Number of nodes (required if not in --compile-only)
EOF
}

usage_general() {
inform "General options:"
      cat <<EOF
  --ntasks-per-node   Comma separated list of number of tasks per node to use in the test.
                      It will have effect if --gpu-per-node is 0.
                      [default: "${DEFAULT_TASKS_PER_NODE}"]
  --ntasks            Total number of tasks. Must be greater than or equal to --nodes.
                      Will override tasks per node and conflicts with --gpu-awareness options.
  --compile-only      Compile only.
                      [default: "${DEFAULT_COMPILE_ONLY}"]
  --output-dir        Output dir of test.
                      [default: "${DEFAULT_TIMESTAMP}" (current timestamp)]
  --types             Data types, comma separated.
                      [default: "${DEFAULT_TYPES}"]
  --sizes             Array sizes in nuber of elements, comma separated.
                      [default: "${DEFAULT_SIZES}"]
  --segment-sizes     Segment sizes in bytes, comma separated.
                      [default: "${DEFAULT_SEGMENT_SIZES}"]
  --collectives       Comma separated list of collectives to test. To each collective, it must correspond a JSON file in 'config/test/'.
                      [default: "${DEFAULT_COLLECTIVES}"]
EOF
}

usage_gpu() {
inform "GPU options:"
      cat <<EOF
  --gpu-awareness     Test GPU aware MPI. Library tested must be GPU aware.
                      Moreover in 'config/environments/.." PARTITION_GPUS_PER_NODE,
                      GPU_LIB and GPU_LIB_VERSION must be defined.
                      [default: "${DEFAULT_GPU_AWARENESS}"]
  --gpu-per-node      Comma separated list of number of gpus per node to use in the test.
                      Each number must be less than or equal to PARTITION_GPUS_PER_NODE.
                      If 0, the test will run on CPU with the --ntasks-per-node value(s).
                      If not specified, it will be set to the value PARTITION_GPUS_PER_NODE defined in 'config/environments/'.
                      [default: "${DEFAULT_GPU_PER_NODE}"]
EOF
}

usage_data() {
inform "Data saving options:"
      cat <<EOF
  --output-level      Specify which test data to save. Allowed values: summarized, all.
                      [default: "${DEFAULT_OUTPUT_LEVEL}"]
  --compress          Compress result dir into a tar.gz.
                      [default: "${DEFAULT_COMPRESS}"]
  --delete            Delete result dir after compression. If --compress is 'no', this will be ignored.
                      [default: "${DEFAULT_DELETE}"]
  --notes             Notes for metadata entry.
                      [default: "${DEFAULT_NOTES}"]
EOF
}

usage_job() {
inform "Various SLURM options:"
      cat <<EOF
  --time              Sbatch time, in format HH:MM:SS.
                      [default: "${DEFAULT_TEST_TIME}"]
  --exclude-nodes     List of nodes to exclude from the test. Refer to SLURM documentation for the format.
                      [default: "${DEFAULT_EXCLUDE_NODES}"]
  --job-dep           Colon separated list of Slurm job dependencies. It is set to 'afterany'.
                      [default: "${DEFAULT_JOB_DEP}"]
  --other-params      Other parameters to pass to the job submission command.
                      [default: "${DEFAULT_OTHER_SLURM_PARAMS}"]
  --interactive       Interactive mode (use salloc instead of sbatch).
                      [default: "${DEFAULT_INTERACTIVE}"]
EOF
}

usage_debug() {
inform "Debug options:"
      cat <<EOF
  --debug             Debug mode:
                        - --time is set to 00:10:00
                        - Run only one iteration for each test instance.
                        - Compile with -g -DDEBUG without optimization.
                        - Do not save results (--compress and --delete are ignored).
                      [default: "${DEFAULT_DEBUG_MODE}"]
  --dry-run           Dry run mode. Test the script without running the actual bench tests.
                      [default: "${DEFAULT_DRY_RUN}"]
  --show-env          Show MPI environment variables when srun is launched. Will only apply if --debug is 'yes'.
                      [default: "${DEFAULT_SHOW_ENV}"]
EOF
}

usage_help() {
      cat <<EOF

--help              Show short help message
--help-full         Show full help message
EOF

}

usage() {
    inform "Usage:" "\$ $0 --location <LOCATION> --nodes <N_NODES> [options...]\n"
    usage_required

    local help_verbosity=$1
    case "$help_verbosity" in
        full)
            usage_general
            usage_gpu
            usage_data
            usage_job
            usage_debug
            usage_help
            ;;
        general)
            usage_general
            ;;
        gpu)
            usage_gpu
            ;;
        data)
            usage_data
            ;;
        job)
            usage_job
            ;;
        debug)
            usage_debug
            ;;
        help)
            usage_help
            ;;
        *)
          ;;
    esac

    inform "For full help, run: $0 --help-full"
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
          # Required arguments
            --location)
                export LOCATION="$2"
                shift 2
                ;;
            --nodes)
                export N_NODES="$2"
                shift 2
                ;;
            # General options
            --ntasks-per-node)
                check_arg "$1" "$2"
                export TASKS_PER_NODE="$2"
                shift 2
                ;;
            --ntasks)
                export FORCE_TASKS="$2"
                shift 2
                ;;
            --compile-only)
                check_arg "$1" "$2"
                export COMPILE_ONLY="$2"
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
            --segment-sizes)
                check_arg "$1" "$2"
                export SEGMENT_SIZES="$2"
                shift 2
                ;;
            --collectives)
                check_arg "$1" "$2"
                export COLLECTIVES="$2"
                shift 2
                ;;
            # GPU options
            --gpu-awareness)
                check_arg "$1" "$2"
                export GPU_AWARENESS="$2"
                shift 2
                ;;
            --gpu-per-node)
                check_arg "$1" "$2"
                export GPU_PER_NODE="$2"
                shift 2
                ;;
            # Data saving options
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
            --notes)
                check_arg "$1" "$2"
                export NOTES="$2"
                shift 2
                ;;
            # Various SLURM options
            --time)
                check_arg "$1" "$2"
                export TEST_TIME="$2"
                shift 2
                ;;
            --exclude-nodes)
                check_arg "$1" "$2"
                export EXCLUDE_NODES="$2"
                shift 2
                ;;
            --job-dep)
                check_arg "$1" "$2"
                export JOB_DEP="$2"
                shift 2
                ;;
            --other-params)
                check_arg "$1" "$2"
                export OTHER_SLURM_PARAMS="$2"
                shift 2
                ;;
            --interactive)
                check_arg "$1" "$2"
                export INTERACTIVE="$2"
                shift 2
                ;;
            # Debug options
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
            --show-env)
                check_arg "$1" "$2"
                export SHOW_ENV="$2"
                shift 2
                ;;
            # Help messages
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
                usage "full"
                cleanup
                ;;
        esac
    done
}

###############################################################################
# Validate required arguments and options
###############################################################################
check_enum() {
    local val=$1 flag=$2 ctx=$3 allowed=$4
    for a in "${allowed//,/ }"; do
        [[ "$val" == "$a" ]]
        return 0
    done

    error "$flag must be one of: ${allowed}."
    usage "$ctx"
    return 1
}

check_regex() {
    local val=$1 flag=$2 ctx=$3 re=$4
    [[ "$val" =~ $re ]] || { error "$flag must match '$re'."; usage "$ctx"; return 1; }
}

check_integer() {
    local val=$1 flag=$2 ctx=$3 min=$4 max=${5-}

    if ! [[ "$val" =~ ^[0-9]+$ ]] || (( val < min )); then
        error "$flag must be an integer ≥ $min."
        usage "$ctx"
        return 1
    fi

    if [[ -n "$max" && "$val" -gt "$max" ]]; then
        error "$flag must be an integer ≤ $max."
        usage "$ctx"
        return 1
    fi
}

check_list() {
    local val=$1 re=$2 flag=$3 ctx=$4
    for item in ${val//,/ }; do
        [[ $item =~ $re ]] || { error "$flag contains invalid entry '$item'."; usage "$ctx"; return 1; }
    done
}

validate_args() {
    # Check validity of arguments
    check_enum "$COMPILE_ONLY" "--compile-only" "general" "yes,no" || return 1
    [[ "$COMPILE_ONLY" == no ]] && { check_integer "$N_NODES" "--nodes" "required" 2 || return 1; }
    [[ -n "$FORCE_TASKS" ]] && { check_integer "$FORCE_TASKS" "--ntasks" "general" "$N_NODES" || return 1; }

    local slurm_tasks_per_node=1
    for tasks in ${TASKS_PER_NODE//,/ }; do
        check_integer "$tasks" "--ntasks-per-node" "general" 1 "$PARTITION_CPUS_PER_NODE" || return 1
        [[ "$tasks" -gt "$slurm_tasks_per_node" ]] && slurm_tasks_per_node="$tasks"
    done

    check_list "$TYPES" "^(int|int8|int16|int32|int64|float|double|char)$" "--types" "general" || return 1
    check_list "$SIZES" "^[0-9]+$" "--sizes" "general" || return 1
    check_list "$SEGMENT_SIZES" "^[0-9]+$" "--segment-sizes" "general" || return 1

    check_enum "$GPU_AWARENESS" "--gpu-awareness" "gpu" "yes,no" || return 1
    if [[ "$GPU_AWARENESS" == "yes" ]]; then
        for gpu in ${GPU_PER_NODE//,/ }; do
            check_integer "$gpu" "--gpu-per-node" "gpu" 0 "$PARTITION_GPUS_PER_NODE" || return 1
            [[ "$gpu" -gt "$slurm_tasks_per_node" ]] && slurm_tasks_per_node="$gpu"
        done
    fi

    check_enum "$OUTPUT_LEVEL" "--output-level" "data" "summarized,all" || return 1
    check_enum "$COMPRESS" "--compress" "data" "yes,no" || return 1
    check_enum "$DELETE" "--delete" "data" "yes,no" || return 1

    check_regex "$TEST_TIME" "--time" "job" "^[0-9]{2}:[0-5][0-9]:[0-5][0-9]$" || return 1
    check_list "$JOB_DEP" "^[0-9]+$" "--job-dep" "job" || return 1
    check_enum "$INTERACTIVE" "--interactive" "job" "yes,no" || return 1

    check_enum "$DEBUG_MODE" "--debug" "debug" "yes,no" || return 1
    check_enum "$DRY_RUN" "--dry-run" "debug" "yes,no" || return 1
    check_enum "$SHOW_ENV" "--show-env" "debug" "yes,no" || return 1

    export SLURM_TASKS_PER_NODE="$slurm_tasks_per_node"

    [[ "$DRY_RUN" == "yes" ]] && warning "DRY RUN MODE: Commands will be printed but not executed"
    [[ "$COMPRESS" == "no" && "$DELETE" == "yes" ]] && warning "--compress is 'no', hence --delete will be ignored" && export DELETE="no"

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

    local test_conf_files=()
    for collective in ${COLLECTIVES//,/ }; do
        if [[ $collective != "allgather" && "$GPU_AWARENESS" == "yes" ]]; then
            error "Only 'allgather' collective is supported with GPUS."
            usage "gpu"
            return 1
        fi

        local file_path="$BINE_DIR/config/test/${collective}.json"
        if [ ! -f "$file_path" ]; then
            error "--collectives must be a comma-separated list. No '${collective}.json' file found in config/test/"
            usage "general"
            return 1
        fi
        test_conf_files+=( "$file_path" )
    done
    export TEST_CONFIG_FILES=$(IFS=','; echo "${test_conf_files[*]}")

    if [[ "$GPU_AWARENESS" == "yes" ]]; then
        [[ -z "$GPU_LIB" || -z "$GPU_LIB_VERSION" ]] && { error "GPU_LIB and GPU_LIB_VERSION must be defined in the environment script."; return 1; }
        check_enum "$GPU_LIB" "--gpu-lib" "gpu" "CUDA,ROCm" || return 1
        [[ "$GPU_PER_NODE" == "0" ]] && { error "GPU_PER_NODE is set to 0 while GPU_AWARENESS is 'yes'."; return 1; }
    else
        [[ "$GPU_PER_NODE" != "0" ]] && { error "GPU_PER_NODE must be 0 when GPU_AWARENESS is 'no'."; return 1; }
    fi

    if [[ -n "$FORCE_TASKS" ]]; then
        warning "--ntasks is set. It will override --ntasks-per-node, --gpu-awareness and --gpu-per-node."
        export GPU_AWARENESS="no"
        export GPU_PER_NODE="0"
    fi

    return 0
}

###############################################################################
# Source the environment script for the given location
###############################################################################
source_environment() {
    [[ -z "$LOCATION" ]] && { error "Location not provided." ; usage "required"; return 1; }

    local env_file="config/environments/$LOCATION.sh"
    [[ ! -f "$env_file" ]] && { error "Environment script for '${LOCATION}' not found!"; usage "required"; return 1; }

    source "$env_file" || { error "Failed to source environment script for '${LOCATION}'."; return 1; }

    local required_vars=(
        BINECC
        RUN
        MPI_LIB
        MPI_LIB_VERSION
        PARTITION_CPUS_PER_NODE
        PARTITION_GPUS_PER_NODE
    )

    if [[ "$LOCATION" != "local" ]]; then
        required_vars+=(
            PARTITION
            ACCOUNT
        )
    fi

    local missing_vars=()
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var}" ]]; then
            missing_vars+=("$var")
        fi
    done

    if (( ${#missing_vars[@]} > 0 )); then
        for var in "${missing_vars[@]}"; do
            error "'$var' is not defined in config/environments/${LOCATION}.sh"
        done
        return 1
    fi

    return 0
}

###############################################################################
# Load required modules
###############################################################################
load_modules(){
    if [ -n "$MODULES" ]; then
        inform "Loading modules: $MODULES"
        for module in ${MODULES//,/ }; do
            module load $module || { error "Failed to load module $module." ; return 1; }
        done
        success "Modules successfully loaded."
    fi

    return 0
}

###############################################################################
# Activate virtual environment and install required packages
###############################################################################
activate_virtualenv() {
    if [ -f "$HOME/.bine_venv/bin/activate" ]; then
        source "$HOME/.bine_venv/bin/activate" || { error "Failed to activate virtual environment." ; return 1; }
        success "Virtual environment 'bine_venv' activated."
    else
        warning "Virtual environment 'bine_venv' does not exist. Creating it..."

        python3 -m venv "$HOME/.bine_venv" || { error "Failed to create virtual environment." ; return 1; }
        source "$HOME/.bine_venv/bin/activate" || { error "Failed to activate virtual environment after creation." ; return 1; }

        success "Virtual environment 'bine_venv' created and activated."
    fi

    if [[ "$LOCATION" != "mare_nostrum" ]]; then
        pip install --upgrade pip > /dev/null || { error "Failed to upgrade pip." ; return 1; }
    fi

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
    make_command="make all"
    [[ "$DEBUG_MODE" == "yes" ]] && make_command+=" DEBUG=1" ||  make_command+=" -s"
    if [[ "$GPU_AWARENESS" == "yes" ]]; then
      case "$GPU_LIB" in
        "CUDA")
            make_command+=" CUDA_AWARE=1"
            ;;
        # "HIP")
        #     make_command+=" HIP_AWARE=1"
        #     ;;
        *)
            error "Invalid GPU_LIB value: $GPU_LIB"
            return 1
            ;;
        esac
    fi

    if [[ "$DRY_RUN" == "yes" ]]; then
        inform "Would run: $make_command"
        success "Compilation would be attempted (dry run)."
        return 0
    fi

    if ! $make_command; then
        error "Compilation failed. Exiting."
        return 1
    fi

    success "Compilation succeeded."
    return 0
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
    echo "  • Total MPI tasks:       $MPI_TASKS"
    [[ -z "$FORCE_TASKS" ]] && echo "  • Task per Node:         $CURRENT_TASKS_PER_NODE"

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
    echo "  • Libbine Version:      $LIBBINE_VERSION"
    echo "  • GPU awareness:         $GPU_AWARENESS"
    if [[ "$GPU_AWARENESS" == "yes" ]]; then
        echo "  • GPU per node:          $CURRENT_TASKS_PER_NODE"
        echo "  • GPU library:           $GPU_LIB"
        echo "  • GPU library version:   $GPU_LIB_VERSION"
    fi
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
    local command="$RUN $RUNFLAGS -n $MPI_TASKS $BENCH_EXEC $size $iter $algo $type"

    [[ "$DEBUG_MODE" == "yes" ]] && inform "DEBUG: $COLLECTIVE_TYPE -> $MPI_TASKS processes ($N_NODES nodes), $size array size, $type datatype ($algo)" && [[ "$SEGMENTED" == "yes" ]] && echo "Segment size: $SEGSIZE"

    if [[ "$DRY_RUN" == "yes" ]]; then
        inform "Would run: $command"
    else
        if [[ "$DEBUG_MODE" == "yes" ]]; then
            $command
        else
            # WARN: Removed panic mode for full cluster run
            #
            # $command || { error "Failed to run bench for coll=$COLLECTIVE_TYPE, algo=$algo, size=$size, dtype=$type" ; cleanup; }
            [[ "$LOCATION" == "mare_nostrum" || "$LOCATION" == "leonardo" ]] && sleep 1  # To avoid step timeout due to previous srun still not finalized
            $command
        fi
    fi
}
export -f run_bench

###############################################################################
# Function to update/select algorithm
###############################################################################
update_algorithm() {
    local algo="$1"
    local cvar_indx="$2"
    case "$MPI_LIB" in
        "OMPI_BINE" | "OMPI")
            success "Updating dynamic rule file for algorithm $algo..."
            python3 "$ALGO_CHANGE_SCRIPT" "$algo" || cleanup
            export OMPI_MCA_coll_tuned_dynamic_rules_filename="${DYNAMIC_RULE_FILE}"
            ;;
        "MPICH")
            local cvar="${CVARS[$cvar_indx]}"
            local var_name="MPICH_CVAR_${COLLECTIVE_TYPE}_INTRA_ALGORITHM"
            export "${var_name}"="$cvar"
            success "Setting MPICH_CVAR_${COLLECTIVE_TYPE}_INTRA_ALGORITHM=$cvar for algorithm $algo..."
            ;;
        "CRAY_MPICH")
            local cvar="${CVARS[$cvar_indx]}"
            local var_name="MPICH_${COLLECTIVE_TYPE}_INTRA_ALGORITHM"
            local var_name_2="MPICH_${COLLECTIVE_TYPE}_DEVICE_COLLECTIVE"
            export MPICH_COLL_OPT_OFF=1
            export MPICH_SHARED_MEM_COLL_OPT=0
            export "${var_name_2}"="0"

            if [[ "$cvar" == "reduce_scatter_allgather"  || "$cvar" == "reduce_scatter_gather" ]]; then
                export MPICH_OFI_CXI_COUNTER_REPORT=0
                export MPICH_OFI_SKIP_NIC_SIMMETRY_TEST=1
            else
                export MPICH_OFI_CXI_COUNTER_REPORT=1
                export MPICH_OFI_SKIP_NIC_SIMMETRY_TEST=0
            fi
            [[ $algo == "default_mpich" ]] && export MPICH_COLL_OPT_OFF=0 && export MPICH_SHARED_MEM_COLL_OPT=1 && export "${var_name_2}"="1"
            export "${var_name}"="$cvar"
            success "Setting MPICH_${COLLECTIVE_TYPE}_INTRA_ALGORITHM=$cvar for algorithm $algo..."
            ;;
        *)
            echo "Error: Unsupported MPI_LIB value: $MPI_LIB" >&2
            return 1
            ;;
    esac
}
export -f update_algorithm

###############################################################################
# Loop through algorithms, sizes, and types to run all tests
###############################################################################
run_all_tests() {
    local i=0
    for algo in ${ALGOS[@]}; do
        update_algorithm $algo $i || { error "Failed to update algorithm $algo" ; cleanup; }
        export SEGMENTED=${IS_SEGMENTED[$i]}
        inform "Segmented: $SEGMENTED"

        [[ "$DEBUG_MODE" == "no" ]] && inform "BENCH: $COLLECTIVE_TYPE -> $MPI_TASKS processes ($N_NODES nodes)"

        for size in ${SIZES//,/ }; do
            if [[ $size -lt $MPI_TASKS && " ${SKIP} " =~ " ${algo} " ]]; then
                echo "Skipping algorithm $algo for size=$size < MPI_TASKS=$MPI_TASKS"
                continue
            fi

            if [[ "$SEGMENTED" == "yes" ]]; then
                for type in ${TYPES//,/ }; do
                    for segment_size in ${SEGMENT_SIZES//,/ }; do
                        export SEGSIZE=$segment_size
                        run_bench $size $algo $type
                    done
                done
            else
                for type in ${TYPES//,/ }; do
                    run_bench $size $algo $type
                done
            fi
        done
        ((i++))
    done
}
export -f run_all_tests
