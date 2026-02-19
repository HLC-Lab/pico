###############################################################################
# Shared shell helpers sourced by the orchestration scripts.
###############################################################################

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
export DEFAULT_INSTRUMENT="no"

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


###############################################################################
# DEBUG HELPERS
###############################################################################
trace_env_snapshot() {
    [[ "$DEBUG_MODE" == "yes" ]] || return 0
    local label="$1"
    echo -e "\n${CYAN}=== ENV SNAPSHOT: ${label} ===${NC}"
    echo "WHOAMI: $(whoami) | SHELL: $SHELL | PWD: $(pwd)"
    echo "PICOCC: ${PICOCC:-<unset>} (which: $(command -v "${PICOCC:-}" 2>/dev/null || echo '<not found>'))"
    echo "PATH:";             echo "${PATH}"              | tr ':' '\n' | nl -w2 -s': ' | sed 's/^/  /'
    echo "LD_LIBRARY_PATH:";  echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | nl -w2 -s': ' | sed 's/^/  /'
    echo "MANPATH:";          echo "${MANPATH:-}"         | tr ':' '\n' | nl -w2 -s': ' | sed 's/^/  /'
    echo -e "${CYAN}=================================${NC}\n"
}
export -f trace_env_snapshot

trace_compiler_wrapper() {
    [[ "$DEBUG_MODE" == "yes" ]] || return 0
    local cc="${1:-$PICOCC}"
    echo -e "${BLUE}>>> Probing compiler wrapper: ${cc}${NC}"
    if ! command -v "$cc" >/dev/null 2>&1; then
        warning "Compiler wrapper '$cc' not found in PATH"
        return 0
    fi
    echo "  realpath: $(readlink -f "$(command -v "$cc")" 2>/dev/null || echo '?')"
    # Open MPI banner/flags (best effort)
    "$cc" -show 2>/dev/null | sed 's/^/  ompi -show: /' || true
    # MPICH banner/flags (best effort)
    "$cc" -compile_info 2>/dev/null | sed 's/^/  mpich -compile_info: /' || true
    # Generic version
    "$cc" --version 2>/dev/null | head -n 1 | sed 's/^/  --version: /' || true
}
export -f trace_compiler_wrapper

trace_ldd() {
    [[ "$DEBUG_MODE" == "yes" ]] || return 0
    local bin="$1"
    if [[ -x "$bin" ]]; then
        echo -e "${BLUE}>>> ldd ${bin}${NC}"
        ldd "$bin" | sed 's/^/  /'
    else
        warning "ldd: '$bin' not found or not executable"
    fi
}
export -f trace_ldd

trace_kv() { [[ "$DEBUG_MODE" == "yes" ]] && echo "  $1=$2"; }
export -f trace_kv

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
    inform "If used with tui json file:" "\$ $0 --file <TUI_FILE>\n"

    local help_verbosity=$1
    case "$help_verbosity" in
        full)
            usage_required
            usage_general
            usage_gpu
            usage_data
            usage_job
            usage_debug
            usage_help
            ;;
        general)
            usage_required
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

    inform "For full help, run: $0 --help-full (or -H)"
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
    # Short circuit if -f is given
    local argv=("$@")
    for ((i=0; i<${#argv[@]}; i++)); do
        case "${argv[i]}" in
            -f|--file)
                export TUI_FILE="${argv[i+1]}"
                return 0
                ;;
        esac
    done

    while [[ $# -gt 0 ]]; do
        case "$1" in
            # Required arguments
            -l|--location)
                export LOCATION="$2"; shift 2 ;;
            -N|--nodes)
                export N_NODES="$2"; shift 2 ;;
            # General options
            --ntasks-per-node)
                check_arg "$1" "$2"; export TASKS_PER_NODE="$2"; shift 2 ;;
            --ntasks)
                check_arg "$1" "$2"; export FORCE_TASKS="$2"; shift 2 ;;
            --compile-only)
                check_arg "$1" "$2"; export COMPILE_ONLY="$2"; shift 2 ;;
            --output-dir)
                check_arg "$1" "$2"; export TIMESTAMP="$2"; shift 2 ;;
            --types)
                check_arg "$1" "$2"; export TYPES="$2"; shift 2 ;;
            --sizes)
                check_arg "$1" "$2"; export SIZES="$2"; shift 2 ;;
            --segment-sizes)
                check_arg "$1" "$2"; export SEGMENT_SIZES="$2"; shift 2 ;;
            --collectives)
                check_arg "$1" "$2"; export COLLECTIVES="$2"; shift 2 ;;
            # GPU options
            --gpu-awareness)
                check_arg "$1" "$2"; export GPU_AWARENESS="$2"; shift 2 ;;
            --gpu-per-node)
                check_arg "$1" "$2"; export GPU_PER_NODE="$2"; shift 2 ;;
            # Data saving options
            --output-level)
                check_arg "$1" "$2"; export OUTPUT_LEVEL="$2"; shift 2 ;;
            --compress)
                check_arg "$1" "$2"; export COMPRESS="$2"; shift 2 ;;
            --delete)
                check_arg "$1" "$2"; export DELETE="$2"; shift 2 ;;
            --notes)
                check_arg "$1" "$2"; export NOTES="$2"; shift 2 ;;
            # Various SLURM options
            --time)
                check_arg "$1" "$2"; export TEST_TIME="$2"; shift 2 ;;
            --exclude-nodes)
                check_arg "$1" "$2"; export EXCLUDE_NODES="$2"; shift 2 ;;
            --job-dep)
                check_arg "$1" "$2"; export JOB_DEP="$2"; shift 2 ;;
            --other-params)
                check_arg "$1" "$2"; export OTHER_SLURM_PARAMS="$2"; shift 2 ;;
            --interactive)
                check_arg "$1" "$2"; export INTERACTIVE="$2"; shift 2 ;;
            # Debug options
            --debug)
                check_arg "$1" "$2"; export DEBUG_MODE="$2"; shift 2 ;;
            --dry-run)
                check_arg "$1" "$2"; export DRY_RUN="$2"; shift 2 ;;
            --show-env)
                check_arg "$1" "$2"; export SHOW_ENV="$2"; shift 2 ;;
            # Help messages
            -h|--help)
                usage; exit 0 ;;
            -H|--help-full)
                usage "full"; exit 0 ;;
            *)
                error "Error: Unknown option $1"
                usage "full"
                cleanup ;;
        esac
    done
}

###############################################################################
# Validate required arguments and options
###############################################################################
check_enum() {
    local val=$1 flag=$2 ctx=$3 allowed=$4
    local match=0
    local allowed_vals=()
    local normalized_val="${val//[[:space:]]/}"
    IFS=',' read -r -a allowed_vals <<< "$allowed"
    for a in "${allowed_vals[@]}"; do
        local trimmed="${a//[[:space:]]/}"
        [[ "$normalized_val" == "$trimmed" ]] && { match=1; break; }
    done

    if (( match )); then
        return 0
    fi

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
        if [[ -z "$GPU_PER_NODE" || "$GPU_PER_NODE" == "0" ]]; then
            export GPU_PER_NODE="$PARTITION_GPUS_PER_NODE"
            inform "GPU_PER_NODE not specified; defaulting to partition GPUs per node: $GPU_PER_NODE"
        fi
        for gpu in ${GPU_PER_NODE//,/ }; do
            check_integer "$gpu" "--gpu-per-node" "gpu" 0 "$PARTITION_GPUS_PER_NODE" || return 1
            [[ "$gpu" -gt "$slurm_tasks_per_node" ]] && slurm_tasks_per_node="$gpu"
        done
    fi

    check_enum "$OUTPUT_LEVEL" "--output-level" "data" "summarized,all" || return 1
    check_enum "$COMPRESS" "--compress" "data" "yes,no" || return 1
    check_enum "$DELETE" "--delete" "data" "yes,no" || return 1

    check_regex "$TEST_TIME" "--time" "job" "^[0-9]{2}:[0-5][0-9]:[0-5][0-9]$" || return 1
    if [[ -n "$JOB_DEP" ]]; then
        local dep
        local -a _job_dep_list=()
        IFS=':' read -r -a _job_dep_list <<< "$JOB_DEP"
        for dep in "${_job_dep_list[@]}"; do
            [[ "$dep" =~ ^[0-9]+$ ]] || { error "--job-dep must be a colon-separated list of numeric job IDs."; usage "job"; return 1; }
        done
    fi
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

        local file_path="$PICO_DIR/config/test/${collective}.json"
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
        PICOCC
        RUN
        MPI_LIB
        MPI_LIB_VERSION
        PARTITION_CPUS_PER_NODE
        PARTITION_GPUS_PER_NODE
    )

    if [[ "$LOCATION" != "local" ]]; then
        required_vars+=(
            PARTITION
            PICO_ACCOUNT
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
# Load and unload required modules or env path
###############################################################################
load_modules(){
    local csv="${1:-$MODULES}"
    if [ -n "$csv" ]; then
        inform "Loading modules: $csv"
        for module in ${csv//,/ }; do
            module load "$module" || { error "Failed to load module $module." ; return 1; }
        done
        success "Modules successfully loaded."
    fi
    return 0
}
export -f load_modules


unload_modules(){
    local csv="$1"
    [ -z "$csv" ] && return 0
    inform "Unloading modules: $csv"
    # reverse order to be safe
    local arr=()
    for m in ${csv//,/ }; do arr+=("$m"); done
    for (( idx=${#arr[@]}-1 ; idx>=0 ; idx-- )); do
        module unload "${arr[$idx]}" >/dev/null 2>&1 || true
    done
    success "Modules successfully unloaded."
    return 0
}
export -f unload_modules



# ---------- apply per-library set_env ----------
apply_lib_env() {
  local i="$1"
  [[ -z "$i" ]] && { error "apply_lib_env: library index required"; return 1; }

  local touched=()
  local changed_any=0

  # controller lists
  local pre_list="$(_get_var "LIB_${i}_ENV_PREPEND_VARS")"
  local app_list="$(_get_var "LIB_${i}_ENV_APPEND_VARS")"
  local set_list="$(_get_var "LIB_${i}_ENV_SET_VARS")"

  # If no controllers exist, signal "nothing to apply"
  if [[ -z "$pre_list" && -z "$app_list" && -z "$set_list" ]]; then
    return 1
  fi

  [[ "$DEBUG_MODE" == "yes" ]] && echo "Applying env for library $i (set_env)"

  # helper: save original and mark touched once
  __save_once() {
    local varname="$1"
    # Skip if already saved
    if [[ -z "$(_get_var "LIB_${i}_SAVED_${varname}")" ]]; then
      local cur_val="${!varname}"
      export "LIB_${i}_SAVED_${varname}=$cur_val"
      touched+=("$varname")
    fi
  }

  # PREPEND
  local names=()
  __csv_to_array "$pre_list" names
  for V in "${names[@]}"; do
    [[ -z "$V" ]] && continue
    local spec="$(_get_var "LIB_${i}_ENV_PREPEND_${V}")"
    [[ -z "$spec" ]] && continue
    __save_once "$V"
    # nameref into the target var
    declare -n ref="$V"
    local old="${ref}"
    ref="${spec}${old:+:$old}"
    changed_any=1
    [[ "$DEBUG_MODE" == "yes" ]] && echo "  PREPEND $V: '${spec}' -> head now '${ref%%:*}'"
  done

  # APPEND
  names=()
  __csv_to_array "$app_list" names
  for V in "${names[@]}"; do
    [[ -z "$V" ]] && continue
    local spec="$(_get_var "LIB_${i}_ENV_APPEND_${V}")"
    [[ -z "$spec" ]] && continue
    __save_once "$V"
    declare -n ref="$V"
    local old="${ref}"
    ref="${old:+$old:}${spec}"
    changed_any=1
    [[ "$DEBUG_MODE" == "yes" ]] && echo "  APPEND $V: '${spec}' -> tail now '${ref##*:}'"
  done

  # SET (expand variables within the spec)
  names=()
  __csv_to_array "$set_list" names
  for V in "${names[@]}"; do
    [[ -z "$V" ]] && continue
    local raw="$(_get_var "LIB_${i}_ENV_SET_${V}")"
    [[ -z "$raw" ]] && continue
    __save_once "$V"
    # Expand $VAR references safely
    local spec_expanded
    spec_expanded="$(eval "printf '%s' \"$raw\"")"
    declare -n ref="$V"
    ref="$spec_expanded"
    changed_any=1
    [[ "$DEBUG_MODE" == "yes" ]] && echo "  SET $V: '${raw}' -> '${spec_expanded}'"
  done

  # record touched list for restore
  if (( changed_any )); then
    # join with commas
    local joined=""
    for V in "${touched[@]}"; do joined+="${joined:+,}$V"; done
    export "LIB_${i}_ENV_TOUCHED=$joined"

    if [[ "$DEBUG_MODE" == "yes" ]]; then
      echo "  AFTER snapshot:"
      [[ -n "$(_get_var "LIB_${i}_ENV_PREPEND_PATH")" || -n "$(_get_var "LIB_${i}_ENV_APPEND_PATH")" || -n "$(_get_var "LIB_${i}_ENV_SET_PATH")" ]] \
        && echo "    PATH head:            ${PATH%%:*}"
      [[ -n "$(_get_var "LIB_${i}_ENV_PREPEND_LD_LIBRARY_PATH")" || -n "$(_get_var "LIB_${i}_ENV_APPEND_LD_LIBRARY_PATH")" || -n "$(_get_var "LIB_${i}_ENV_SET_LD_LIBRARY_PATH")" ]] \
        && echo "    LD_LIBRARY_PATH head: ${LD_LIBRARY_PATH%%:*}"
      [[ -n "$(_get_var "LIB_${i}_ENV_PREPEND_MANPATH")" || -n "$(_get_var "LIB_${i}_ENV_APPEND_MANPATH")" || -n "$(_get_var "LIB_${i}_ENV_SET_MANPATH")" ]] \
        && echo "    MANPATH head:         ${MANPATH%%:*}"
    fi
    return 0
  fi

  # nothing actually applied from the lists provided
  return 1
}
export -f apply_lib_env

# ---------- restore per-library set_env ----------
restore_lib_env() {
  local i="$1"
  [[ -z "$i" ]] && { error "restore_lib_env: library index required"; return 1; }

  local touched_csv="$(_get_var "LIB_${i}_ENV_TOUCHED")"
  [[ -z "$touched_csv" ]] && return 0

  [[ "$DEBUG_MODE" == "yes" ]] && echo "Restoring env for library $i (set_env)"

  local names=()
  __csv_to_array "$touched_csv" names
  for V in "${names[@]}"; do
    local saved="$(_get_var "LIB_${i}_SAVED_${V}")"
    declare -n ref="$V"
    ref="$saved"
    unset "LIB_${i}_SAVED_${V}"
    [[ "$DEBUG_MODE" == "yes" ]] && trace_kv "restore $V" "$ref"
  done

  unset "LIB_${i}_ENV_TOUCHED"
  return 0
}
export -f restore_lib_env


###############################################################################
# Activate virtual environment and install required packages
###############################################################################
activate_virtualenv() {
    if [ -f "$HOME/.pico_venv/bin/activate" ]; then
        source "$HOME/.pico_venv/bin/activate" || { error "Failed to activate virtual environment." ; return 1; }
        success "Virtual environment 'pico_venv' activated."
    else
        warning "Virtual environment 'pico_venv' does not exist. Creating it..."

        python3 -m venv "$HOME/.pico_venv" || { error "Failed to create virtual environment." ; return 1; }
        source "$HOME/.pico_venv/bin/activate" || { error "Failed to activate virtual environment after creation." ; return 1; }

        success "Virtual environment 'pico_venv' created and activated."
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
# WARN: will be deprecated
compile_code() {
    [[ "$BEAR_COMPILE" == "yes" ]] && make_command="bear -- make all" || make_command="make all" # Used to create compile_command.json file for lsp
    [[ "$DEBUG_MODE" == "yes" ]] && make_command+=" DEBUG=1" ||  make_command+=" -s"
    [[ "$INSTRUMENT" == "yes" ]] && make_command+=" PICO_INSTRUMENT=1" && inform "Instrumented build requested"

    if [[ "$GPU_AWARENESS" == "yes" ]]; then
      case "$GPU_LIB" in
        "CUDA")
            make_command+=" PICO_MPI_CUDA_AWARE=1"
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

    #if [[ "$GPU_NATIVE_SUPPORT" == "yes"]]: then
    #    make_command+=" GPU_NATIVE_SUPPORT=1"
    #fi

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

# INFO: new function to compile libraries, will replace the previous one
compile_all_libraries_tui() {
    make clean

    local count="${LIB_COUNT:-0}"
    if ! [[ "$count" =~ ^[0-9]+$ ]] || (( count == 0 )); then
        warning "LIB_COUNT is zero or unset; nothing to compile"
        return 0
    fi

    local mk_debug=0
    local mk_instr=0
    [[ "$DEBUG_MODE" == "yes" ]] && mk_debug=1
    [[ "$INSTRUMENT" == "yes" ]] && mk_instr=1

    for (( i=0; i<count; i++ )); do
        export_lib_identity "$i"   # sets: MPI_LIB, MPI_LIB_VERSION, PICOCC

        local libmods="$(_get_var "LIB_${i}_MODULES")"
        local load_type="$(_get_var "LIB_${i}_LOAD_TYPE")"
        local lib_standard="$(_get_var "LIB_${i}_STANDARD")"
        local lib_standard_lc="${lib_standard,,}"

        # CUDA build only if: GPU_AWARENESS=yes ∧ any GPU>0 ∧ GPU_LIB=cuda
        local gaw="$(_get_var "LIB_${i}_GPU_AWARENESS")"
        local gns="$(_get_var "LIB_${i}_GPU_NATIVE_SUPPORT")"
        local gpn="$(_get_var "LIB_${i}_GPU_PER_NODE")"
        local gpu_lib_raw="$(_get_var "LIB_${i}_GPU_LIB")"
        local gpu_lib="${gpu_lib_raw,,}"   # lowercase

        local any_gpu_nonzero=0
        if [[ -n "$gpn" ]]; then
            for n in ${gpn//,/ }; do
                if [[ "$n" =~ ^[0-9]+$ ]] && (( n > 0 )); then any_gpu_nonzero=1; break; fi
            done
        fi
        local need_cuda_build=0
        if [[ "$gaw" == "yes" && $any_gpu_nonzero -eq 1 && "$gpu_lib" == "cuda" ]]; then
            need_cuda_build=1
        fi
        local need_nccl_build=0
        if [[ "$lib_standard_lc" == "nccl" || "$MPI_LIB" == "NCCL" ]]; then
            need_nccl_build=1
        fi

        # NCCL sources are C-only and require proper MPI wrapper link flags.
        if (( need_nccl_build )) && [[ "${PICOCC##*/}" == "nvcc" ]] && command -v mpicc >/dev/null 2>&1; then
            [[ "$DEBUG_MODE" == "yes" ]] && inform "[lib $i] NCCL standard detected: overriding compiler nvcc -> mpicc"
            export PICOCC="mpicc"
        fi

        local this_instrument=0
        if [[ "$mk_instr" -eq 1 ]]; then
            inform "Instrumented build requested for library $i"
            this_instrument=1
        fi

        # -------- Activate per-lib context (module | set_env | default) --------
        trace_env_snapshot "lib $i BEFORE apply/load"
        activate_lib_context "$i" || { error "Failed to activate context for library $i"; return 1; }
        [[ "$DEBUG_MODE" == "yes" ]] && inform "[lib $i] LOAD_TYPE=${load_type:-<unset>} modules=${libmods:-<none>}"
        trace_env_snapshot "lib $i AFTER apply/load"
        trace_compiler_wrapper "$PICOCC"

        # Per-lib output dirs
        local OUT_BIN="$PICO_DIR/bin/lib_${i}"
        local OUT_LIB="$PICO_DIR/lib/lib_${i}"
        local OUT_OBJ="$PICO_DIR/obj/lib_${i}"
        mkdir -p "$OUT_BIN" "$OUT_LIB" "$OUT_OBJ" || true

        # Single make call; top-level Makefile: all -> force_rebuild + build
        local mk="make -C \"$PICO_DIR\" all"
        mk+=" BIN_DIR=\"$OUT_BIN\" LIB_DIR=\"$OUT_LIB\""
        mk+=" PICO_CORE_OBJ_DIR=\"$OUT_OBJ/pico_core\" PICO_CORE_OBJ_DIR_CUDA=\"$OUT_OBJ/pico_core_cuda\""
        mk+=" LIB_OBJ_DIR=\"$OUT_OBJ/lib\" LIB_OBJ_DIR_CUDA=\"$OUT_OBJ/lib_cuda\""
        mk+=" DEBUG=$mk_debug"
        mk+=" PICO_INSTRUMENT=$this_instrument"
        if (( need_nccl_build )); then
            mk+=" PICO_NCCL=1"
        elif (( need_cuda_build )); then
            mk+=" PICO_MPI_CUDA_AWARE=1"
        fi
        if (( need_cuda_build )) && [[ "$gns" == "yes" ]]; then mk+=" GPU_NATIVE_SUPPORT=1"; fi

        if [[ "$DEBUG_MODE" == "yes" ]]; then
            echo -e "${BLUE}>>> [lib ${i}] Invoking build${NC}"
            echo "  MPI_LIB:         ${MPI_LIB}"
            echo "  MPI_LIB_VERSION: ${MPI_LIB_VERSION}"
            echo "  PICOCC:          ${PICOCC}"
            echo "  CMD:             PICOCC=\"${PICOCC}\" MPI_LIB=\"${MPI_LIB}\" ${mk}"
        fi

        if [[ "$DRY_RUN" == "yes" ]]; then
            inform "Would run (lib $i): PICOCC=\"$PICOCC\" MPI_LIB=\"$MPI_LIB\" $mk"
        else
            PICOCC="$PICOCC" MPI_LIB="$MPI_LIB" eval "$mk" || { error "Compilation failed for library $i"; restore_lib_context "$i"; return 1; }
        fi

        # Export the per-lib execs if they exist
        [[ -f "$OUT_BIN/pico_core" ]] && export "LIB_${i}_PICO_EXEC_CPU=$OUT_BIN/pico_core" && trace_ldd "$OUT_BIN/pico_core"
        [[ -f "$OUT_BIN/pico_core_cuda" ]] && export "LIB_${i}_PICO_EXEC_GPU=$OUT_BIN/pico_core_cuda" && trace_ldd "$OUT_BIN/pico_core_cuda"
        [[ -f "$OUT_BIN/pico_core_nccl" ]] && export "LIB_${i}_PICO_EXEC_GPU=$OUT_BIN/pico_core_nccl" && trace_ldd "$OUT_BIN/pico_core_nccl"

        # Unload only this library's modules (keep general ones)
        restore_lib_context "$i" || true
    done

    success "Per-library compilation completed."
    return 0
}
export -f compile_all_libraries_tui

###############################################################################
# Calculate SLURM values (max tasks-per-node, max gpus-per-node, any gpu-aware)
###############################################################################
# Exports global caps across all libraries:
#   ANY_GPU_AWARE="yes|no"
#   MAX_TASKS_PER_NODE=<max of all CPU TASKS_PER_NODE and GPU_PER_NODE values>
#   MAX_GPU_PER_NODE=<max of all GPU_PER_NODE values among GPU-aware libs>
#   SLURM_TASKS_PER_NODE=max($MAX_TASKS_PER_NODE, $MAX_GPU_PER_NODE)
compute_slurm_caps_from_libs() {
    local count="${LIB_COUNT:-0}"
    local max_tpn=""
    local max_gpn=""
    local any_gpu="no"

    # Guard: nothing to do if LIB_COUNT not set or zero
    if ! [[ "$count" =~ ^[0-9]+$ ]] || (( count == 0 )); then
        export ANY_GPU_AWARE="no"
        return 0
    fi

    for (( i=0; i<count; i++ )); do
        local tpn_csv="$(_get_var "LIB_${i}_TASKS_PER_NODE")"
        if [[ -n "$tpn_csv" ]]; then
            local t
            for t in ${tpn_csv//,/ }; do
                [[ "$t" =~ ^[0-9]+$ ]] || continue
                if [[ -z "$max_tpn" || "$t" -gt "$max_tpn" ]]; then
                    max_tpn="$t"
                fi
            done
        fi

        local gaw_val="$(_get_var "LIB_${i}_GPU_AWARENESS")"
        local gpn_csv="$(_get_var "LIB_${i}_GPU_PER_NODE")"
        if [[ "$gaw_val" == "yes" && -n "$gpn_csv" ]]; then
            any_gpu="yes"
            local g
            for g in ${gpn_csv//,/ }; do
                [[ "$g" =~ ^[0-9]+$ ]] || continue
                if [[ -z "$max_gpn" || "$g" -gt "$max_gpn" ]]; then
                    max_gpn="$g"
                fi
                if [[ -z "$max_tpn" || "$g" -gt "$max_tpn" ]]; then
                    max_tpn="$g"
                fi
            done
        fi
    done

    export ANY_GPU_AWARE="$any_gpu"
    [[ -n "$max_gpn" ]] && export MAX_GPU_PER_NODE="$max_gpn"
    [[ -n "$max_tpn" ]] && export MAX_TASKS_PER_NODE="$max_tpn"

    local _tpn="${MAX_TASKS_PER_NODE:-0}"
    local _gpn="${MAX_GPU_PER_NODE:-0}"
    local _slurm_tpn=$_tpn
    (( _gpn > _slurm_tpn )) && _slurm_tpn=$_gpn
    if (( _slurm_tpn > 0 )); then
        export SLURM_TASKS_PER_NODE="$_slurm_tpn"
    fi

    return 0
}
export -f compute_slurm_caps_from_libs

###############################################################################
# Sanity checks
###############################################################################
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
    local command="$RUN $RUNFLAGS -n $MPI_TASKS $PICO_EXEC $size $iter $algo $type"

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
nccl_selector_from_algo_name() {
    local algo="${1,,}"

    case "$algo" in
        auto|default|auto_nccl|default_nccl)
            echo ""
            return 0
            ;;
    esac

    # Accept explicit protocol suffixes in algorithm aliases, e.g. ring_nccl_ll.
    case "$algo" in
        *_simple) algo="${algo%_simple}" ;;
        *_ll128)  algo="${algo%_ll128}" ;;
        *_ll)     algo="${algo%_ll}" ;;
    esac

    if [[ "$algo" == *_nccl ]]; then
        algo="${algo%_nccl}"
    fi
    echo "$algo"
}

nccl_algo_token() {
    local selector="$1"

    case "${selector,,}" in
        ring)         echo "Ring" ;;
        tree)         echo "Tree" ;;
        collnetdirect) echo "CollnetDirect" ;;
        collnetchain)  echo "CollnetChain" ;;
        nvls)         echo "NVLS" ;;
        nvlstree)     echo "NVLSTree" ;;
        pat)          echo "PAT" ;;
        *)            echo "$selector" ;;
    esac
}

is_nccl_ring_algo() {
    [[ "$MPI_LIB" == "NCCL" ]] || return 1

    local selector
    selector="$(nccl_selector_from_algo_name "$1")"
    [[ "${selector,,}" == "ring" ]]
}

nccl_protocol_from_algo_name() {
    local algo="${1,,}"

    case "$algo" in
        *_simple) echo "Simple" ;;
        *_ll128)  echo "LL128" ;;
        *_ll)     echo "LL" ;;
        *)        echo "" ;;
    esac
}

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
        "NCCL")
            # NCCL algorithm forcing is controlled through NCCL_ALGO at runtime.
            local selector
            selector="$(nccl_selector_from_algo_name "$algo")"
            if [[ -z "$selector" ]]; then
                unset NCCL_ALGO
                success "Clearing NCCL_ALGO for algorithm $algo (runtime auto-selection)..."
            else
                local nccl_algo
                nccl_algo="$(nccl_algo_token "$selector")"
                [[ "$nccl_algo" == "$selector" ]] && warning "Unknown NCCL selector '$selector' for '$algo'; using raw value."
                export NCCL_ALGO="$nccl_algo"
                success "Setting NCCL_ALGO=$NCCL_ALGO for algorithm $algo..."
            fi
            ;;
        *)
            echo "Error: Unsupported MPI_LIB value: $MPI_LIB" >&2
            return 1
            ;;
    esac
}
export -f update_algorithm

# This wrapper handles running benchmarks with different NCCL protocols
# if the algorithm is a ring variant and the library is NCCL.
# For non-NCCL or non-ring algorithms, it just runs the benchmark once.
# TODO: Clean this wrapper and put this inside the run_bench
run_bench_with_nccl_protocols() {
    local size="$1" algo="$2" type="$3"

    if is_nccl_ring_algo "$algo"; then
        local explicit_proto
        explicit_proto="$(nccl_protocol_from_algo_name "$algo")"

        if [[ -n "$explicit_proto" ]]; then
            export NCCL_PROTO="$explicit_proto"
            [[ "$DEBUG_MODE" == "yes" ]] && inform "NCCL ring protocol: $NCCL_PROTO"
            run_bench "$size" "$algo" "$type"
            return 0
        fi

        local proto
        for proto in Simple LL LL128; do
            local proto_suffix="${proto,,}"
            local bench_algo="${algo}_${proto_suffix}"
            export NCCL_PROTO="$proto"
            [[ "$DEBUG_MODE" == "yes" ]] && inform "NCCL ring protocol: $NCCL_PROTO"
            run_bench "$size" "$bench_algo" "$type"
        done
    else
        [[ "$MPI_LIB" == "NCCL" ]] && unset NCCL_PROTO
        run_bench "$size" "$algo" "$type"
    fi
}
export -f run_bench_with_nccl_protocols

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
                        run_bench_with_nccl_protocols $size $algo $type
                    done
                done
            else
                for type in ${TYPES//,/ }; do
                    run_bench_with_nccl_protocols $size $algo $type
                done
            fi
        done
        ((i++))
    done
}
export -f run_all_tests

###############################################################################
# TEMPORARY TRANSLATION LAYER TO ALLOW FOR TUI FILE, to be removed
###############################################################################
_get_var() {
  local __name="$1"
  printf '%s' "${!__name-}"
}
export -f _get_var

_var_declared() {
  local __name="$1"
  declare -p "$__name" &>/dev/null
}
export -f _var_declared

_build_all_no_array_literal() {
  local -n __names_ref=$1
  local n i
  n=${#__names_ref[@]}
  local out="("
  for (( i=0; i<n; i++ )); do
    out+="no "
  done
  out="${out%% }"
  out+=")"
  printf '%s' "$out"
}
export -f _build_all_no_array_literal


load_other_env_var(){
    if [[ "$MPI_LIB" == "OMPI" ]]; then
        export OMPI_MCA_coll_hcoll_enable=0
        export OMPI_MCA_coll_tuned_use_dynamic_rules=1
        if [ "$GPU_AWARENESS" == "no" ]; then
            export OMPI_MCA_btl="^smcuda"
            export OMPI_MCA_mpi_cuda_support=0
        else
            export OMPI_MCA_btl=""
            export OMPI_MCA_mpi_cuda_support=1
        fi
    fi
}
export -f load_other_env_var


__csv_to_array() {
  local csv="$1"; local -n out_ref=$2
  out_ref=()
  [[ -z "$csv" ]] && return 0
  local tok
  IFS=',' read -r -a out_ref <<< "$csv"
  # trim whitespace around tokens
  for (( __i=0; __i<${#out_ref[@]}; __i++ )); do
    tok="${out_ref[__i]}"
    tok="${tok##+([[:space:]])}"; tok="${tok%%+([[:space:]])}"
    out_ref[__i]="$tok"
  done
}
export -f __csv_to_array



###############################################################################
# Per-library context: apply/restore (module | set_env | default)
###############################################################################
activate_lib_context() {
    local i="$1"
    local load_type="$(_get_var "LIB_${i}_LOAD_TYPE")"
    local libmods="$(_get_var "LIB_${i}_MODULES")"

    case "$load_type" in
        module)
            [[ -n "$libmods" ]] && load_modules "$libmods" || { error "LIB_${i}: no MODULES to load"; return 1; }
            ;;
        set_env)
            apply_lib_env "$i" || { error "LIB_${i}: set_env requested but no env vars found"; return 1; }
            ;;
        ""|default)
            : # nothing
            ;;
        *)
            warning "LIB_${i}: unknown LOAD_TYPE='$load_type'; proceeding without env change"
            ;;
    esac
    return 0
}
export -f activate_lib_context

restore_lib_context() {
    local i="$1"
    local load_type="$(_get_var "LIB_${i}_LOAD_TYPE")"
    local libmods="$(_get_var "LIB_${i}_MODULES")"

    case "$load_type" in
        module)   [[ -n "$libmods" ]] && unload_modules "$libmods" || true ;;
        set_env)  restore_lib_env "$i" || true ;;
        ""|default) : ;;
        *) : ;;
    esac
    return 0
}
export -f restore_lib_context

###############################################################################
# Per-library exports (compiler/lib identity) used by helpers & sanity prints
###############################################################################
export_lib_identity() {
    local i="$1"
    export MPI_LIB="$(_get_var "LIB_${i}_MPI_LIB")"
    export MPI_LIB_VERSION="$(_get_var "LIB_${i}_MPI_LIB_VERSION")"
    export PICOCC="$(_get_var "LIB_${i}_PICOCC")"
    export GPU_LIB="$(_get_var "LIB_${i}_GPU_LIB")"
    export GPU_LIB_VERSION="$(_get_var "LIB_${i}_GPU_LIB_VERSION")"
}
export -f export_lib_identity

###############################################################################
# Prepare per-collective legacy-compatible vars (ALGOS, SKIP, IS_SEGMENTED, CVARS?)
# Sets:
#   COLLECTIVE_TYPE, ALGOS, SKIP, IS_SEGMENTED (array), CVARS (array if MPICH-family)
# Returns 0 on success, 1 to skip collective (e.g. no algorithms)
###############################################################################
prepare_collective_vars() {
    local i="$1" coll="$2"
    local COLL_UPPER="${coll^^}"

    export COLLECTIVE_TYPE="$COLL_UPPER"

    local ALGS_CSV="$(_get_var "LIB_${i}_${COLL_UPPER}_ALGORITHMS")"
    if [[ -z "$ALGS_CSV" ]]; then
        warning "LIB_${i} ${coll}: no algorithms; skipping"
        return 1
    fi
    IFS=',' read -r -a _ALG_NAMES <<< "$ALGS_CSV"
    export ALGOS="${ALGS_CSV//,/ }"

    local SKIPS_CSV="$(_get_var "LIB_${i}_${COLL_UPPER}_ALGORITHMS_SKIP")"
    IFS=',' read -r -a _SKIP_FLAGS <<< "$SKIPS_CSV"
    export SKIP="${SKIPS_CSV//,/ }"

    local SEG_CSV="$(_get_var "LIB_${i}_${COLL_UPPER}_ALGORITHMS_IS_SEGMENTED")"
    if [[ -z "$SEG_CSV" ]]; then
        IS_SEGMENTED=()
        for ((k=0; k<${#_ALG_NAMES[@]}; k++)); do IS_SEGMENTED+=(no); done
    else
        IFS=',' read -r -a IS_SEGMENTED <<< "$SEG_CSV"
        if (( ${#IS_SEGMENTED[@]} < ${#_ALG_NAMES[@]} )); then
            for ((k=${#IS_SEGMENTED[@]}; k<${#_ALG_NAMES[@]}; k++)); do IS_SEGMENTED+=(no); done
        elif (( ${#IS_SEGMENTED[@]} > ${#_ALG_NAMES[@]} )); then
            IS_SEGMENTED=( "${IS_SEGMENTED[@]:0:${#_ALG_NAMES[@]}}" )
        fi
    fi

    unset CVARS
    if [[ "$MPI_LIB" == "MPICH" || "$MPI_LIB" == "CRAY_MPICH" ]]; then
        local CVARS_CSV="$(_get_var "LIB_${i}_${COLL_UPPER}_ALGORITHMS_CVARS")"
        if [[ -z "$CVARS_CSV" ]]; then
            error "LIB_${i} ${coll}: MPICH-family requires ${COLL_UPPER}_ALGORITHMS_CVARS"
            return 1
        fi
        IFS=',' read -r -a CVARS <<< "$CVARS_CSV"
    fi

    return 0
}
export -f prepare_collective_vars

###############################################################################
# Run one test “mode” (CPU or GPU) for a given lib/collective & concurrency
# Inputs (exports expected to be set by caller):
#   - COLLECTIVE_TYPE, ALGOS, SKIP, IS_SEGMENTED[], (CVARS[])
#   - N_NODES, TYPES, SIZES, DEBUG_MODE, DRY_RUN
#   - PICO_EXEC, CURRENT_TASKS_PER_NODE, MPI_TASKS, GPU_AWARENESS
# Side-effects:
#   - generates metadata (unless debug/dry)
#   - prints sanity and calls run_all_tests
###############################################################################
run_mode_once() {
    local -n _iter_ref=$1  # pass-by-name: iter variable name

    load_other_env_var

    if [[ "$DEBUG_MODE" == "no" && "$DRY_RUN" == "no" ]]; then
        export DATA_DIR="$OUTPUT_DIR/${_iter_ref}"
        mkdir -p "$DATA_DIR"
        python3 "$PICO_DIR/results/generate_metadata.py" "${_iter_ref}" || return 1
    fi

    print_sanity_checks
    run_all_tests
    ((_iter_ref++))
    return 0
}
export -f run_mode_once

###############################################################################
# Per-lib: run GPU loop (if any) and CPU loop
# Chooses PICO_EXEC from per-lib compiled paths with fallback to global ones
###############################################################################
run_collective_for_lib() {
    local i="$1" coll="$2"
    local iter_name="$3" # name of the iter variable to bump

    # 1) Make legacy compatibles for this collective
    prepare_collective_vars "$i" "$coll" || return 0

    # 2) Resolve execs compiled for this library
    local LIB_PICO_CPU="$(_get_var "LIB_${i}_PICO_EXEC_CPU")"
    local LIB_PICO_GPU="$(_get_var "LIB_${i}_PICO_EXEC_GPU")"

    # 3) GPU loop (if awareness yes)
    local gaw="$(_get_var "LIB_${i}_GPU_AWARENESS")"
    local gpn_csv="$(_get_var "LIB_${i}_GPU_PER_NODE")"
    if [[ "$gaw" == "yes" && -n "$gpn_csv" ]]; then
        for n_gpu in ${gpn_csv//,/ }; do
            [[ "$n_gpu" =~ ^[0-9]+$ ]] || continue
            (( n_gpu == 0 )) && continue
            export GPU_AWARENESS="yes"
            export CURRENT_TASKS_PER_NODE=$n_gpu
            export MPI_TASKS=$(( N_NODES * n_gpu ))
            export PICO_EXEC="${LIB_PICO_GPU:-$PICO_EXEC_GPU}"
            run_mode_once "$iter_name" || return 1
        done
    fi

    # 4) CPU loop
    local tpn_csv="$(_get_var "LIB_${i}_TASKS_PER_NODE")"
    if [[ -n "$tpn_csv" ]]; then
        for ntasks in ${tpn_csv//,/ }; do
            [[ "$ntasks" =~ ^[0-9]+$ ]] || continue
            export GPU_AWARENESS="no"
            export CURRENT_TASKS_PER_NODE=$ntasks
            export MPI_TASKS=$(( N_NODES * CURRENT_TASKS_PER_NODE ))
            export PICO_EXEC="${LIB_PICO_CPU:-$PICO_EXEC_CPU}"

            if [[ -n "$FORCE_TASKS" ]]; then
                export MPI_TASKS=$FORCE_TASKS
                export CURRENT_TASKS_PER_NODE=$(( FORCE_TASKS / N_NODES ))
            fi

            run_mode_once "$iter_name" || return 1

            if [[ -n "$FORCE_TASKS" ]]; then
                warning "--ntasks set; skipping remaining ntasks-per-node for this lib/collective"
                break
            fi
        done
    fi
    return 0
}
export -f run_collective_for_lib

###############################################################################
# Per-lib: drive all collectives for a library index
###############################################################################
run_library_tui() {
    local i="$1"
    local iter_name="$2"  # name of outer iter variable

    # Export per-lib identity so helpers & prints show correct lib
    export_lib_identity "$i" || return 1

    # Activate the library runtime context
    activate_lib_context "$i" || return 1

    # Per-lib collectives list
    local lib_collectives="$(_get_var "LIB_${i}_COLLECTIVES")"
    if [[ -z "$lib_collectives" ]]; then
        warning "LIB_${i}: no COLLECTIVES; skipping"
        restore_lib_context "$i"
        return 0
    fi

    # Info
    local name="$(_get_var "LIB_${i}_NAME")"
    local ver="$(_get_var "LIB_${i}_VERSION")"
    inform "▶ Running library $i: ${name:-<unknown>} ${ver:-}"

    for coll in ${lib_collectives//,/ }; do
        run_collective_for_lib "$i" "$coll" "$iter_name" || { restore_lib_context "$i"; return 1; }
    done

    restore_lib_context "$i"
    return 0
}
export -f run_library_tui




##################################################################################
# CLI mode functions (outer loops, env prep, metadata, sanity, run)
##################################################################################
# WARN: TO BE DEPRECATED

# Decide awareness / tasks / exec for a single n_gpu value
cli_set_awareness_and_tasks() {
    local n_gpu="$1"
    if [[ "$n_gpu" == "0" ]]; then
        export GPU_AWARENESS="no"
        export PICO_EXEC="$PICO_EXEC_CPU"
        # CURRENT_TASKS_PER_NODE will be set by the CPU inner loop
    else
        export GPU_AWARENESS="yes"
        export CURRENT_TASKS_PER_NODE="$n_gpu"
        export MPI_TASKS=$(( N_NODES * CURRENT_TASKS_PER_NODE ))
        export PICO_EXEC="$PICO_EXEC_GPU"
    fi
}
export -f cli_set_awareness_and_tasks

# Prepare env for one iteration (parse + load test env)
cli_prepare_iteration_env() {
    # parse JSON -> TEST_ENV + export vars
    python3 "$PICO_DIR/config/parse_test.py" || return 1
    source "$TEST_ENV"
    load_other_env_var
    return 0
}
export -f cli_prepare_iteration_env

# Create metadata dir (if enabled) and call generator; name-ref for iter
cli_prepare_metadata() {
    local -n _iter_ref=$1
    if [[ "$DEBUG_MODE" == "no" && "$DRY_RUN" == "no" ]]; then
        export DATA_DIR="$OUTPUT_DIR/${_iter_ref}"
        mkdir -p "$DATA_DIR"
        if ! python3 "$PICO_DIR/results/generate_metadata.py" "${_iter_ref}"; then
            error "Metadata generation failed for iteration ${_iter_ref}. For GPU-aware runs, ensure GPU_LIB and GPU_LIB_VERSION are exported."
            return 1
        fi
        success "📂 Metadata of $DATA_DIR created"
    fi
    return 0
}
export -f cli_prepare_metadata

# One full iteration run (env already set): print summary + run tests; name-ref iter++
cli_run_one_iteration() {
    local -n _iter_ref=$1
    print_sanity_checks
    run_all_tests
    ((_iter_ref++))
}
export -f cli_run_one_iteration

# CPU inner loop for one config; pass the *name* of the iter variable
cli_run_cpu_set() {
    local iter_name="$1"          # raw name of the iter var (e.g., "iter")
    local -n _iter_ref="$iter_name"  # local nameref for convenience

    for ntasks in ${TASKS_PER_NODE//,/ }; do
        export CURRENT_TASKS_PER_NODE="$ntasks"
        export MPI_TASKS=$(( N_NODES * CURRENT_TASKS_PER_NODE ))

        if [[ -n "$FORCE_TASKS" ]]; then
            export MPI_TASKS="$FORCE_TASKS"
            export CURRENT_TASKS_PER_NODE=$(( FORCE_TASKS / N_NODES ))
        fi

        cli_prepare_iteration_env || { error "Failed to prepare test env (CPU)"; return 1; }
        success "📄 Config ${TEST_CONFIG} parsed (CPU, ntasks=${CURRENT_TASKS_PER_NODE})"

        # pass the ORIGINAL name downstream
        cli_prepare_metadata "$iter_name" || return 1
        cli_run_one_iteration "$iter_name"

        if [[ -n "$FORCE_TASKS" ]]; then
            warning "--ntasks is set, skipping possible --tasks-per-node values"
            break
        fi
    done
}
export -f cli_run_cpu_set

# GPU path for one n_gpu; pass the *name* of the iter variable
cli_run_gpu_once() {
    local iter_name="$1"           # raw name
    local -n _iter_ref="$iter_name"

    cli_prepare_iteration_env || { error "Failed to prepare test env (GPU)"; return 1; }
    success "📄 Config ${TEST_CONFIG} parsed (GPU, gpus per node=${CURRENT_TASKS_PER_NODE})"

    cli_prepare_metadata "$iter_name" || return 1
    cli_run_one_iteration "$iter_name"
}
export -f cli_run_gpu_once
