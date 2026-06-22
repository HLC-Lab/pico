#!/bin/bash
# High-level entry point that validates CLI flags and submits jobs to SLURM.
# Usage: scripts/submit_wrapper.sh [options]
# Delegates execution to scripts/orchestrator.sh or runs locally when requested.

source scripts/utils.sh

# 1. Set default values for the variables (are defined in `utils.sh`)
if [[ -n "${BASH_SOURCE[0]}" ]]; then
    export PICO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
else
  warning "BASH_SOURCE is not set.""Using current working directory (pwd) as fallback.""This may cause issues in some environments"
    export PICO_DIR="$(pwd)"
fi



export TASKS_PER_NODE=$DEFAULT_TASKS_PER_NODE
export COMPILE_ONLY=$DEFAULT_COMPILE_ONLY
export TIMESTAMP=$DEFAULT_TIMESTAMP
export TYPES=$DEFAULT_TYPES
export SIZES=$DEFAULT_SIZES
export SEGMENT_SIZES=$DEFAULT_SEGMENT_SIZES
export COLLECTIVES=$DEFAULT_COLLECTIVES

export GPU_AWARENESS=$DEFAULT_GPU_AWARENESS
export GPU_PER_NODE=$DEFAULT_GPU_PER_NODE

export OUTPUT_LEVEL=$DEFAULT_OUTPUT_LEVEL
export COMPRESS=$DEFAULT_COMPRESS
export DELETE=$DEFAULT_DELETE
export NOTES=$DEFAULT_NOTES
export INSTRUMENT=$DEFAULT_INSTRUMENT

export TEST_TIME=$DEFAULT_TEST_TIME
export EXCLUDE_NODES=$DEFAULT_EXCLUDE_NODES
export JOB_DEP=$DEFAULT_JOB_DEP
export OTHER_SLURM_PARAMS=$DEFAULT_OTHER_SLURM_PARAMS
export SHOW_ENV=$DEFAULT_SHOW_ENV

export DEBUG_MODE=$DEFAULT_DEBUG_MODE
export DRY_RUN=$DEFAULT_DRY_RUN
export INTERACTIVE=$DEFAULT_INTERACTIVE

# 2. Parse and validate command line arguments
parse_cli_args "$@"

if [[ -n "$TUI_FILE" ]]; then
    success "Using TUI file: $TUI_FILE"
    warning "Ignoring other command line arguments"
    [[ -f "$TUI_FILE" ]] || { error "TUI file '$TUI_FILE' does not exist."; exit 1; }
    source $TUI_FILE
else
    source_environment || exit 1
    validate_args || exit 1
fi

[[ -z "$PICO_ACCOUNT" && "$LOCATION" != "local" ]] && warning "PICO_ACCOUNT environment variable not set, please export it with your slurm project's name" && exit 0

# 5. Load required modules
# TUI: only load general modules here; per-library modules are loaded during compilation
# CLI: load all modules here
if [[ -n "$TUI_FILE" ]]; then
    load_modules "$GENERAL_MODULES" || exit 1
else
    load_modules || exit 1
fi

# 6. Compile code. If `$DEBUG_MODE` is `yes`, debug flags will be added
if [[ -n "$TUI_FILE" ]]; then
    compile_all_libraries_tui || exit 1
else
    compile_code || exit 1
fi
[[ "$COMPILE_ONLY" == "yes" ]] && success "Compile only mode. Exiting..." && exit 0


# 7. Activate the virtual environment, install Python packages if not presents
activate_virtualenv || exit 1
success "Virtual environment activated."

# 8. Defines env dependant variables
export ALGORITHM_CONFIG_FILE="$PICO_DIR/config/algorithm_config.json" # WARN: to be deprecated
export LOCATION_DIR="$PICO_DIR/results/$LOCATION"
export OUTPUT_DIR="$PICO_DIR/results/$LOCATION/$TIMESTAMP"

if [[ -z "$TUI_FILE" ]]; then
    # CLI/back-compat: single-binary layout
    export PICO_EXEC_CPU=$PICO_DIR/bin/pico_core
    [[ "$GPU_AWARENESS" == "yes" ]] && export PICO_EXEC_GPU=$PICO_DIR/bin/pico_core_cuda
fi

export ALGO_CHANGE_SCRIPT=$PICO_DIR/selector/change_dynamic_rules.py
export DYNAMIC_RULE_FILE=$PICO_DIR/selector/ompi_dynamic_rules.txt

# 9. Create output directories if not in debug mode or dry run
if [[ "$DEBUG_MODE" == "no" && "$DRY_RUN" == "no" ]]; then
    success "📂 Creating output directories..."
    mkdir -p "$LOCATION_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

# 10. Submit the job.
if [[ "$LOCATION" == "local" ]]; then
    scripts/orchestrator.sh
else
    if [[ -n "$TUI_FILE" ]]; then
        compute_slurm_caps_from_libs
        [[ "$DEBUG_MODE" == "yes" ]] && inform "Derived submission caps (TUI):" \
              "Max tasks per node: ${MAX_TASKS_PER_NODE:-<none>}" \
              "Max GPUs per node:  ${MAX_GPU_PER_NODE:-<none>}" \
              "Any GPU aware:      ${ANY_GPU_AWARE}"
    fi

    SLURM_PARAMS=" --account $PICO_ACCOUNT --nodes $N_NODES --time $TEST_TIME --partition $PARTITION"

    if [[ -n "$QOS" ]]; then
        SLURM_PARAMS+=" --qos $QOS"
        [[ -n "$QOS_TASKS_PER_NODE" ]] && export SLURM_TASKS_PER_NODE="$QOS_TASKS_PER_NODE"
        [[ -n "$QOS_GRES" ]] && GRES="$QOS_GRES"
    fi

    if [[ "${ANY_GPU_AWARE:-$GPU_AWARENESS}" == "yes" &&  -n "$MAX_GPU_PER_NODE" ]]; then
        [[ -z "$GRES" ]] && GRES="gpu:$MAX_GPU_PER_NODE"
        SLURM_PARAMS+=" --gpus-per-node $MAX_GPU_PER_NODE"
    fi

    if [[ "$LOCATION" != "leonardo" || -n "$SLURM_TASKS_PER_NODE" ]]; then
        if [[ -n "$FORCE_TASKS" && -z "$QOS_TASKS_PER_NODE" ]]; then
            SLURM_PARAMS+=" --ntasks $FORCE_TASKS"
        else
            if [[ -n "$SLURM_TASKS_PER_NODE" ]]; then
                SLURM_PARAMS+=" --ntasks-per-node $SLURM_TASKS_PER_NODE"
            fi
        fi
    fi

    [[ -n "$GRES" ]] && SLURM_PARAMS+=" --gres=$GRES"
    [[ -n "$EXCLUDE_NODES" ]] && SLURM_PARAMS+=" --exclude $EXCLUDE_NODES"
    [[ -n "$JOB_DEP" ]] && SLURM_PARAMS+=" --dependency=afterany:$JOB_DEP"
    [[ -n "$OTHER_SLURM_PARAMS" ]] && SLURM_PARAMS+=" $OTHER_SLURM_PARAMS"

    if [[ "$INTERACTIVE" == "yes" ]]; then
        export SLURM_PARAMS="$SLURM_PARAMS"
        inform "Salloc with parameters: $SLURM_PARAMS"
        salloc $SLURM_PARAMS
    else
        [[ "$DEBUG_MODE" == "no" && "$DRY_RUN" == "no" ]] && \
          SLURM_PARAMS+=" --exclusive --output=$OUTPUT_DIR/slurm_%j.out --error=$OUTPUT_DIR/slurm_%j.err" || \
          SLURM_PARAMS+=" --output=debug_%j.out"
        export SLURM_PARAMS="$SLURM_PARAMS"
        inform "Sbatching job with parameters: $SLURM_PARAMS"
        sbatch $SLURM_PARAMS "$PICO_DIR/scripts/orchestrator.sh"
    fi
fi
