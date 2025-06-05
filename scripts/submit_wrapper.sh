#!/bin/bash

source scripts/utils.sh

# 1. Set default values for the variables (are defined in `utils.sh`)
if [[ -n "${BASH_SOURCE[0]}" ]]; then
    export SWING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
else
    echo "Warning: BASH_SOURCE is not set. Using current working directory as fallback."
    export SWING_DIR="$(pwd)"
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

# 3. Set the location-specific configuration (defined in `config/environment/$LOCATION.sh`)
source_environment || exit 1

# 4. Validate all the given arguments
validate_args || exit 1

# 5. Load required modules (defined in `config/environment/$LOCATION.sh`)
load_modules || exit 1

# 6. Activate the virtual environment, install Python packages if not presents
if [[ "$COMPILE_ONLY" == "no" ]]; then
    activate_virtualenv || exit 1
    success "Virtual environment activated."
fi

# 7. Compile code. If `$DEBUG_MODE` is `yes`, debug flags will be added
compile_code || exit 1
[[ "$COMPILE_ONLY" == "yes" ]] && success "Compile only mode. Exiting..." && exit 0

# 8. Defines env dependant variables
export ALGORITHM_CONFIG_FILE="$SWING_DIR/config/algorithm_config.json"
export LOCATION_DIR="$SWING_DIR/results/$LOCATION"
export OUTPUT_DIR="$SWING_DIR/results/$LOCATION/$TIMESTAMP"
export BENCH_EXEC_CPU=$SWING_DIR/bin/bench
[[ "$GPU_AWARENESS" == "yes" ]] && export BENCH_EXEC_GPU=$SWING_DIR/bin/bench_cuda
export ALGO_CHANGE_SCRIPT=$SWING_DIR/selector/change_dynamic_rules.py
export DYNAMIC_RULE_FILE=$SWING_DIR/selector/ompi_dynamic_rules.txt

# 9. Create output directories if not in debug mode or dry run
if [[ "$DEBUG_MODE" == "no" && "$DRY_RUN" == "no" ]]; then
    success "📂 Creating output directories..."
    mkdir -p "$LOCATION_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

# 10. Submit the job.
if [[ "$LOCATION" == "local" ]]; then
    scripts/run_test_suite.sh
else
    SLURM_PARAMS="--account $ACCOUNT --nodes $N_NODES --time $TEST_TIME --partition $PARTITION"

    if [[ -n "$QOS" ]]; then
        SLURM_PARAMS+=" --qos $QOS"
        [[ -n "$QOS_TASKS_PER_NODE" ]] && export SLURM_TASKS_PER_NODE="$QOS_TASKS_PER_NODE"
        [[ -n "$QOS_GRES" ]] && GRES="$QOS_GRES"
    fi

    if [[ "$GPU_AWARENESS" == "yes" ]]; then
        [[ -z "$GRES" ]] && GRES="gpu:$MAX_GPU_TEST"
        SLURM_PARAMS+=" --gpus-per-node $MAX_GPU_TEST"
    fi

    [[ -n "$FORCE_TASKS" && -z "$QOS_TASKS_PER_NODE" ]] && SLURM_PARAMS+=" --ntasks $FORCE_TASKS" || SLURM_PARAMS+=" --ntasks-per-node $SLURM_TASKS_PER_NODE"
    [[ -n "$GRES" ]] && SLURM_PARAMS+=" --gres=$GRES"
    [[ -n "$EXCLUDE_NODES" ]] && SLURM_PARAMS+=" --exclude $EXCLUDE_NODES" 
    [[ -n "$JOB_DEP" ]] && SLURM_PARAMS+=" --dependency=afterany:$JOB_DEP"
    [[ -n "$OTHER_SLURM_PARAMS" ]] && SLURM_PARAMS+=" $OTHER_SLURM_PARAMS"

    if [[ "$INTERACTIVE" == "yes" ]]; then
        inform "Salloc with parameters: $SLURM_PARAMS"
        export SLURM_PARAMS="$SLURM_PARAMS"
        salloc $SLURM_PARAMS
    else
        [[ "$DEBUG_MODE" == "no" && "$DRY_RUN" == "no" ]] && SLURM_PARAMS+=" --exclusive --output=$OUTPUT_DIR/slurm_%j.out --error=$OUTPUT_DIR/slurm_%j.err" || SLURM_PARAMS+=" --output=debug_%j.out"
        export SLURM_PARAMS="$SLURM_PARAMS"
        inform "Sbatching job with parameters: $SLURM_PARAMS"
        sbatch $SLURM_PARAMS "$SWING_DIR/scripts/run_test_suite.sh"
    fi
fi
