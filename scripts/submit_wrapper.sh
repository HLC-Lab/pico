#!/bin/bash

source scripts/utils.sh

# 1. Set default values for the variables (are defined in `utils.sh`)
export TIMESTAMP=$DEFAULT_TIMESTAMP
export TYPES=$DEFAULT_TYPES
export SIZES=$DEFAULT_SIZES
export SEGMENT_SIZES=$DEFAULT_SEGMENT_SIZES
export COLLECTIVES=$DEFAULT_COLLECTIVES
export TEST_TIME=$DEFAULT_TEST_TIME
export CUDA=$DEFAULT_CUDA
export GPU_PER_NODE=$DEFAULT_GPU_PER_NODE
export OUTPUT_LEVEL=$DEFAULT_OUTPUT_LEVEL
export COMPRESS=$DEFAULT_COMPRESS
export DELETE=$DEFAULT_DELETE
export COMPILE_ONLY=$DEFAULT_COMPILE_ONLY
export DEBUG_MODE=$DEFAULT_DEBUG_MODE
export DRY_RUN=$DEFAULT_DRY_RUN
export INTERACTIVE=$DEFAULT_INTERACTIVE
export SHOW_MPICH_ENV=$DEFAULT_SHOW_MPICH_ENV
export NOTES=$DEFAULT_NOTES
export TASK_PER_NODE=$DEFAULT_TASK_PER_NODE

# 2. Parse and validate command line arguments
parse_cli_args "$@"

# 3. Set the location-specific configuration (defined in `config/environment/$LOCATION.sh`)
source_environment "$LOCATION" || exit 1

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
[[ "$COMPILE_ONLY" == "yes" ]] && exit 0

# 8. Defines env dependant variables
export ALGORITHM_CONFIG_FILE="$SWING_DIR/config/algorithm_config.json"
export LOCATION_DIR="$SWING_DIR/results/$LOCATION"
export OUTPUT_DIR="$SWING_DIR/results/$LOCATION/$TIMESTAMP"
export BENCH_EXEC_CPU=$SWING_DIR/bin/bench
[[ "$CUDA" == "True" ]] && export BENCH_EXEC_CUDA=$SWING_DIR/bin/bench_cuda
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
    PARAMS="--account $ACCOUNT --nodes $N_NODES --time $TEST_TIME --partition $PARTITION"

    if [[ -n "$QOS" ]]; then
        PARAMS+=" --qos $QOS"
        [[ -n "$QOS_TASKS_PER_NODE" ]] && export TASK_PER_NODE="$QOS_TASKS_PER_NODE"
        [[ -n "$QOS_GRES" ]] && GRES="$QOS_GRES"
    fi

    if [[ "$CUDA" == "True" ]]; then
        [[ -z "$GRES" ]] && GRES="gpu:$MAX_GPU_TEST"
        PARAMS+=" --gpus-per-node $MAX_GPU_TEST"
    fi

    [[ -n "$FORCE_TASKS" ]] && PARAMS+=" --ntasks $FORCE_TASKS" || PARAMS+=" --ntasks-per-node $TASK_PER_NODE"
    [[ -n "$GRES" ]] && PARAMS+=" --gres=$GRES"
    [[ -n "$EXCLUDE_NODES" ]] && PARAMS+=" --exclude $EXCLUDE_NODES" 

    PARAMS+=" --reservation=s_int_lped_boost"
    if [[ "$INTERACTIVE" == "yes" ]]; then
        salloc $PARAMS
    else
        if [[ "$DEBUG_MODE" == "no" && "$DRY_RUN" == "no" ]]; then
            sbatch $PARAMS --exclusive --output="$OUTPUT_DIR/slurm_%j.out" --error="$OUTPUT_DIR/slurm_%j.err" "$SWING_DIR/scripts/run_test_suite.sh"
        else
            sbatch $PARAMS --output="debug_%j.out" "$SWING_DIR/scripts/run_test_suite.sh"
        fi
    fi
fi
