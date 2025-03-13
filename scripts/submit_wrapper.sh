#!/bin/bash

source scripts/utils.sh

# 1. Set default values for the variables (are defined in `utils.sh`)
export TIMESTAMP=$DEFAULT_TIMESTAMP
export TYPES=$DEFAULT_TYPES
export SIZES=$DEFAULT_SIZES
export CUDA=$DEFAULT_CUDA
export DEBUG_MODE=$DEFAULT_DEBUG_MODE
export OUTPUT_LEVEL=$DEFAULT_OUTPUT_LEVEL
export INTERACTIVE=$DEFAULT_INTERACTIVE
export COMPRESS=$DEFAULT_COMPRESS
export DELETE=$DEFAULT_DELETE
export NOTES=$DEFAULT_NOTES
export TASK_PER_NODE=$DEFAULT_TASK_PER_NODE
export TEST_TIME=$DEFAULT_TEST_TIME

# 2. Parse and validate command line arguments
parse_cli_args "$@"

# 3. Set the location-specific environment variables
source_environment "$LOCATION" || exit 1
validate_args || exit 1

# 4. Set the test configuration files, or use --test-config if provided.
if [ -n "$TEST_CONFIG_OVERRIDE" ]; then
    TEST_CONFIG_FILE_LIST=()
    for f_path in ${TEST_CONFIG_OVERRIDE//,/ }; do
        if [ ! -f "$SWING_DIR/$f_path" ]; then
            error "Test configuration file '$SWING_DIR/$f_path' not found!"
            exit 1
        fi
        TEST_CONFIG_FILE_LIST+=( "$SWING_DIR/$f_path" )
    done
else
    TEST_CONFIG_FILE_LIST=(
        "$SWING_DIR/config/test/allreduce.json"
        "$SWING_DIR/config/test/allgather.json"
        "$SWING_DIR/config/test/bcast.json"
        "$SWING_DIR/config/test/reduce_scatter.json"
    )
fi
export TEST_CONFIG_FILES="${TEST_CONFIG_FILE_LIST[*]}"

###################################################################################
#               PARSE THE TEST CONFIGURATION FILE TO GET THE TEST VARIABLES       #
###################################################################################
export ALGORITHM_CONFIG_FILE="$SWING_DIR/config/algorithm_config.json"

load_modules || exit 1
success "Modules successfully loaded."

activate_virtualenv || exit 1
success "Virtual environment activated."

###################################################################################
#           COMPILE CODE, CREATE OUTPUT DIRECTORIES AND GENERATE METADATA         #
###################################################################################
compile_code || exit 1

export LOCATION_DIR="$SWING_DIR/results/$LOCATION"
export OUTPUT_DIR="$SWING_DIR/results/$LOCATION/$TIMESTAMP"
if [ $DEBUG_MODE == "no" ]; then
    success "📂 Creating output directories..."
    mkdir -p "$LOCATION_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

###################################################################################
#               DO NOT MODIFY THE FOLLOWING VARIABLES                             #
###################################################################################
export BENCH_EXEC=$SWING_DIR/bin/bench
export ALGO_CHANGE_SCRIPT=$SWING_DIR/selector/change_dynamic_rules.py
export DYNAMIC_RULE_FILE=$SWING_DIR/selector/ompi_dynamic_rules.txt

# Submit the job.
if [[ "$LOCATION" == "local" ]]; then
    scripts/run_test_suite.sh
else
    PARAMS="--account=$ACCOUNT --partition=$PARTITION --nodes=$N_NODES --ntasks-per-node=$TASK_PER_NODE --exclusive --time=$TEST_TIME"
    [[ -n "$QOS" ]] &&  PARAMS+=" --qos=$QOS"
    [[ "$CUDA" == "True" ]] && PARAMS+=" --gres=gpu:1 --gpus-per-task=1 --gpus-per-node=1"
    
    if [[ "$INTERACTIVE" == "yes" ]]; then
        salloc $PARAMS
    else
        if [[ "$DEBUG_MODE" == "yes" ]]; then
            sbatch $PARAMS --output="debug_%j.out" "$SWING_DIR/scripts/run_test_suite.sh"
        else
            sbatch $PARAMS --output="$OUTPUT_DIR/slurm_%j.out" --error="$OUTPUT_DIR/slurm_%j.err" "$SWING_DIR/scripts/run_test_suite.sh"
        fi
    fi
fi
