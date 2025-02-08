#!/bin/bash

source scripts/utils.sh

##################################################################################
#               MODIFY THESE VARIABLES TO SUIT YOUR TEST ENVIRONMENT             #
##################################################################################
export N_NODES=2
export TASK_PER_NODE=1
export TEST_TIME=01:00:00
export LOCATION="local"
export DEBUG_MODE="no"
export TIMESTAMP=$(date +"%Y_%m_%d___%H_%M_%S")
export NOTES="This is a test run"

if ! source_environment "$LOCATION"; then
    error "Environment script for '${LOCATION}' not found!"
    exit 1
else 
    success "Environment script for '${LOCATION}' loaded successfully."
fi

export ALGORITHM_CONFIG_FILE="$SWING_DIR/scripts/config/algorithm_config.json"
export TEST_CONFIG_FILE="$SWING_DIR/scripts/config/test.json"
export TEST_ENV="${TEST_CONFIG_FILE}_env.sh"

# Select here what to do in debug mode
if [ "$DEBUG_MODE" == yes ]; then
    export COLLECTIVE_TYPE="ALLREDUCE"
    export ALGOS="default_ompi"
    export ARR_SIZES="8"
    export TYPES="int" # For now only int,int32,int64 are supported in debug mode 
fi
###################################################################################
#               PARSE THE TEST CONFIGURATION FILE TO GET THE TEST VARIABLES       #
###################################################################################
if [ $LOCATION != "local" ]; then
    load_python || exit 1
fi

activate_virtualenv || exit 1

python3 $SWING_DIR/scripts/config/parse_test.py || exit 1
# Source the test specific environment variables
source $TEST_ENV

# Load library-location specific environment variables
load_other_env_var

###################################################################################
#           COMPILE CODE, CREATE OUTPUT DIRECTORIES AND GENERATE METADATA         #
###################################################################################
compile_code || exit 1

export LOCATION_DIR="$SWING_DIR/results/$LOCATION"
export OUTPUT_DIR="$SWING_DIR/results/$LOCATION/$TIMESTAMP"
export DATA_DIR="$SWING_DIR/results/$LOCATION/$TIMESTAMP/data"
if [ $DEBUG_MODE == "no" ]; then
    success "📂 Creating output directories..."
    mkdir -p "$LOCATION_DIR"
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$DATA_DIR"
    
    # Generate test metadata
    python3 $SWING_DIR/results/generate_metadata.py  || exit 1
fi
###################################################################################


###################################################################################
#               DO NOT MODIFY THE FOLLOWING VARIABLES                             #
###################################################################################
export TEST_EXEC=$SWING_DIR/bin/test
export RULE_UPDATER_EXEC=$SWING_DIR/ompi_rules/change_dynamic_rules.py
export DYNAMIC_RULE_FILE=$SWING_DIR/ompi_rules/dynamic_rules.txt

# Submit the job.
if [ $LOCATION == "local" ]; then
    scripts/run_test_suite.sh $N_NODES
else
    sbatch scripts/submit.sh
fi
