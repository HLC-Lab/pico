#!/bin/bash

source scripts/utils.sh

##################################################################################
#               MODIFY THESE VARIABLES TO SUIT YOUR TEST ENVIRONMENT             #
##################################################################################
# Global variables
export N_NODES=4
export LOCATION="leonardo"
export TIMESTAMP=$(date +"%Y_%m_%d___%H_%M_%S")
export DEBUG_MODE="no"
export NOTES="Testing on 4 nodes new refactoring"
# SLURM variables
export TASK_PER_NODE=1              # Beware that the script will still run only one task per node
export TEST_TIME=01:00:00
export PARTITION=boost_usr_prod
export QOS=''
export ACCOUNT=IscrC_ASCEND

if ! source_environment "$LOCATION"; then
    error "Environment script for '${LOCATION}' not found!"
    exit 1
else 
    success "Environment script for '${LOCATION}' loaded successfully."
fi

export TEST_CONFIG_FILE="$SWING_DIR/config/test.json"

###################################################################################
#               PARSE THE TEST CONFIGURATION FILE TO GET THE TEST VARIABLES       #
###################################################################################
export ALGORITHM_CONFIG_FILE="$SWING_DIR/config/algorithm_config.json"
export TEST_ENV="${TEST_CONFIG_FILE}_env.sh"

if [ $LOCATION != "local" ]; then
    load_python || exit 1
fi

activate_virtualenv || exit 1

python3 $SWING_DIR/config/parse_test.py || exit 1
# Source the test specific environment variables
source $TEST_ENV

# Select here what to do in debug mode
if [ "$DEBUG_MODE" == yes ]; then
    export COLLECTIVE_TYPE="ALLREDUCE"
    export ALGOS="default_ompi"
    export ARR_SIZES="8"
    export TYPES="int" # For now only int,int32,int64 are supported in debug mode 
fi

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
    if [ $QOS != '' ]; then
    sbatch --account=$ACCOUNT --partition=$PARTITION --qos=$QOS --nodes=$N_NODES --ntasks-per-node=$TASK_PER_NODE --exclusive --time=$TEST_TIME --output="${OUTPUT_DIR}/slurm_%j.out" --error="${OUTPUT_DIR}/slurm_%j.err" $SWING_DIR/scripts/run_test_suite.sh $SLURM_NNODES
    else
    sbatch --account=$ACCOUNT --partition=$PARTITION --nodes=$N_NODES --ntasks-per-node=$TASK_PER_NODE --exclusive --time=$TEST_TIME --output="${OUTPUT_DIR}/slurm_%j.out" --error="${OUTPUT_DIR}/slurm_%j.err" $SWING_DIR/scripts/run_test_suite.sh $SLURM_NNODES
    fi
fi
