#!/bin/bash

source scripts/utils.sh

##################################################################################
#               MODIFY THESE VARIABLES TO SUIT YOUR TEST ENVIRONMENT             #
##################################################################################
# Global variables
export N_NODES=8
export LOCATION="local"
export TIMESTAMP=$(date +"%Y_%m_%d___%H_%M_%S")
export DEBUG_MODE="yes"
export NOTES="debugging..."
# SLURM specific variables, other variables are set in the environment script
export TASK_PER_NODE=1              # Beware that the script will still run only one task per node
export TEST_TIME=01:00:00

if ! source_environment "$LOCATION"; then
    error "Environment script for '${LOCATION}' not found!"
    exit 1
fi 

TEST_CONFIG_FILE_LIST=(
    "$SWING_DIR/config/allreduce.json"
    "$SWING_DIR/config/allgather.json"
    "$SWING_DIR/config/bcast.json"
    "$SWING_DIR/config/reduce_scatter.json"
)

# Convert array to a colon-separated string
export TEST_CONFIG_FILES="${TEST_CONFIG_FILE_LIST[*]}"

###################################################################################
#               PARSE THE TEST CONFIGURATION FILE TO GET THE TEST VARIABLES       #
###################################################################################
export ALGORITHM_CONFIG_FILE="$SWING_DIR/config/algorithm_config.json"

if [ $LOCATION != "local" ]; then
    load_python || exit 1
fi

activate_virtualenv || exit 1


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
    if [ "$QOS" != "" ]; then
    sbatch --account=$ACCOUNT --partition=$PARTITION --qos=$QOS --nodes=$N_NODES --ntasks-per-node=$TASK_PER_NODE --exclusive --time=$TEST_TIME --output="${OUTPUT_DIR}/slurm_%j.out" --error="${OUTPUT_DIR}/slurm_%j.err" $SWING_DIR/scripts/run_test_suite.sh $N_NODES
    else
    sbatch --account=$ACCOUNT --partition=$PARTITION --nodes=$N_NODES --ntasks-per-node=$TASK_PER_NODE --exclusive --time=$TEST_TIME --output="${OUTPUT_DIR}/slurm_%j.out" --error="${OUTPUT_DIR}/slurm_%j.err" $SWING_DIR/scripts/run_test_suite.sh $N_NODES
    fi
fi
