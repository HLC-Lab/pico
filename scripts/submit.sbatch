#!/bin/bash
#SBATCH -p boost_usr_prod                  # Partition name
# #SBATCH -q <qos_name>              # Required Quality of Service (optional)
#SBATCH -N 4                 # Number of nodes
#SBATCH --ntasks-per-node=1          # Number of taasks per node
#SBATCH --time=02:00:00      # Time limit
#SBATCH --exclusive                  # Exclusive access to nodes
#SBATCH --account=IscrC_ASCEND     # Account name

DEBUG_MODE=no
TEST_CONFIG_FILE=$HOME/Swing_Test/scripts/config/test.json
TIMESTAMP=$(date +"%Y_%m_%d___%H:%M:%S")
LOCATION=leonardo # possible locations are leonardo, snellius and local
NOTES="Test run for allreduce"

# Create the results directory if it doesn't exist
# (for redundacy will be done also in run_test_suite)
mkdir -p $HOME/Swing_Test/results/$LOCATION/
mkdir -p $HOME/Swing_Test/results/$LOCATION/$TIMESTAMP
mkdir -p $HOME/Swing_Test/results/$LOCATION/$TIMESTAMP/data

# TODO: adjust this
# Redirect standard output and error to the specified directory
#SBATCH --output=$HOME/Swing_Test/results/$LOCATION/$TIMESTAMP/slurm-%j.out
#SBATCH --error=$HOME/Swing_Test/results/$LOCATION/$TIMESTAMP/slurm-%j.err

$HOME/Swing_Test/scripts/run_test_suite.sh $SLURM_NNODES $DEBUG_MODE $TEST_CONFIG_FILE $TIMESTAMP $LOCATION $NOTES

