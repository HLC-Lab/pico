#!/bin/bash
#SBATCH -p <p_name>                  # Partition name
# #SBATCH -q <qos_name>              # Required Quality of Service (optional)
#SBATCH -N <n_nodes>                 # Number of nodes
#SBATCH --ntasks-per-node=1          # Number of taasks per node
#SBATCH --time=<requested_time>      # Time limit
#SBATCH --exclusive                  # Exclusive access to nodes
#SBATCH --account=<account_name>     # Account name

TIMESTAMP=$(date +"%Y_%m_%d___%H:%M:%S")
LOCATION=leonardo # possible locations are leonardo, snellius and local
ENABLE_CUDA=no
ENABLE_OMPI_TEST=yes
COLLECTIVE_TO_TEST=ALLREDUCE

# Create the results directory if it doesn't exist
# (for redundacy will be done also in run_test_suite)
mkdir -p $HOME/Swing_Test/results/$LOCATION/
mkdir -p $HOME/Swing_Test/results/$LOCATION/$TIMESTAMP
mkdir -p $HOME/Swing_Test/results/$LOCATION/$TIMESTAMP/data

# Redirect standard output and error to the specified directory
#SBATCH --output=$HOME/Swing_Test/results/$LOCATION/$TIMESTAMP/slurm-%j.out
#SBATCH --error=$HOME/Swing_Test/results/$LOCATION/$TIMESTAMP/slurm-%j.err

$HOME/Swing_Test/scripts/run_test_suite.sh $SLURM_NNODES $TIMESTAMP $LOCATION $COLLECTIVE_TO_TEST $ENABLE_CUDA $ENABLE_OMPI_TEST
