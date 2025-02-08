#!/bin/bash
#SBATCH -p boost_usr_prod                   # Partition name
# #SBATCH -q <qos_name>                     # Required Quality of Service (optional)
#SBATCH --nodes=$N_NODES                    # Number of nodes
#SBATCH --ntasks-per-node=$TASK_PER_NODE    # Number of taasks per node
#SBATCH --time=$TIME_LIMIT                  # Time limit
#SBATCH --exclusive                         # Exclusive access to nodes
#SBATCH --output=$RESULTS_DIR/slurm-%j.out  # Output file
#SBATCH --error=$RESULTS_DIR/slurm-%j.err   # Error file
#SBATCH --account=IscrC_ASCEND              # Account name

$HOME/Swing_Test/scripts/run_test_suite.sh $SLURM_NNODES

if [ $DEBUG_MODE == "no" ]; then
    $HOME/Swing_Test/results/compress_results.sh
fi

