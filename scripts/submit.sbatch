#!/bin/bash
#SBATCH -p boost_usr_prod                   # Partition name
# #SBATCH -q <qos_name>                     # Required Quality of Service (optional)
#SBATCH --exclusive                         # Exclusive access to nodes
#SBATCH --account=IscrC_ASCEND              # Account name

$HOME/Swing_Test/scripts/run_test_suite.sh $SLURM_NNODES

if [ $DEBUG_MODE == "no" ]; then
    $HOME/Swing_Test/results/compress_results.sh
fi

