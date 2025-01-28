#!/bin/bash

if [ "$ENABLE_OMPI_TEST" == "yes" ]; then
    export PATH=$HOME/bin:$PATH
    export LD_LIBRARY_PATH=$HOME/lib:$LD_LIBRARY_PATH
    export MANPATH=$HOME/share/man:$MANPATH
fi

if [ "$ENABLE_CUDA" == "no" ]; then
    export CUDA_VISIBLE_DEVICES=""
    export OMPI_MCA_btl="^smcuda"
    export OMPI_MCA_mpi_cuda_support=0
fi

export RUN=srun
export RUNFLAGS=

export UCX_IB_SL=1

export SWING_DIR=$HOME/Swing_Test/

module load python
