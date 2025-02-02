#!/bin/bash

if [ "$MPI_LIB" == "OMPI_SWING" ]; then
    export PATH=$HOME/bin:$PATH
    export LD_LIBRARY_PATH=$HOME/lib:$LD_LIBRARY_PATH
    export MANPATH=$HOME/share/man:$MANPATH
fi

if [[ "$MPI_LIB" == "OMPI_SWING" ]] || [[ "$MPI_LIB" == "OMPI" ]]; then
    export OMPI_MCA_coll_hcoll_enable=0
    export OMPI_MCA_coll_tuned_use_dynamic_rules=1
    if [ "$CUDA" == "False" ]; then
        export CUDA_VISIBLE_DEVICES=""
        export OMPI_MCA_btl="^smcuda"
        export OMPI_MCA_mpi_cuda_support=0
    fi
fi

export RUN=srun
export RUNFLAGS=--mpi=pmix

export SWING_DIR=$HOME/Swing_Test/
