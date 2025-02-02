#!/bin/bash

if [ "$MPI_LIB" == "OMPI_SWING" ]; then
    source ~/use_ompi.sh
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
source ~/.venv1/bin/activate

export RUN=mpiexec
export RUNFLAGS=

export SWING_DIR=$HOME/University/Tesi/test/

