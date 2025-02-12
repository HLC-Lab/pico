#!/bin/bash

# Variables always needed
export CC=mpicc
export RUN=srun
export SWING_DIR=$HOME/Swing_Test

# Account/partition specific variables
export PARTITION=boost_usr_prod
export QOS=''
export ACCOUNT=IscrC_ASCEND

export UCX_IB_SL=1

# MPI library specific variables
export MPI_LIB='OMPI_SWING'    # Possible values: OMPI, OMPI_SWING (beware that OMPI_SWING must be manually installed in the home directory)
export MPI_LIB_VERSION='5.0.0'
if [ "$MPI_LIB" == "OMPI_SWING" ]; then
    export PATH=$HOME/bin:$PATH
    export LD_LIBRARY_PATH=$HOME/lib:$LD_LIBRARY_PATH
    export MANPATH=$HOME/share/man:$MANPATH
fi
export OMPI_MCA_coll_hcoll_enable=0
export OMPI_MCA_coll_tuned_use_dynamic_rules=1

# Load test dependnt environment variables
load_other_env_var(){
    if [ "$CUDA" == "False" ]; then
        export CUDA_VISIBLE_DEVICES=""
        export OMPI_MCA_btl="^smcuda"
        export OMPI_MCA_mpi_cuda_support=0
    else
        export CUDA_VISIBLE_DEVICES=0,1,2,3
        export OMPI_MCA_btl=""
        export OMPI_MCA_mpi_cuda_support=1
    fi
}

# Used to load python and virtual environment
load_python() {
    module load python/3.11.6--gcc--8.5.0 || { error "Failed to load Python module." ; return 1; }
    return 0
}

