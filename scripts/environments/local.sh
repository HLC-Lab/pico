#!/bin/bash

# Variables always needed
export RUN=mpiexec
export SWING_DIR=$HOME/University/Tesi/test

# MPI library specific variables
export MPI_LIB='OMPI_SWING'    # Possible values: OMPI, OMPI_SWING
export MPI_LIB_VERSION='5.0.0'
if [ "$MPI_LIB" == "OMPI_SWING" ]; then
    source ~/use_ompi.sh
fi
export OMPI_MCA_coll_hcoll_enable=0
export OMPI_MCA_coll_tuned_use_dynamic_rules=1

# Load test dependnt environment variables
load_other_env_var(){
    if [ "$CUDA" == "False" ]; then
        export CUDA_VISIBLE_DEVICES=""
        export OMPI_MCA_btl="^smcuda"
        export OMPI_MCA_mpi_cuda_support=0
    fi
}
