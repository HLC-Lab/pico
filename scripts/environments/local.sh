#!/bin/bash
# Colors for styling output, otherwise utils needs to be sourced at every make
export RED='\033[0;31m'
export GREEN='\033[0;32m'
export YELLOW='\033[0;33m'
export BLUE='\033[1;34m'
export NC='\033[0m'

# Variables always needed
export CC=mpicc
export CFLAGS_COMP_SPECIFIC="-O3 -MMD -MP"
export RUN=mpiexec
export RUNFLAGS="--map-by :OVERSUBSCRIBE"
export SWING_DIR=$HOME/University/Tesi/test

# MPI library specific variables
export MPI_LIB='OMPI'    # Possible values: OMPI, OMPI_SWING
if [[ "$MPI_LIB" == "OMPI_SWING" ]]; then
    export PATH=/opt/ompi_test/bin:$PATH
    export LD_LIBRARY_PATH=/opt/ompi_test/lib:$LD_LIBRARY_PATH
    export MANPATH=/opt/ompi_test/share/man:$MANPATH
    export MPI_LIB_VERSION='5.0.0'
    export OMPI_MCA_coll_hcoll_enable=0
    export OMPI_MCA_coll_tuned_use_dynamic_rules=1
elif [[ "$MPI_LIB" == "OMPI" ]]; then
    export MPI_LIB_VERSION='5.0.6'
    export OMPI_MCA_coll_hcoll_enable=0
    export OMPI_MCA_coll_tuned_use_dynamic_rules=1
elif [[ "$MPI_LIB" == "MPICH" ]]; then
    export PATH=/opt/mpich-test/bin:$PATH
    export LD_LIBRARY_PATH=/opt/mpich-test/lib:$LD_LIBRARY_PATH
    export MANPATH=/opt/mpich-test/share:$MANPATH
    export MPI_LIB_VERSION='4.3.0'
fi


# Load test dependnt environment variables
load_other_env_var(){
    if [[ "$MPI_LIB" == "OMPI_SWING" ]] || [[ "$MPI_LIB" == "OMPI" ]]; then
        if [[ "$CUDA" == "False" ]]; then
            export CUDA_VISIBLE_DEVICES=""
            export OMPI_MCA_btl="^smcuda"
            export OMPI_MCA_mpi_cuda_support=0
        fi
    fi
}

export -f load_other_env_var
