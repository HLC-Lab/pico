# Variables always needed
export SWINGCC=mpicc
export RUN=srun
export SWING_DIR=$HOME/Swing_Test

# Account/partition specific variables
export PARTITION=acc
export ACCOUNT=ehpc181
[[ "$DEBUG_MODE" == "yes" && "$N_NODES" -le 8 ]] && export QOS="acc_debug" || export QOS="acc_ehpc"

export UCX_MAX_RNDV_RAILS=1
export NOTES="$NOTES UCX_MAX_RNDV_RAILS=$UCX_MAX_RNDV_RAILS"
export MODULES="mkl/2024.2,python/3.12.1"

# MPI library specific variables
export MPI_LIB='OMPI'
export MPI_LIB_VERSION='4.1.5'
export MODULES="intel/2024.2,openmpi/4.1.5,$MODULES"

# Load test dependnt environment variables
load_other_env_var(){
    export OMPI_MCA_coll_hcoll_enable=0
    export OMPI_MCA_coll_tuned_use_dynamic_rules=1
    if [ "$CUDA" == "False" ]; then
        export OMPI_MCA_btl="^smcuda"
        export OMPI_MCA_mpi_cuda_support=0
    else
        export OMPI_MCA_btl=""
        export OMPI_MCA_mpi_cuda_support=1
    fi
}
export -f load_other_env_var

