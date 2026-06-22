# Variables always needed
export PICOCC=mpicc
export RUN=srun

# Account/partition specific variables
export PARTITION=acc
export PICO_ACCOUNT=ehpc181

export UCX_MAX_RNDV_RAILS=1
export NOTES="$NOTES UCX_MAX_RNDV_RAILS=$UCX_MAX_RNDV_RAILS"
export MODULES="mkl/2024.2,python/3.12.1"

if [[ "$PARTITION" == "acc" ]]; then
    export PARTITION_GPUS_PER_NODE=4
    export PARTITION_CPUS_PER_NODE=80
    [[ "$DEBUG_MODE" == "yes" && "$N_NODES" -le 8 ]] && export QOS="acc_debug" || export QOS="acc_ehpc"
fi

[[ "$GPU_AWARENESS" == "yes" ]] && export MODULES="cuda/12.2,$MODULES"
export GPU_LIB='CUDA'
export GPU_LIB_VERSION='12.2'

# MPI library specific variables
export MPI_LIB='OMPI'
export MPI_LIB_VERSION='4.1.5'
export MODULES="intel/2024.2,openmpi/4.1.5,$MODULES"

# Load test dependnt environment variables
load_other_env_var(){
    export OMPI_MCA_coll_hcoll_enable=0
    export OMPI_MCA_coll_tuned_use_dynamic_rules=1
    if [ "$GPU_AWARENESS" == "no" ]; then
        export OMPI_MCA_btl="^smcuda"
        export OMPI_MCA_mpi_cuda_support=0
    else
        export OMPI_MCA_btl=""
        export OMPI_MCA_mpi_cuda_support=1
    fi
}
export -f load_other_env_var

