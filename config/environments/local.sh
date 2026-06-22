# Variables always needed
export PICOCC=mpicc
export RUN=mpiexec
export RUNFLAGS="--map-by :OVERSUBSCRIBE"
export PARTITION_GPUS_PER_NODE=1
export PARTITION_CPUS_PER_NODE=16

export GPU_LIB='CUDA'
export GPU_LIB_VERSION='11.8.0'

# MPI library specific variables
export MPI_LIB='OMPI'    # Possible values: OMPI, OMPI_BINE
if [[ "$MPI_LIB" == "OMPI_BINE" ]]; then
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
    if [[ "$MPI_LIB" == "OMPI_BINE" || "$MPI_LIB" == "OMPI" ]]; then
        if [[ "$GPU_AWARENESS" == "no" ]]; then
            export OMPI_MCA_btl="^smcuda"
            export OMPI_MCA_mpi_cuda_support=0
        else
            export OMPI_MCA_btl=""
            export OMPI_MCA_mpi_cuda_support=1
        fi
    elif [[ "$MPI_LIB" == "MPICH" ]]; then
        [[ "$DEBUG_MODE" == "yes" && "$SHOW_ENV" == "yes" ]] && export MPICH_ENV_DISPLAY=1 || export MPICH_ENV_DISPLAY=0
    fi
}
export -f load_other_env_var
