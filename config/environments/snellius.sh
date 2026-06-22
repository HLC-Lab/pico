# Variables always needed
export PICOCC=mpicc
export RUN=srun
export RUNFLAGS=--mpi=pmix

export PARTITION=rome
export PICO_ACCOUNT=vusei7310
export MODULES="python"

# TODO: insert correct values
if [[ "$PARTITION" == "rome" ]]; then
    export PARTITION_GPUS_PER_NODE=0
    export PARTITION_CPUS_PER_NODE=32
fi

export MPI_LIB="OMPI_BINE"
if [ "$MPI_LIB" == "OMPI_BINE" ]; then
    export PATH=$HOME/bin:$PATH
    export LD_LIBRARY_PATH=$HOME/lib:$LD_LIBRARY_PATH
    export MANPATH=$HOME/share/man:$MANPATH
    export MPI_LIB_VERSION='5.0.0'
fi
export OMPI_MCA_coll_hcoll_enable=0
export OMPI_MCA_coll_tuned_use_dynamic_rules=1

# Load environment variables dependant on the MPI library
load_other_env_var() {
    if [ "$GPU_AWARENESS" == "no" ]; then
        export CUDA_VISIBLE_DEVICES=""
        export OMPI_MCA_btl="^smcuda"
        export OMPI_MCA_mpi_cuda_support=0
    fi
}
export -f load_other_env_var
