# Variables always needed
export CC=mpicc
export CFLAGS_COMP_SPECIFIC="-O3 -MMD -MP"
export RUN=srun
export RUNFLAGS=--mpi=pmix
export SWING_DIR=$HOME/Swing_Test/

# TODO: insert correct values
export PARTITION=
export ACCOUNT=
export MODULES="python"

export MPI_LIB="OMPI_SWING"
if [ "$MPI_LIB" == "OMPI_SWING" ]; then
    export PATH=$HOME/bin:$PATH
    export LD_LIBRARY_PATH=$HOME/lib:$LD_LIBRARY_PATH
    export MANPATH=$HOME/share/man:$MANPATH
    export MPI_LIB_VERSION='5.0.0'
fi
export OMPI_MCA_coll_hcoll_enable=0
export OMPI_MCA_coll_tuned_use_dynamic_rules=1

# Load environment variables dependant on the MPI library
load_other_env_var() {
    if [ "$CUDA" == "False" ]; then
        export CUDA_VISIBLE_DEVICES=""
        export OMPI_MCA_btl="^smcuda"
        export OMPI_MCA_mpi_cuda_support=0
    fi
}
export -f load_other_env_var

