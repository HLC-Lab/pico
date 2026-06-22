# General variables always needed
export PICOCC=cc
export RUN=srun

# Account/partition specific variables
export PARTITION=standard-g
export PICO_ACCOUNT=project_465001736

if [[ "$PARTITION" == "standard-g" ]]; then
    export PARTITION_GPUS_PER_NODE=4
    export PARTITION_CPUS_PER_NODE=56 # 64 cores per node, 8 are reserved for the system
fi

# MPI library specific variables
export MPI_LIB='CRAY_MPICH'
export MPI_LIB_VERSION='8.1.29'

export MODULES="LUMI/24.03 partition/L,cray-python/3.11.5"

# Dummy function to load other environment variables to silence errors
load_other_env_var() {
    [[ "$DEBUG_MODE" == "yes" && "$SHOW_ENV" == "yes" ]] && export MPICH_ENV_DISPLAY=1 || export MPICH_ENV_DISPLAY=0
}
export -f load_other_env_var
