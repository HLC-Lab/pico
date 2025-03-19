# General variables always needed
export SWINGCC=cc
export RUN=srun
export SWING_DIR=$HOME/Swing_Test

# Account/partition specific variables
export PARTITION=standard-g
export QOS=''
export ACCOUNT=project_465000997

# MPI library specific variables
export MPI_LIB='CRAY_MPICH'
export MPI_LIB_VERSION='8.1.29'

export MODULES="LUMI/24.03 partition/L,cray-python/3.11.5,EasyBuild-user,git-lfs/3.4.0"

# Dummy function to load other environment variables to silence errors
load_other_env_var() {
  if [[ "$DEBUG_MODE" == "yes" ]]; then
    export MPICH_ENV_DISPLAY=1
  else
    export MPICH_ENV_DISPLAY=0
  fi
}
export -f load_other_env_var
