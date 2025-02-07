#!/bin/bash

# Variables always needed
export RUN=srun
export RUNFLAGS=

export UCX_IB_SL=1

export SWING_DIR=$HOME/Swing_Test/

# Used to load python and virtual environment
load_python() {
    module load python/3.11.6--gcc--8.5.0
    if [ -f "$HOME/.swing_venv/bin/activate" ]; then
        success "Virtual environment 'swing_venv' exists. Activating it..."
        source $HOME/.swing_venv/bin/activate
    else
        error "Virtual environment 'swing_venv' does not exist."
        error "Please create a virtual environment and install the required packages from login node."
        error "Required packages: jsonschema, pandas, matplotlib, seaborn"
    fi
}

# Load environment variables dependant on the MPI library
load_other_env_var() {
    if [ "$MPI_LIB" == "OMPI_SWING" ]; then
        export PATH=$HOME/bin:$PATH
        export LD_LIBRARY_PATH=$HOME/lib:$LD_LIBRARY_PATH
        export MANPATH=$HOME/share/man:$MANPATH
    fi

    if [[ "$MPI_LIB" == "OMPI_SWING" ]] || [[ "$MPI_LIB" == "OMPI" ]]; then
        export OMPI_MCA_coll_hcoll_enable=0
        export OMPI_MCA_coll_tuned_use_dynamic_rules=1
        if [ "$CUDA" == "False" ]; then
            export CUDA_VISIBLE_DEVICES=""
            export OMPI_MCA_btl="^smcuda"
            export OMPI_MCA_mpi_cuda_support=0
        fi
    fi
}


