#!/bin/bash

# Variables always needed
export RUN=mpiexec
export RUNFLAGS=

export SWING_DIR=$HOME/University/Tesi/test/


# Used to load python and virtual environment
load_python() {
    if [ -f "~/.swing_venv" ]; then
        success "Virtual environment 'swing_venv' exists. Activating it..."
        source ~/.swing_venv/bin/activate
    else
        error "Virtual environment 'swing_venv' does not exist."
        error "Please create a virtual environment and install the required packages."
        error "Required packages: jsonschema, pandas, matplotlib, seaborn"
    fi
}

# Load environment variables dependant on the MPI library
load_other_env_var() {
    if [ "$MPI_LIB" == "OMPI_SWING" ]; then
        source ~/use_ompi.sh
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


