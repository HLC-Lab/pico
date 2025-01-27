#!/bin/bash

if [ "$ENABLE_OMPI_TEST" == "yes" ]; then
    source ~/use_ompi.sh
fi

# source ~/.venv1/bin/activate

export RUN=mpiexec
export RUNFLAGS=

export SWING_DIR=$HOME/University/Tesi/test/

