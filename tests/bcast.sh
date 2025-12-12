#!/bin/bash
export LOCATION="local"
export RUN=mpirun
# skipped: environment.partition missing
export COMPILE_ONLY="no"
export DEBUG_MODE="no"
export DRY_RUN="no"
export DELETE="yes"
export COMPRESS="yes"
export N_NODES=1
export OUTPUT_LEVEL="minimal"
# skipped: test.test_time missing
export TYPES="int32"
export SIZES="8,64,512,4096,32768"
export SEGMENT_SIZES="0"
export LIB_COUNT=1
export LIB_0_NAME="Open MPI"
export LIB_0_VERSION="5.0.7"
export LIB_0_STANDARD="MPI"
export LIB_0_MPI_LIB="OMPI"
export LIB_0_PICOCC="mpicc"
export LIB_0_MPI_LIB_VERSION="5.0.7"
export LIB_0_TASKS_PER_NODE="8"
export LIB_0_LOAD_TYPE="default"
export LIB_0_COLLECTIVES="bcast"
export LIB_0_BCAST_ALGORITHMS="linear_over,binomial_over,scatter_allgather_over"
export LIB_0_BCAST_ALGORITHMS_SKIP="scatter_allgather_over"
export LIB_0_BCAST_ALGORITHMS_IS_SEGMENTED="no,no,no"
