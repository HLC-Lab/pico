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
export SIZES="8"
export SEGMENT_SIZES="16384,131072,1048576"
export LIB_COUNT=1
export LIB_0_NAME="Open MPI"
export LIB_0_VERSION="5.0.7"
export LIB_0_STANDARD="MPI"
export LIB_0_MPI_LIB="OMPI"
export LIB_0_PICOCC="mpicc"
export LIB_0_MPI_LIB_VERSION="5.0.7"
export LIB_0_TASKS_PER_NODE="8"
export LIB_0_LOAD_TYPE="default"
export LIB_0_COLLECTIVES="alltoallv"
export LIB_0_ALLTOALLV_ALGORITHMS="bine_dh_over"
export LIB_0_ALLTOALLV_ALGORITHMS_SKIP="bine_dh_over"
export LIB_0_ALLTOALLV_ALGORITHMS_IS_SEGMENTED="no"
