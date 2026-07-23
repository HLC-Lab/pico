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
# skipped: test.inject_params missing
export TYPES="int32"
export SIZES="8,64,512,4096,32768,262144,2097152,16777216,134217728"
export SEGMENT_SIZES="16384,131072,1048576"
export LIB_COUNT=1
export LIB_0_NAME="MPICH Custom Bine"
export LIB_0_VERSION="5.0.0b1"
export LIB_0_STANDARD="MPI"
export LIB_0_MPI_LIB="MPICH"
export LIB_0_PICOCC="mpicc"
export LIB_0_MPI_LIB_VERSION="5.0.0b1"
export LIB_0_TASKS_PER_NODE="2"
export LIB_0_LOAD_TYPE="set_env"
export LIB_0_ENV_PREPEND_PATH="/home/angelo/mpich_install/bin"
export LIB_0_ENV_PREPEND_LD_LIBRARY_PATH="/home/angelo/mpich_install/lib"
export LIB_0_ENV_PREPEND_VARS="PATH,LD_LIBRARY_PATH"
export LIB_0_COLLECTIVES="allreduce"
export LIB_0_ALLREDUCE_ALGORITHMS="bine_bdw_mpich,bine_bdw_remap_segmented_over"
export LIB_0_ALLREDUCE_ALGORITHMS_SKIP="bine_bdw_mpich,bine_bdw_remap_segmented_over"
export LIB_0_ALLREDUCE_ALGORITHMS_IS_SEGMENTED="yes,yes"
export LIB_0_ALLREDUCE_ALGORITHMS_CVARS="bine_bdw,auto"
