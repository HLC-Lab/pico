#!/bin/bash
export LOCATION="leonardo"
export RUN=srun
export UCX_IB_SL=1
export PARTITION="boost_usr_prod"
export GENERAL_MODULES="python/3.11.6--gcc--8.5.0"
export COMPILE_ONLY="yes"
export DEBUG_MODE="no"
export OUTPUT_LEVEL="minimal"
export DRY_RUN="no"
export DELETE="no"
export COMPRESS="no"
export N_NODES=1
# skipped: test.output_level missing
# skipped: test.test_time missing
# skipped: test.dimensions missing
export LIB_COUNT=2
export LIB_0_NAME="Open MPI 4.1.6"
export LIB_0_VERSION="4.1.6"
export LIB_0_STANDARD="MPI"
export LIB_0_MPI_LIB="OMPI"
export LIB_0_PICOCC="mpicc"
export LIB_0_MPI_LIB_VERSION="4.1.6"
export LIB_0_TASKS_PER_NODE="1"
export LIB_0_LOAD_TYPE="module"
export LIB_0_MODULES="openmpi/4.1.6--gcc--12.2.0"
export LIB_0_COLLECTIVES="allreduce"
export LIB_0_ALLREDUCE_ALGORITHMS="default_ompi"
export LIB_0_ALLREDUCE_ALGORITHMS_SKIP=""
export LIB_0_ALLREDUCE_ALGORITHMS_IS_SEGMENTED="no"
export LIB_1_NAME="Open MPI NVHPC 23.11"
export LIB_1_VERSION="4.1.6"
export LIB_1_STANDARD="MPI"
export LIB_1_MPI_LIB="OMPI"
export LIB_1_PICOCC="mpicc"
export LIB_1_MPI_LIB_VERSION="4.1.6"
export LIB_1_GPU_PER_NODE="1"
export LIB_1_GPU_AWARENESS="yes"
export LIB_1_LOAD_TYPE="module"
# skipped: libraries[1].gpu_support.gpu_load module unavailable for GPU-aware lib
export LIB_1_MODULES="openmpi/4.1.6--nvhpc--23.11"
export LIB_1_COLLECTIVES="allreduce"
export LIB_1_ALLREDUCE_ALGORITHMS="default_ompi"
export LIB_1_ALLREDUCE_ALGORITHMS_SKIP=""
export LIB_1_ALLREDUCE_ALGORITHMS_IS_SEGMENTED="no"
