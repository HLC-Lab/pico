#ifndef SUPPORT_KERNEL_H
#define SUPPORT_KERNEL_H

#include <mpi.h>
#include <cuda.h> 
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

enum ReduceType {
    R_INT8,
    R_INT16,
    R_INT32,
    R_INT64,
    R_INT,
    R_FLOAT,
    R_DOUBLE,
    R_CHAR,
    R_UNNOWN_TYPE,
    R_TYPE_NUM
};

enum ReduceOp {
    R_SUM,
    R_PROD,
    R_MAX,
    R_MIN,
    R_LAND,
    R_BAND,
    R_LOR,
    R_BOR,
    R_LXOR,
    R_BXOR,
    R_UNNOWN_OP,
    R_OP_NUM
};

// simple reduce wrappe reduce 2 buffer
int reduce_wrapper(void* inbuff, void* inoutbuff, int count, MPI_Datatype dtype, MPI_Op op);

// reduce wrapper that reducea group of buffer of a certent size
int reduce_wrapper_grops(void *inbuff, void *inoutbuff, int group_size, int groups, MPI_Datatype dtype, MPI_Op op);

// reduce wrapper that reducea group of buffer of a certent size where ther are a set of starter data on another buffer
int reduce_wrapper_grops_inoutsplit(void *inbuff, void *outbuff, const void *currentbuff, int group_size, int groups, MPI_Datatype dtype, MPI_Op op);

int reorder_kernel_wrapper(void *inbuff, void *outbuff, int elem, int size, int gpu_on_node, MPI_Datatype dtype);

#ifdef __cplusplus
}
#endif
#endif