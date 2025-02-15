#ifndef LIBSWING_H
#define LIBSWING_H

#include <mpi.h>
#include <stddef.h>

#define ALLREDUCE_ARGS        const void *sbuf, void *rbuf, size_t count, \
                              MPI_Datatype dtype, MPI_Op op, MPI_Comm comm
#define ALLGATHER_ARGS        const void *sbuf, size_t scount, MPI_Datatype sdtype, \
                              void* rbuf, size_t rcount, MPI_Datatype rdtype, MPI_Comm comm
#define BCAST_ARGS            void *buf, size_t count, MPI_Datatype dtype, int root, MPI_Comm comm
#define REDUCE_SCATTER_ARGS   const void *sbuf, void *rbuf, const int rcounts[], \
                              MPI_Datatype dtype, MPI_Op op, MPI_Comm comm

int allreduce_recursivedoubling(ALLREDUCE_ARGS);
int allreduce_ring(ALLREDUCE_ARGS);
int allreduce_rabenseifner(ALLREDUCE_ARGS);
int allreduce_swing_lat(ALLREDUCE_ARGS);
int allreduce_swing_bdw_static(ALLREDUCE_ARGS);

int allgather_k_bruck(ALLGATHER_ARGS, int radix);
int allgather_recursivedoubling(ALLGATHER_ARGS);
int allgather_ring(ALLGATHER_ARGS);
int allgather_swing_static(ALLGATHER_ARGS);

int bcast_scatter_allgather(BCAST_ARGS);

int reduce_scatter_recursivehalving(REDUCE_SCATTER_ARGS);
int reduce_scatter_ring(REDUCE_SCATTER_ARGS);
int reduce_scatter_butterfly(REDUCE_SCATTER_ARGS);
int reduce_scatter_swing_static(REDUCE_SCATTER_ARGS);

#endif
