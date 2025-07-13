#ifndef LIBBINE_H
#define LIBBINE_H

#include <mpi.h>
#include <stddef.h>

#define ALLREDUCE_ARGS        const void *sbuf, void *rbuf, size_t count, \
                              MPI_Datatype dtype, MPI_Op op, MPI_Comm comm
#define ALLGATHER_ARGS        const void *sbuf, size_t scount, MPI_Datatype sdtype, \
                              void* rbuf, size_t rcount, MPI_Datatype rdtype, MPI_Comm comm
#define ALLTOALL_ARGS         const void *sbuf, size_t scount, MPI_Datatype sdtype, \
                              void *rbuf, size_t rcount, MPI_Datatype rdtype, MPI_Comm comm
#define BCAST_ARGS            void *buf, size_t count, MPI_Datatype dtype, int root, MPI_Comm comm
#define GATHER_ARGS           const void *sbuf, size_t scount, MPI_Datatype sdtype, \
                              void *rbuf, size_t rcount, MPI_Datatype rdtype, int root, MPI_Comm comm
#define REDUCE_ARGS           const void *sbuf, void *rbuf, size_t count, \
                              MPI_Datatype dtype, MPI_Op op, int root, MPI_Comm comm
#define REDUCE_SCATTER_ARGS   const void *sbuf, void *rbuf, const int rcounts[], \
                              MPI_Datatype dtype, MPI_Op op, MPI_Comm comm
#define SCATTER_ARGS          const void *sbuf, size_t scount, MPI_Datatype sdtype, \
                              void *rbuf, size_t rcount, MPI_Datatype rdtype, int root, MPI_Comm comm

extern size_t bine_allreduce_segsize;

int allreduce_recursivedoubling(ALLREDUCE_ARGS);
int allreduce_ring(ALLREDUCE_ARGS);
int allreduce_rabenseifner(ALLREDUCE_ARGS);
int allreduce_bine_lat(ALLREDUCE_ARGS);
int allreduce_bine_bdw_static(ALLREDUCE_ARGS);
int allreduce_bine_bdw_remap(ALLREDUCE_ARGS);
int allreduce_bine_bdw_remap_segmented(ALLREDUCE_ARGS);
int allreduce_bine_block_by_block_any_even(ALLREDUCE_ARGS);

int allgather_k_bruck(ALLGATHER_ARGS);
int allgather_recursivedoubling(ALLGATHER_ARGS);
int allgather_ring(ALLGATHER_ARGS);
int allgather_sparbit(ALLGATHER_ARGS);
int allgather_bine_block_by_block(ALLGATHER_ARGS);
int allgather_bine_block_by_block_any_even(ALLGATHER_ARGS);
int allgather_bine_permute_static(ALLGATHER_ARGS);
int allgather_bine_send_static(ALLGATHER_ARGS);
int allgather_bine_permute_remap(ALLGATHER_ARGS);
int allgather_bine_send_remap(ALLGATHER_ARGS);
int allgather_bine_2_blocks(ALLGATHER_ARGS);
int allgather_bine_2_blocks_dtype(ALLGATHER_ARGS);

int alltoall_bine(ALLTOALL_ARGS);

int bcast_scatter_allgather(BCAST_ARGS);
int bcast_bine_lat(BCAST_ARGS);
int bcast_bine_lat_reversed(BCAST_ARGS);
int bcast_bine_lat_new(BCAST_ARGS);
int bcast_bine_lat_i_new(BCAST_ARGS);
int bcast_bine_bdw_static(BCAST_ARGS);
int bcast_bine_bdw_remap(BCAST_ARGS);
// int bcast_bine_bdw_static_reversed(BCAST_ARGS);

int gather_bine(GATHER_ARGS);

int reduce_bine_lat(REDUCE_ARGS);
int reduce_bine_bdw(REDUCE_ARGS);

int reduce_scatter_recursivehalving(REDUCE_SCATTER_ARGS);
int reduce_scatter_recursive_distance_doubling(REDUCE_SCATTER_ARGS);
int reduce_scatter_ring(REDUCE_SCATTER_ARGS);
int reduce_scatter_butterfly(REDUCE_SCATTER_ARGS);
int reduce_scatter_bine_static(REDUCE_SCATTER_ARGS);
int reduce_scatter_bine_send_remap(REDUCE_SCATTER_ARGS);
int reduce_scatter_bine_permute_remap(REDUCE_SCATTER_ARGS);
int reduce_scatter_bine_block_by_block(REDUCE_SCATTER_ARGS);
int reduce_scatter_bine_block_by_block_any_even(REDUCE_SCATTER_ARGS);

int scatter_bine(SCATTER_ARGS);

#endif
