#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "bench_utils.h"


int allgather_allocator(void **sbuf, void **rbuf, void **rbuf_gt, size_t count,
                        size_t type_size, MPI_Comm comm) {
  int comm_sz;
  MPI_Comm_size(comm, &comm_sz);
  
  // rbuf must contain only the data specific to the current rank,
  // while sbuf must contain the data from all ranks.
  *sbuf = (char *)malloc((count / (size_t) comm_sz) * type_size );
  *rbuf = (char *)calloc(count, type_size);
  *rbuf_gt = (char *)calloc(count, type_size);
  if(*sbuf == NULL || *rbuf == NULL || *rbuf_gt == NULL) {
    fprintf(stderr, "Error: Memory allocation failed. Aborting...");
    return -1;
  }
  return 0; // Success
}

#ifdef CUDA_AWARE

int allgather_allocator_cuda(void **d_sbuf, void **d_rbuf, void **d_rbuf_gt, size_t count,
                        size_t type_size, MPI_Comm comm) {
  int comm_sz;
  MPI_Comm_size(comm, &comm_sz);
  cudaError_t err;

  BENCH_CUDA_CHECK(cudaMalloc(d_sbuf, (count / (size_t) comm_sz) * type_size), err);

  BENCH_CUDA_CHECK(cudaMalloc(d_rbuf, count * type_size), err);
  BENCH_CUDA_CHECK(cudaMemset(*d_rbuf, 0, count * type_size), err);

  BENCH_CUDA_CHECK(cudaMalloc(d_rbuf_gt, count * type_size), err);
  BENCH_CUDA_CHECK(cudaMemset(*d_rbuf_gt, 0, count * type_size), err);

  return 0; // Success
}

#endif
