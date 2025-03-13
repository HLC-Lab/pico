#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "bench_utils.h"

//#ifndef CUDA_AWARE

int reduce_scatter_allocator(void **sbuf, void **rbuf, void **rbuf_gt, size_t count,
                             size_t type_size, MPI_Comm comm) {
  int comm_sz;
  MPI_Comm_size(comm, &comm_sz);
  
  // sbuf must contain only the data specific to the current rank,
  // while rbuf (and rbuf_gt) must contain the data from all ranks.
  *sbuf = (char *)malloc(count * type_size);
  *rbuf = (char *)calloc((count / (size_t) comm_sz), type_size);
  *rbuf_gt = (char *)calloc((count / (size_t) comm_sz), type_size);
  if(*sbuf == NULL || *rbuf == NULL || *rbuf_gt == NULL) {
    fprintf(stderr, "Error: Memory allocation failed. Aborting...");
    return -1;
  }
  return 0; // Success
}

/*
#else

int reduce_scatter_allocator(void **sbuf, void **rbuf, void **rbuf_gt, size_t count,
  size_t type_size, MPI_Comm comm) {
  int comm_sz;
  MPI_Comm_size(comm, &comm_sz);

  // sbuf must contain only the data specific to the current rank,
  // while rbuf (and rbuf_gt) must contain the data from all ranks.
  CUDA_CHECK(cudaMalloc(sbuf, count * type_size));
  CUDA_CHECK(cudaMalloc(rbuf, count * type_size));
  CUDA_CHECK(cudaMalloc(rbuf_gt, count * type_size));

  return 0; // Success
}

#endif
*/
