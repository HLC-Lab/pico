#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "bench_utils.h"

//#ifndef CUDA_AWARE

int bcast_allocator(void** sbuf, void** rbuf, void** rbuf_gt,
                        size_t count, size_t type_size, MPI_Comm comm) {
  *sbuf = (char *)calloc(count, type_size);
  *rbuf_gt = (char *)calloc(count, type_size);

  if(*sbuf == NULL || *rbuf_gt == NULL) {
    fprintf(stderr, "Error: Memory allocation failed. Aborting...");
    return -1;
  }

  return 0; // Success
}
/*
#else

int bcast_allocator(void** sbuf, void** rbuf, void** rbuf_gt,
  size_t count, size_t type_size, MPI_Comm comm) {

  CUDA_CHECK(cudaMalloc(sbuf, count * type_size));
  CUDA_CHECK(cudaMalloc(rbuf_gt, count * type_size));

  return 0; // Success
}

#endif
*/