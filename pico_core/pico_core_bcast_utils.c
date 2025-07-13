#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "pico_core_utils.h"

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

#ifdef CUDA_AWARE

int bcast_allocator(void** d_sbuf, void** d_rbuf, void** d_rbuf_gt,
                        size_t count, size_t type_size, MPI_Comm comm) {
  cudaError_t err;
  PICO_CORE_CUDA_CHECK(cudaMalloc(d_sbuf, count * type_size), err);
  PICO_CORE_CUDA_CHECK(cudaMemset(*d_sbuf, 0, count * type_size), err);

  PICO_CORE_CUDA_CHECK(cudaMalloc(d_rbuf_gt, count * type_size), err);
  PICO_CORE_CUDA_CHECK(cudaMemset(*d_rbuf_gt, 0, count * type_size), err);

  return 0; // Success
}

#endif
