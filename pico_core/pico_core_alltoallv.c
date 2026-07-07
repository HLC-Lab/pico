#include <stdlib.h>
#include "pico_core_utils.h"

int alltoallv_allocator(ALLOCATOR_ARGS) {
  *sbuf = malloc(count * type_size);
  *rbuf = malloc(count * type_size);
  *rbuf_gt = malloc(count * type_size);

  if (*sbuf == NULL || *rbuf == NULL || *rbuf_gt == NULL) {
    fprintf(stderr, "Error: failed to allocate host buffers for alltoallv\n");
    return -1;
  }

  return 0;
}

#if defined PICO_MPI_CUDA_AWARE || defined PICO_NCCL
int alltoallv_allocator_cuda(ALLOCATOR_ARGS) {
  cudaError_t err;

  PICO_CORE_CUDA_CHECK(cudaMalloc(d_sbuf, count * type_size), err);
  if (err != cudaSuccess) return -1;

  PICO_CORE_CUDA_CHECK(cudaMalloc(d_rbuf, count * type_size), err);
  if (err != cudaSuccess) return -1;

  PICO_CORE_CUDA_CHECK(cudaMalloc(d_rbuf_gt, count * type_size), err);
  if (err != cudaSuccess) return -1;

  return 0;
}
#endif