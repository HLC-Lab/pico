#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "pico_core_utils.h"

int gather_allocator(void **sbuf, void **rbuf, void **rbuf_gt, size_t count,
                             size_t type_size, MPI_Comm comm) {
  int comm_sz, rank;
  MPI_Comm_size(comm, &comm_sz);
  MPI_Comm_rank(comm, &rank);

  *sbuf = (char *)malloc((count / (size_t) comm_sz) * type_size);
  if (*sbuf == NULL) {
    fprintf(stderr, "Error: Memory allocation failed. Aborting...");
    return -1;
  }
  if (rank == 0){
    *rbuf = (char *) calloc(count, type_size);
    *rbuf_gt = (char *) calloc(count, type_size);
    if(*rbuf == NULL || *rbuf_gt == NULL) {
      fprintf(stderr, "Error: Memory allocation failed. Aborting...");
      return -1;
    }
  }
  return 0; // Success
}

#ifdef CUDA_AWARE

int gather_allocator_cuda(void **d_sbuf, void **d_rbuf, void **d_rbuf_gt, size_t count,
                             size_t type_size, MPI_Comm comm) {
  int comm_sz, rank;
  MPI_Comm_size(comm, &comm_sz);
  MPI_Comm_rank(comm, &rank);
  cudaError_t err;

  PICO_CORE_CUDA_CHECK(cudaMalloc(d_sbuf, (count / (size_t) comm_sz) * type_size), err);

  if (rank == 0){
    PICO_CORE_CUDA_CHECK(cudaMalloc(d_rbuf, count * type_size), err);
    PICO_CORE_CUDA_CHECK(cudaMemset(*d_rbuf, 0, count * type_size), err);

    PICO_CORE_CUDA_CHECK(cudaMalloc(d_rbuf_gt, count * type_size), err);
    PICO_CORE_CUDA_CHECK(cudaMemset(*d_rbuf_gt, 0, count * type_size), err);
  }

  return 0; // Success
}

#endif
