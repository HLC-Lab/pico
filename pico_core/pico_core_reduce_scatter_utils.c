/*
 * Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
 * Licensed under the MIT License
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "pico_core_utils.h"

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

#if defined PICO_MPI_CUDA_AWARE || defined PICO_NCCL

int reduce_scatter_allocator_cuda(void **d_sbuf, void **d_rbuf, void **d_rbuf_gt, size_t count,
                             size_t type_size, MPI_Comm comm) {
  int comm_sz;
  MPI_Comm_size(comm, &comm_sz);
  cudaError_t err;

  PICO_CORE_CUDA_CHECK(cudaMalloc(d_sbuf, count * type_size), err);

  PICO_CORE_CUDA_CHECK(cudaMalloc(d_rbuf, (count / (size_t) comm_sz) * type_size), err);
  PICO_CORE_CUDA_CHECK(cudaMemset(*d_rbuf, 0, (count / (size_t) comm_sz) * type_size), err);

  PICO_CORE_CUDA_CHECK(cudaMalloc(d_rbuf_gt, (count / (size_t) comm_sz) * type_size), err);
  PICO_CORE_CUDA_CHECK(cudaMemset(*d_rbuf_gt, 0, (count / (size_t) comm_sz) * type_size), err);

  return 0; // Success
}


#endif
