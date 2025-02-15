#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "test_utils.h"

int reduce_scatter_allocator(void **sbuf, void **rbuf, void **rbuf_gt, size_t count,
                             size_t type_size, MPI_Comm comm) {
  int comm_sz;
  MPI_Comm_size(comm, &comm_sz);
  
  // sbuf must contain only the data specific to the current rank,
  // while rbuf (and rbuf_gt) must contain the data from all ranks.
  *sbuf = (char *)malloc(count * type_size);
  *rbuf = (char *)calloc((count / (size_t) comm_sz), type_size);
  *rbuf_gt = (char *)calloc((count / (size_t) comm_sz), type_size);
  if (*sbuf == NULL || *rbuf == NULL || *rbuf_gt == NULL) {
    fprintf(stderr, "Error: Memory allocation failed. Aborting...\n");
    return -1;
  }
  return 0; // Success
}

int reduce_scatter_gt_check(REDUCE_SCATTER_ARGS, void *rbuf_gt) {
  // Compute the ground-truth result using PMPI_Reduce_scatter.
  PMPI_Reduce_scatter(sbuf, rbuf_gt, rcounts, dtype, op, comm);

  int rank, type_size;
  MPI_Comm_rank(comm, &rank);
  MPI_Type_size(dtype, &type_size);

  GT_CHECK_BUFFER(rbuf, rbuf_gt, rcounts[rank], dtype, comm);

  return 0; // Success.
}
