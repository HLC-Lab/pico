#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "test_utils.h"

int allreduce_allocator(void** sbuf, void** rbuf, void** rbuf_gt,
                        size_t count, size_t type_size, MPI_Comm comm) {
  *sbuf = (char *)malloc(count * type_size);
  *rbuf = (char *)calloc(count, type_size);
  *rbuf_gt = (char *)calloc(count, type_size);

  if (*sbuf == NULL || *rbuf == NULL || *rbuf_gt == NULL) {
    fprintf(stderr, "Error: Memory allocation failed. Aborting...\n");
    return -1;
  }

  return 0; // Success
}


int allreduce_gt_check(ALLREDUCE_ARGS, void *rbuf_gt) {
  // Compute the ground-truth result using PMPI_Allreduce.
  PMPI_Allreduce(sbuf, rbuf_gt, count, dtype, op, comm);

  int type_size;
  MPI_Type_size(dtype, &type_size);

  GT_CHECK_BUFFER(rbuf, rbuf_gt, count, dtype, comm);

  return 0; // Success.
}

