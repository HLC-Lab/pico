#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "test_utils.h"

int bcast_allocator(void** sbuf, void** rbuf, void** rbuf_gt,
                        size_t count, size_t type_size, MPI_Comm comm) {
  *sbuf = (char *)calloc(count, type_size);
  *rbuf_gt = (char *)calloc(count, type_size);

  if (*sbuf == NULL || *rbuf_gt == NULL) {
    fprintf(stderr, "Error: Memory allocation failed. Aborting...\n");
    return -1;
  }

  return 0; // Success
}


int bcast_gt_check(BCAST_ARGS, void *buf_gt) {
  // Compute the ground-truth result using PMPI_Bcast.
  int rank, type_size;
  MPI_Comm_rank(comm, &rank);
  MPI_Type_size(dtype, &type_size);

  if (rank == root) {
    memcpy(buf_gt, buf, count * type_size);
  }
  PMPI_Bcast(buf_gt, count, dtype, root, comm);

  GT_CHECK_BUFFER(buf, buf_gt, count, dtype, comm);

  return 0; // Success.
}
