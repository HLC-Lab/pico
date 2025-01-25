#include <stdio.h>
#include <string.h>
#include <mpi.h>

#include "test_utils.h"


int reduce_scatter_gt_check(REDUCE_SCATTER_ARGS, void *recvbuf_gt) {
  // Compute the ground-truth result using PMPI_Reduce_scatter.
  PMPI_Reduce_scatter(sbuf, rbuf, rcounts, dtype, op, comm);

  int rank, comm_sz;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &comm_sz);
  
  int type_size;
  MPI_Type_size(dtype, &type_size);

  if (dtype != MPI_DOUBLE && dtype != MPI_FLOAT) {
    // For non-floating-point types, use memcmp for exact comparison.
    if (memcmp(rbuf, recvbuf_gt, rcounts[rank] * type_size) != 0) {
      fprintf(stderr, "Error: results are not valid. Aborting...\n");
      return -1;
    }
  } else {
    // For floating-point types, use an epsilon-based comparison to account for rounding errors.
    if (are_equal_eps(recvbuf_gt, rbuf, rcounts[rank], dtype, comm_sz) == -1) {
      fprintf(stderr, "Error: results are not valid. Aborting...\n");
      return -1;
    }
  }

  return 0; // Success.
}
