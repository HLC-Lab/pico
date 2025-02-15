#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "test_utils.h"

int allgather_allocator(void **sbuf, void **rbuf, void **rbuf_gt, size_t count,
                        size_t type_size, MPI_Comm comm) {
  int comm_sz;
  MPI_Comm_size(comm, &comm_sz);
  
  // sbuf must contain only the data specific to the current rank,
  // while rbuf (and rbuf_gt) must contain the data from all ranks.
  *sbuf = (char *)malloc((count / (size_t) comm_sz) * type_size );
  *rbuf = (char *)calloc(count, type_size);
  *rbuf_gt = (char *)calloc(count, type_size);
  if (*sbuf == NULL || *rbuf == NULL || *rbuf_gt == NULL) {
    fprintf(stderr, "Error: Memory allocation failed. Aborting...\n");
    return -1;
  }
  return 0; // Success
}



int allgather_gt_check(ALLGATHER_ARGS, void *rbuf_gt) {
  // Compute the ground-truth result using PMPI_Allgather.
  PMPI_Allgather(sbuf, scount, sdtype, rbuf_gt, rcount, rdtype, comm);

  int comm_sz, type_size;
  MPI_Comm_size(comm, &comm_sz);
  MPI_Type_size(rdtype, &type_size);

  GT_CHECK_BUFFER(rbuf, rbuf_gt, rcount * comm_sz, rdtype, comm);

  return 0; // Success.
}

