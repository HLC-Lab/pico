#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "test_utils.h"

int allgather_allocator(void **sbuf, void **rbuf, void **rbuf_gt, size_t count,
                        size_t type_size, MPI_Comm comm) {
  int comm_sz;
  MPI_Comm_size(comm, &comm_sz);
  
  // rbuf must contain only the data specific to the current rank,
  // while sbuf must contain the data from all ranks.
  *sbuf = (char *)malloc((count / (size_t) comm_sz) * type_size );
  *rbuf = (char *)calloc(count, type_size);
  *rbuf_gt = (char *)calloc(count, type_size);
  if (*sbuf == NULL || *rbuf == NULL || *rbuf_gt == NULL) {
    fprintf(stderr, "Error: Memory allocation failed. Aborting...\n");
    return -1;
  }
  return 0; // Success
}

