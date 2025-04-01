#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "bench_utils.h"

int scatter_allocator(void **sbuf, void **rbuf, void **rbuf_gt, size_t count,
                             size_t type_size, MPI_Comm comm) {
  int comm_sz, rank;
  MPI_Comm_size(comm, &comm_sz);
  MPI_Comm_rank(comm, &rank);

  if (rank == 0) {
    *sbuf = (char *) malloc(count * type_size);
    if (*sbuf == NULL) {
      fprintf(stderr, "Error: Memory allocation failed. Aborting...");
      return -1;
    }
  }
  *rbuf    = (char *) calloc((count / (size_t) comm_sz), type_size);
  *rbuf_gt = (char *) calloc((count / (size_t) comm_sz), type_size);
  if (*rbuf == NULL || *rbuf_gt == NULL) {
    fprintf(stderr, "Error: Memory allocation failed. Aborting...");
    return -1;
  }

  return 0; // Success
}
