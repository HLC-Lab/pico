#include <stdlib.h>
#include "pico_core_utils.h"

int alltoallv_allocator(ALLOCATOR_ARGS) {
  int comm_sz;
  MPI_Comm_size(comm, &comm_sz);

  *sbuf = malloc(count * type_size);
  *rbuf = malloc(count * type_size);
  *rbuf_gt = malloc(count * type_size);

  if (*sbuf == NULL || *rbuf == NULL || *rbuf_gt == NULL) {
    fprintf(stderr, "Error: failed to allocate buffers for alltoallv\n");
    return -1;
  }

  return 0;
}