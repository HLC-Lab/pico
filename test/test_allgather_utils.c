#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "libswing.h"
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


void allgather_test_loop(ALLGATHER_ARGS, int iter, double *times, test_routine_t test_routine) {
  double start_time, end_time;

  for (int i = 0; i < iter; i++) {
    MPI_Barrier(comm);
    start_time = MPI_Wtime();
    test_routine.function.allgather(sbuf, scount, sdtype, rbuf, rcount, rdtype, comm);
    end_time = MPI_Wtime();
    times[i] = end_time - start_time;
  }
}


int allgather_gt_check(ALLGATHER_ARGS, void *rbuf_gt) {
  // Compute the ground-truth result using PMPI_Allgather.
  PMPI_Allgather(sbuf, scount, sdtype, rbuf_gt, rcount, rdtype, comm);

  int comm_sz;
  MPI_Comm_size(comm, &comm_sz);

  int type_size;
  MPI_Type_size(rdtype, &type_size);

  if (rdtype != MPI_DOUBLE && rdtype != MPI_FLOAT) {
    // For non-floating-point types, use memcmp for exact comparison.
    if (memcmp(rbuf, rbuf_gt, rcount * (size_t) (type_size * comm_sz)) != 0) {
      #ifdef DEBUG
      debug_print_buffers(rbuf, rbuf_gt, rcount * (size_t) comm_sz, rdtype, comm);
      #endif
      fprintf(stderr, "Error: results are not valid. Aborting...\n");
      return -1;
    }
  } else {
    // For floating-point types, use an epsilon-based comparison to account for rounding errors.
    if (are_equal_eps(rbuf_gt, rbuf, rcount * (size_t) comm_sz, rdtype, comm_sz) == -1) {
      #ifdef DEBUG
      debug_print_buffers(rbuf, rbuf_gt, rcount * (size_t) comm_sz, rdtype, comm);
      #endif
      fprintf(stderr, "Error: results are not valid. Aborting...\n");
      return -1;
    }
  }

  return 0; // Success.
}

