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
  *rbuf = (char *)malloc(count * type_size);
  *rbuf_gt = (char *)malloc(count * type_size);
  if (*sbuf == NULL || *rbuf == NULL || *rbuf_gt == NULL) {
    fprintf(stderr, "Error: Memory allocation failed. Aborting...\n");
    return -1;
  }
  return 0; // Success
}

/**
 * @brief Selects the appropriate allgather algorithm based on an algorithm number.
 * 
 * This function returns a pointer to the appropriate allgather function based on the provided
 * algorithm enum. If the algorithm number does not match any custom implementation, it returns
 * a pointer to the allgather_wrapper function by default.
 * 
 * @param algorithm The algorithm enum specifying which allgather function to use.
 * 
 * @return Pointer to the selected allgather function.
 *
 * WARNING: This function does not check if the algorithm number is valid. It is the caller's
 * responsibility
 */
static inline allgather_func_ptr get_allgather_function(allgather_algo_t algorithm) {
  switch (algorithm) {
    case ALLGATHER_RECURSIVE_DOUBLING_OVER:
      return allgather_recursivedoubling;
    case ALLGATHER_RING_OVER:
      return allgather_ring;
    case ALLGATHER_SWING_STATIC_OVER:
      return allgather_swing_static;
    default:
      return allgather_wrapper;
  }
}


void allgather_test_loop(ALLGATHER_ARGS, int iter, double *times, allgather_algo_t algorithm){
  allgather_func_ptr allgather_func = get_allgather_function(algorithm);

  double start_time, end_time;

  for (int i = 0; i < iter; i++) {
    MPI_Barrier(comm);
    start_time = MPI_Wtime();
    allgather_func(sbuf, scount, sdtype, rbuf, rcount, rdtype, comm);
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

