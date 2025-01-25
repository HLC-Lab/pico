#include <stdio.h>
#include <string.h>
#include <mpi.h>

#include "libswing.h"
#include "test_utils.h"

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


int allgather_gt_check(ALLGATHER_ARGS, void *recvbuf_gt) {
  // Compute the ground-truth result using PMPI_Allgather.
  PMPI_Allgather(sbuf, scount, sdtype, rbuf, rcount, rdtype, comm);

  int comm_sz;
  MPI_Comm_size(comm, &comm_sz);

  int type_size;
  MPI_Type_size(rdtype, &type_size);

  if (rdtype != MPI_DOUBLE && rdtype != MPI_FLOAT) {
    // For non-floating-point types, use memcmp for exact comparison.
    if (memcmp(rbuf, recvbuf_gt, rcount * (size_t) (type_size * comm_sz)) != 0) {
      fprintf(stderr, "Error: results are not valid. Aborting...\n");
      return -1;
    }
  } else {
    // For floating-point types, use an epsilon-based comparison to account for rounding errors.
    if (are_equal_eps(recvbuf_gt, rbuf, rcount * (size_t) comm_sz, rdtype, comm_sz) == -1) {
      fprintf(stderr, "Error: results are not valid. Aborting...\n");
      return -1;
    }
  }

  return 0; // Success.
}

