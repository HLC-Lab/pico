#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "libswing.h"
#include "test_utils.h"

int allreduce_allocator(void** sbuf, void** rbuf, void** rbuf_gt,
                        size_t count, size_t type_size, MPI_Comm comm) {
  *sbuf = (char *)malloc(count * type_size);
  *rbuf = (char *)malloc(count * type_size);
  *rbuf_gt = (char *)malloc(count * type_size);

  if (*sbuf == NULL || *rbuf == NULL || *rbuf_gt == NULL) {
    fprintf(stderr, "Error: Memory allocation failed. Aborting...\n");
    return -1;
  }

  return 0; // Success
}

/**
 * @brief Selects the appropriate allreduce algorithm based on an algorithm number.
 * 
 * This function returns a pointer to the appropriate allreduce function based on the provided
 * algorithm enum. If the algorithm number does not match any custom implementation, it returns
 * a pointer to the allreduce_wrapper function by default.
 * 
 * @param algorithm The algorithm enum specifying which allreduce function to use.
 * 
 * @return Pointer to the selected allreduce function.
 *
 * WARNING: This function does not check if the algorithm number is valid. It is the caller's
 * responsibility
 */
static inline allreduce_func_ptr get_allreduce_function(allreduce_algo_t algorithm) {
  switch (algorithm) {
    case ALLREDUCE_RECURSIVE_DOUBLING_OVER:
      return allreduce_recursivedoubling;
    case ALLREDUCE_SWING_LAT_OVER:
      return allreduce_swing_lat;
    case ALLREDUCE_SWING_BDW_STATIC_OVER:
      return allreduce_swing_bdw_static;
    default:
      return allreduce_wrapper;
  }
}

void allreduce_test_loop(ALLREDUCE_ARGS, int iter, double *times, allreduce_algo_t algorithm) {
  allreduce_func_ptr allreduce_func = get_allreduce_function(algorithm);

  double start_time, end_time;

  for (int i = 0; i < iter; i++) {
    MPI_Barrier(comm);
    start_time = MPI_Wtime();
    allreduce_func(sbuf, rbuf, count, dtype, MPI_SUM, comm);
    end_time = MPI_Wtime();
    times[i] = end_time - start_time;
  }
}

int allreduce_gt_check(ALLREDUCE_ARGS, void *rbuf_gt) {
  // Compute the ground-truth result using PMPI_Allreduce.
  PMPI_Allreduce(sbuf, rbuf_gt, count, dtype, op, comm);

  int comm_sz;
  MPI_Comm_size(comm, &comm_sz);

  int type_size;
  MPI_Type_size(dtype, &type_size);

  if (dtype != MPI_DOUBLE && dtype != MPI_FLOAT) {
    // For non-floating-point types, use memcmp for exact comparison.
    if (memcmp(rbuf, rbuf_gt, count * (size_t) type_size) != 0) {
      #ifdef DEBUG
      debug_print_buffers(rbuf, rbuf_gt, count, dtype, comm);
      #endif
      fprintf(stderr, "Error: results are not valid. Aborting...\n");
      return -1;
    }
  } else {
    // For floating-point types, use an epsilon-based comparison to account for rounding errors.
    if (are_equal_eps(rbuf_gt, rbuf, count, dtype, comm_sz) == -1) {
      #ifdef DEBUG
      debug_print_buffers(rbuf, rbuf_gt, count, dtype, comm);
      #endif
      fprintf(stderr, "Error: results are not valid. Aborting...\n");
      return -1;
    }
  }

  return 0; // Success.
}

