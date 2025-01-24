#include <mpi.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "libswing.h"
#include "test_utils.h"


/**
 * @brief Compares two buffers with an epsilon tolerance for float or double datatypes.
 *
 * @param buf_1 First buffer.
 * @param buf_2 Second buffer.
 * @param buffer_count Size of the buffers.
 * @param dtype MPI_Datatype of the recvbuf.
 * @param comm_sz Communication size to scale the epsilon.
 * @return 0 if buffers are equal within tolerance, -1 otherwise.
 */
static inline int are_equal_eps(const void *buf_1, const void *buf_2, size_t buffer_count, const MPI_Datatype dtype, int comm_sz) {
  if (buffer_count == 0) return 0;

  size_t i;

  if (MPI_FLOAT == dtype) {
    float *b1 = (float *) buf_1;
    float *b2 = (float *) buf_2;

    float epsilon = comm_sz * TEST_BASE_EPSILON_FLOAT * 100.0f;

    for (i = 0; i < buffer_count; i++) {
      if (fabs(b1[i] - b2[i]) > epsilon) {
        return -1;
      }
    }
  } else if (MPI_DOUBLE == dtype) {
    double *b1 = (double *) buf_1;
    double *b2 = (double *) buf_2;

    double epsilon = comm_sz * TEST_BASE_EPSILON_DOUBLE * 100.0;

    for (i = 0; i < buffer_count; i++) {
      if (fabs(b1[i] - b2[i]) > epsilon) {
        return -1;
      }
    }
  }

  return 0;
}


/**
 * @brief Performs a ground-truth check for the result of an MPI Allreduce operation.
 *
 * This function computes the ground-truth result of an Allreduce operation using PMPI_Allreduce
 * and compares it with the result from the last iteration to ensure correctness. It supports both
 * integer and floating-point data types, with special handling for rounding errors in floating-point
 * arithmetic.
 *
 * @param ALLREDUCE_ARGS Macro containing the input arguments for Allreduce
 *        (sbuf, rbuf, count, dtype, op, comm).
 * @param recvbuf_gt A pointer to the buffer that will store the ground-truth result.
 * @return int Returns 0 on success, -1 if there is a mismatch or an error in type handling.
 */
int allreduce_gt_check(ALLREDUCE_ARGS, void *recvbuf_gt) {
  // Compute the ground-truth result using PMPI_Allreduce.
  PMPI_Allreduce(sbuf, recvbuf_gt, count, dtype, op, comm);

  int comm_sz;
  MPI_Comm_size(comm, &comm_sz);

  int type_size;
  MPI_Type_size(dtype, &type_size);

  if (dtype != MPI_DOUBLE && dtype != MPI_FLOAT) {
    // For non-floating-point types, use memcmp for exact comparison.
    if (memcmp(rbuf, recvbuf_gt, count * (size_t) type_size) != 0) {
      fprintf(stderr, "Error: results are not valid. Aborting...\n");
      return -1;
    }
  } else {
    // For floating-point types, use an epsilon-based comparison to account for rounding errors.
    if (are_equal_eps(recvbuf_gt, rbuf, count, dtype, comm_sz) == -1) {
      fprintf(stderr, "Error: results are not valid. Aborting...\n");
      return -1;
    }
  }

  return 0; // Success.
}

/**
 * @brief Performs a ground-truth check for the result of an MPI Allgather operation.
 *
 * This function computes the ground-truth result of an Allgather operation using PMPI_Allgather
 * and compares it with the result from the last iteration to ensure correctness. It supports both
 * integer and floating-point data types, with special handling for rounding errors in floating-point
 * arithmetic.
 *
 * @param ALLGATHER_ARGS Macro containing the input arguments for Allgather
 *        (sbuf, scount, sdtype, rbuf, rcount, rdtype, comm).
 * @param recvbuf_gt A pointer to the buffer that will store the ground-truth result.
 * @return int Returns 0 on success, -1 if there is a mismatch or an error in type handling.
 */
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

/**
 * @brief Performs a ground-truth check for the result of a Reduce Scatter operation.
 *
 * This function computes the ground-truth result of a Reduce Scatter operation using PMPI_Reduce_scatter
 * and compares it with the result from the last iteration to ensure correctness. It supports both
 * integer and floating-point data types, with special handling for rounding errors in floating-point
 * arithmetic.
 *
 * @param REDUCE_SCATTER_ARGS Macro containing the input arguments for Reduce Scatter
 *        (sbuf, scount, sdtype, rbuf, rcount, rdtype, comm).
 * @param recvbuf_gt A pointer to the buffer that will store the ground-truth result.
 * @return int Returns 0 on success, -1 if there is a mismatch or an error in type handling.
 */
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
