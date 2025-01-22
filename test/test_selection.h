#ifndef TEST_SELECTION_H
#define TEST_SELECTION_H

#include <mpi.h>
#include "libswing.h"

/**
 * @enum allreduce_algo_t
 *
 * @brief Defines the standard Allreduce algorithms implemented in Open MPI coll module,
 * swing algorithms in Ompi Test repo and algorithms in `libswing.a`
 *
 * It only provides symbolic names for algorithm selection, conditional checks (use Open MPI
 * or not, use Ompi_Test...) must be done in selection stage. 
 *
 * TODO: implement conditional checks for OMPI vs MPICH
 * */
typedef enum{
  DEFAULT = 0,
  LINEAR,
  NON_OVERLAPPING,
  RECURSIVE_DOUBLING,
  RING,
  RING_SEGMENTED,
  RABENSEIFNER,
  ALLGATHER_REDUCE,
#ifdef OMPI_TEST
  SWING_LAT = 8,
  SWING_BDW_MEMCPY,
  SWING_BDW_DT_1,
  SWING_BDW_DT_2,
  SWING_BDW_SEG,
  SWING_BDW_STATIC,
#endif
  RECURSIVE_DOUBLING_OVER = 14,
  SWING_LAT_OVER,
  SWING_BDW_STATIC_OVER

} allreduce_algo_t;


/**
 * @typedef allreduce_func_ptr
 * 
 * A function pointer type for custom allreduce functions. It has the same signature
 * of MPI_Allreduce but with `count` being `size_t`.
 * 
 * @param sendbuf  Pointer to the buffer with data to send.
 * @param recvbuf  Pointer to the buffer where the result will be stored.
 * @param count    Number of elements to reduce (size_t).
 * @param datatype MPI datatype of the elements.
 * @param op       MPI reduction operation.
 * @param comm     MPI communicator.
 * 
 * @return Status code (int), typically MPI_SUCCESS on success.
 */
typedef int (*allreduce_func_ptr)(const void*, void*, size_t, MPI_Datatype, MPI_Op, MPI_Comm);


/**
 * @brief Wrapper function to adapt MPI_Allreduce to match a function signature using size_t.
 * 
 * This function calls MPI_Allreduce, casting the count parameter to int to match MPI's API.
 */
static inline int allreduce_wrapper(const void *sendbuf, void *recvbuf, size_t count,
                                    MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
    return MPI_Allreduce(sendbuf, recvbuf, (int)count, datatype, op, comm);
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
 * @return Pointer to the selected allreduce function. -1 for invalid selection
 */
static inline allreduce_func_ptr select_algorithm(allreduce_algo_t algorithm) {
  switch (algorithm) {
    case RECURSIVE_DOUBLING_OVER:
      return allreduce_recursivedoubling;
    case SWING_LAT_OVER:
      return allreduce_swing_lat;
    case SWING_BDW_STATIC_OVER:
      return allreduce_swing_bdw_static;
    default:
      return allreduce_wrapper;
  }
}

#endif
