/**
 * @file test_selection.h
 * @brief Defines a structure for selecting collective operations and their corresponding algorithms.
 *
 * This structure is used to represent a decision regarding which collective 
 * operation (e.g., allreduce, allgather, reduce-scatter) to perform and which 
 * specific algorithm to use for that operation.
 */

#ifndef TEST_SELECTION_H
#define TEST_SELECTION_H

#include <mpi.h>

#include "libswing.h"

/**
 * @enum coll_t
 *
 * @brief Defines the collective operation to be used in this test. It only provide symbolic
 * name for collective selection.
 * */
typedef enum{
  ALLREDUCE = 0,
  ALLGATHER,
  REDUCE_SCATTER,
  COLL_UNKNOWN
}coll_t;


/**
 * @enum allreduce_algo_t
 *
 * @brief Defines the standard Allreduce algorithms implemented in Open MPI coll module,
 * swing algorithms in Ompi Test repo and algorithms in `libswing.a`. It only provides
 * symbolic names for algorithm selection.
 *
 * TODO: implement conditional checks for OMPI vs MPICH
 * */
typedef enum{
  ALLREDUCE_DEFAULT = 0,
  ALLREDUCE_LINEAR,
  ALLREDUCE_NON_OVERLAPPING,
  ALLREDUCE_RECURSIVE_DOUBLING,
  ALLREDUCE_RING,
  ALLREDUCE_RING_SEGMENTED,
  ALLREDUCE_RABENSEIFNER,
  ALLREDUCE_ALLGATHER_REDUCE,
#ifdef OMPI_TEST
  ALLREDUCE_SWING_LAT = 8,
  ALLREDUCE_SWING_BDW_MEMCPY,
  ALLREDUCE_SWING_BDW_DT_1,
  ALLREDUCE_SWING_BDW_DT_2,
  ALLREDUCE_SWING_BDW_SEG,
  ALLREDUCE_SWING_BDW_STATIC,
#endif
  ALLREDUCE_RECURSIVE_DOUBLING_OVER = 14,
  ALLREDUCE_SWING_LAT_OVER,
  ALLREDUCE_SWING_BDW_STATIC_OVER

} allreduce_algo_t;

/**
 * @enum allgather_algo_t
 *
 * @brief Defines the standard Allgather algorithms implemented in Open MPI coll module,
 * and algorithms in `libswing.a`. It only provides symbolic names for algorithm selection.
 * */
typedef enum{
  ALLGATHER_DEFAULT = 0,
  ALLGATHER_LINEAR,
  ALLGATHER_K_BRUCK,
  ALLGATHER_RECURSIVE_DOUBLING,
  ALLGATHER_RING,
  ALLGATHER_NEIGHBOR,
  ALLGATHER_TWO_PROC,
  ALLGATHER_K_BRUCK_OVER = 7,
  ALLGATHER_RECURSIVE_DOUBLING_OVER,
  ALLGATHER_RING_OVER,
  ALLGATHER_SWING_STATIC_OVER = 10,

}allgather_algo_t;

/**
 * @enum reduce_scatter_algo_t
 *
 * @brief Defines the standard Reduce Scatter algorithms implemented in Open MPI coll module,
 * and algorithms in `libswing.a`. It only provides symbolic names for algorithm selection.
 * */
typedef enum{
  REDUCE_SCATTER_DEFAULT = 0,
  REDUCE_SCATTER_NON_OVERLAPPING,
  REDUCE_SCATTER_RECURSIVE_HALVING,
  REDUCE_SCATTER_RING,
  REDUCE_SCATTER_BUTTERFLY,
  REDUCE_SCATTER_RECURSIVE_HALVING_OVER = 5,
  REDUCE_SCATTER_RING_OVER,
  REDUCE_SCATTER_BUTTERFLY_OVER,

}reduce_scatter_algo_t;


/**
 * @struct routine_decision_t
 * @brief Represents a decision about a collective operation and its corresponding algorithm.
 *
 * This structure allows the user to specify:
 * - The collective operation to be performed (`coll_t`).
 * - The specific algorithm to use for that operation, based on the collective type.
 */
typedef struct {
  coll_t collective; /**< Specifies the type of collective operation. */

  /**
    * @union algorithm
    * @brief Holds the specific algorithm for the selected collective operation.
    */
  union {
    allreduce_algo_t allreduce_algorithm;
    allgather_algo_t allgather_algorithm;
    reduce_scatter_algo_t reduce_scatter_algorithm;
  } algorithm;

} routine_decision_t;


int get_routine(routine_decision_t *test_routine, int algorithm);


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

#endif
