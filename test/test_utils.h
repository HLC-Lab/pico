#ifndef TEST_TOOLS_H
#define TEST_TOOLS_H

#include <mpi.h>

#include "libswing.h"

#define TEST_MAX_PATH_LENGTH 512
#define TEST_BASE_EPSILON_FLOAT 1e-6    // Base epsilon for float
#define TEST_BASE_EPSILON_DOUBLE 1e-15  // Base epsilon for double

//-----------------------------------------------------------------------------------------------
//                    TYPEDEFs AND ENUMs FOR COLLECTIVE SELECTION
// ----------------------------------------------------------------------------------------------

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

//-----------------------------------------------------------------------------------------------
//                    COLLECTIVE SPECIFIC MEMORY ALLOCATORS
//-----------------------------------------------------------------------------------------------

#define ALLOCATOR_ARGS    void **sbuf, void **rbuf, void **rbuf_gt, size_t count,\
                          size_t type_size, MPI_Comm comm
/**
* @typedef allocator_func_ptr
*
* A function pointer type for custom memory allocation functions.
*/
typedef int (*allocator_func_ptr)(ALLOCATOR_ARGS);


/**
* @brief Allocates memory for the send buffer, receive buffer,
* and ground-truth buffer for an Allreduce operation.
*
* @return 0 on success, -1 on error.
*/
int allreduce_allocator(ALLOCATOR_ARGS);

/**
* @brief Allocates memory for the send buffer, receive buffer,
* and ground-truth buffer for an Allgather operation.
*
* @return 0 on success, -1 on error.
*/
int allgather_allocator(ALLOCATOR_ARGS);


/**
* @brief Select and returns the appropriate allocator function based
* on the collective type. It returns NULL if the collective type is 
* not supported.
*
* @param test_routine `routine_decision_t` structure containing the
*                      collective type informations.
*
* @return Pointer to the selected allocator function, or NULL if the
*         collective type is not supported.
*
* WARNING: While `count` always represents the number of elements in the
* biggest buffer (be it send or receive), the actual memory allocation
* for each buffer is determined by the collective type.
*/
static inline allocator_func_ptr get_allocator(routine_decision_t test_routine) {
  switch (test_routine.collective) {
    case ALLREDUCE:
      return allreduce_allocator;
    case ALLGATHER:
      return allgather_allocator;
    // case REDUCE_SCATTER:
    //   return reduce_scatter_allocator;
    default:
      return NULL;
  }
}
//-----------------------------------------------------------------------------------------------
//                FUNCTION POINTER TYPES AND WRAPPER FOR CUSTOM COLLECTIVE FUNCTIONS
// ----------------------------------------------------------------------------------------------

/**
 * @typedef allreduce_func_ptr
 * 
 * A function pointer type for custom allreduce functions. It has the same signature
 * of MPI_Allreduce but with `count` being `size_t`.
 */
typedef int (*allreduce_func_ptr)(ALLREDUCE_ARGS);

/**
 * @typedef allgather_func_ptr
 * 
 * A function pointer type for custom allgather functions. It has the same signature
 * of MPI_Allgather but with `scount` and `rcount` being `size_t`.
 */
typedef int (*allgather_func_ptr)(ALLGATHER_ARGS);

/**
 * @typedef reduce_scatter_func_ptr
 * 
 * A function pointer type for custom reduce scatter functions. It has the same signature
 * of MPI_Reduce_scatter.
 */
typedef int (*reduce_scatter_func_ptr)(REDUCE_SCATTER_ARGS);

/**
 * @brief Wrapper function to adapt MPI_Allreduce to match a function signature using size_t.
 * 
 * This function calls MPI_Allreduce, casting the count parameter to int to match MPI's API.
 */
static inline int allreduce_wrapper(ALLREDUCE_ARGS){
    return MPI_Allreduce(sbuf, rbuf, (int)count, dtype, op, comm);
}

/**
* @brief Wrapper function to adapt MPI_Allgather to match a function signature using size_t.
* 
* This function calls MPI_Allgather, casting the scount and rcount parameters to int to match MPI's API.
*/
static inline int allgather_wrapper(ALLGATHER_ARGS){
    return MPI_Allgather(sbuf, (int)scount, sdtype, rbuf, (int)rcount, rdtype, comm);
}


//-----------------------------------------------------------------------------------------------
//                                MAIN BENCHMARK LOOP FUNCTIONS
//-----------------------------------------------------------------------------------------------

/**
 * @brief This fucntion benchmarks the allreduce operation using the selected algorithm.
 */
void allreduce_test_loop(ALLREDUCE_ARGS, int iter, double *times, allreduce_algo_t algorithm);

/**
 * @brief This fucntion benchmarks the allgather operation using the selected algorithm.
 */
void allgather_test_loop(ALLGATHER_ARGS, int iter, double *times, allgather_algo_t algorithm);

//TODO: IMPLEMENT THIS
/**
 * @brief This fucntion benchmarks the reduce scatter operation using the selected algorithm.
 */
// void reduce_scatter_test_loop(REDUCE_SCATTER_ARGS, int iter, double start_time, double end_time, double *times, reduce_scatter_algo_t algorithm);
//

//-----------------------------------------------------------------------------------------------
//                     SELECT ALGORITHM AND COMMAND LINE PARSING FUNCTIONS
//-----------------------------------------------------------------------------------------------

/**
 * @brief Populates a `routine_decision_t` structure based on environment variables.
 *
 * This function reads the `COLLECTIVE_TYPE` environment variable to determine
 * the collective type. Then, it reads the corresponding `ALGORITHM` environment
 * variable to determine the algorithm for the selected collective.
 *
 * @param test_routine Pointer to a `routine_decision_t` structure to populate.
 * @return `0` on success, `-1` on error.
 */
int get_routine(routine_decision_t *test_routine, int algorithm);


/**
 * @brief Parses command-line arguments and extracts parameters.
 *
 * @param argc Number of arguments.
 * @param argv Argument vector.
 * @param[out] array_size Size of the array.
 * @param[out] iter Number of iterations.
 * @param[out] type_string Data type as a string.
 * @param[out] alg_number Algorithm number.
 * @param[out] outputdir Output directory path.
 * @return 0 on success, -1 on error.
 */
int get_command_line_arguments(int argc, char** argv, size_t *array_size, int* iter,
                               const char **type_string, int *alg_number,
                               const char ** outputdir);


/**
 * @brief Retrieves the MPI datatype and size based on a string identifier utilizing `type_map`.
 *
 * @param type_string String representation of the data type.
 * @param[out] dtype MPI datatype corresponding to the string.
 * @param[out] type_size Size of the datatype in bytes.
 * @return 0 on success, -1 if the data type is invalid.
 */
int get_data_type(const char *type_string, MPI_Datatype *dtype, size_t *type_size);

//-----------------------------------------------------------------------------------------------
//                                   GROUND TRUTH CHECK FUNCTIONS
//-----------------------------------------------------------------------------------------------

/**
 * @brief Performs a ground-truth check for the result of an MPI Allreduce operation.
 *
 * @return int Returns 0 on success, -1 if there is a mismatch or an error in type handling.
 */
int allreduce_gt_check(ALLREDUCE_ARGS, void *recvbuf_gt);

/**
 * @brief Performs a ground-truth check for the result of an MPI Allgather operation.
 *
 * @return int Returns 0 on success, -1 if there is a mismatch or an error in type handling.
 */
int allgather_gt_check(ALLGATHER_ARGS, void *recvbuf_gt);


/**
 * @brief Performs a ground-truth check for the result of a Reduce Scatter operation.
 *
 * @return int Returns 0 on success, -1 if there is a mismatch or an error in type handling.
 */
int reduce_scatter_gt_check(REDUCE_SCATTER_ARGS, void *recvbuf_gt);

//-----------------------------------------------------------------------------------------------
//                                  I/O FUNCTIONS
//-----------------------------------------------------------------------------------------------

/**
 * @brief Writes the timing results to a specified output file.
 *
 * This function writes the highest timing values and the timings for each rank 
 * to a file in CSV format. The header includes "highest" followed by each rank 
 * from rank0 to rankN (where N is the number of ranks).
 *
 * @param fullpath The full path to the output file.
 * @param highest An array containing the highest timing values for each iteration.
 * @param all_times A 2D array flattened into 1D containing timing values for all ranks 
 *                  across all iterations.
 * @param iter The number of iterations.
 *
 * @return int Returns 0 on success, or -1 if an error occurs while opening the file.
 *
 * @note Time is saved in ns (i.e. 10^-9 s).
 */
int write_output_to_file(const char *fullpath, double *highest, double *all_times, int iter);


/**
 * @brief Checks if a file does not exists.
 *
 * @param filename The name of the file to check.
 * @return int Returns 1 if the file does not exists, 0 otherwise.
 */
int file_not_exists(const char* filename);


/**
 * @brief Writes MPI rank and processor name allocations to a specified file using MPI I/O.
 *
 * This function collects the MPI rank and processor name for each process and writes
 * them to a file in CSV format using parallel I/O. The file will have the format:
 *
 * rank,allocation
 * 0,processor_name_0
 * 1,processor_name_1
 * ...
 *
 * Each rank calculates its unique offset using a fixed entry size, based on MPI_MAX_PROCESSOR_NAME,
 * ensuring non-overlapping writes without requiring data size communication or gathering.
 *
 * This implementation uses `MPI_File_write_at` for concurrent, safe file access and a barrier
 * to synchronize ranks after writing the header.
 *
 * @param filename The name of the file to which the allocations will be written.
 * @param comm The MPI communicator.
 *
 * @return int Returns MPI_SUCCESS on success, MPI_ERR otherwise.
 */
int write_allocations_to_file(const char* filename, MPI_Comm comm);


//-----------------------------------------------------------------------------------------------
//                             GENERAL UTILITY FUNCTIONS
//-----------------------------------------------------------------------------------------------

/**
 * @brief Generates a random sbuf based on the specified type, size and collective.
 *
 * @param sbuf Pointer to the sbuf to fill with random values.
 * @param type_string Data type as a string.
 * @param array_size Number of elements in the array.
 * @param comm MPI communicator.
 * @param test_routine Routine decision structure.
 *
 * @return 0 on success, -1 if the data type is unsupported.
 */
int rand_sbuf_generator(void *sbuf, const char *type_string, size_t array_size,
                         MPI_Comm comm, routine_decision_t test_routine);


/**
 * @brief Concatenates a directory path and a filename into a full file path.
 *
 * @param dir_path Directory path.
 * @param filename Filename to append.
 * @param fullpath Buffer where the concatenated path is stored.
 * @return 0 on success, -1 on error.
 */
int concatenate_path(const char *dirpath, const char *filename, char *fullpath);


/**
 * @brief Compares two buffers with an epsilon tolerance for float or double datatypes.
 *
 * @param buf_1 First buffer.
 * @param buf_2 Second buffer.
 * @param count Size of the buffers in number of elements.
 * @param dtype MPI_Datatype of the recvbuf.
 * @param comm_sz Communication size to scale the epsilon.
 * @return 0 if buffers are equal within tolerance, -1 otherwise.
 */
int are_equal_eps(const void *buf_1, const void *buf_2, size_t count,
                  MPI_Datatype dtype, int comm_sz);

#endif
