#ifndef TEST_TOOLS_H
#define TEST_TOOLS_H

#include <mpi.h>

#include "libswing.h"

#define TEST_MAX_PATH_LENGTH 512
#define TEST_BASE_EPSILON_FLOAT 1e-6    // Base epsilon for float
#define TEST_BASE_EPSILON_DOUBLE 1e-15  // Base epsilon for double


//-----------------------------------------------------------------------------------------------
//                        ENUM FOR COLLECTIVE SELECTION
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


//-----------------------------------------------------------------------------------------------
//                         ALLOCATOR FUNCTIONS
//-----------------------------------------------------------------------------------------------

#define ALLOCATOR_ARGS    void **sbuf, void **rbuf, void **rbuf_gt, size_t count,\
                          size_t type_size, MPI_Comm comm
/**
* @typedef allocator_func_ptr
*
* A function pointer type for custom memory allocation functions.
*/
typedef int (*allocator_func_ptr)(ALLOCATOR_ARGS);

int allreduce_allocator(ALLOCATOR_ARGS);
int allgather_allocator(ALLOCATOR_ARGS);
int reduce_scatter_allocator(ALLOCATOR_ARGS);

//-----------------------------------------------------------------------------------------------
//                               FUNCTION POINTER AND WRAPPER
//                       (for specific collective function and gt_check)
//-----------------------------------------------------------------------------------------------
typedef int (*allreduce_func_ptr)(ALLREDUCE_ARGS);
typedef int (*allgather_func_ptr)(ALLGATHER_ARGS);
typedef int (*reduce_scatter_func_ptr)(REDUCE_SCATTER_ARGS);

static inline int allreduce_wrapper(ALLREDUCE_ARGS){
    return MPI_Allreduce(sbuf, rbuf, (int)count, dtype, op, comm);
}
static inline int allgather_wrapper(ALLGATHER_ARGS){
    return MPI_Allgather(sbuf, (int)scount, sdtype, rbuf, (int)rcount, rdtype, comm);
}

typedef int (*reduce_scatter_gt_func_ptr)(REDUCE_SCATTER_ARGS, void *rbuf_gt);
typedef int (*allreduce_gt_func_ptr)(ALLREDUCE_ARGS, void *rbuf_gt);
typedef int (*allgather_gt_func_ptr)(ALLGATHER_ARGS, void *rbuf_gt);

//-----------------------------------------------------------------------------------------------
//                                TEST ROUTINE STRUCTURE
//-----------------------------------------------------------------------------------------------

/**
 * @struct test_routine_t
 * @brief Structure to hold collective type and function pointers
 * for collective specific allocator, custom collective and 
 * ground truth functions pointers.
 *
 * @var collective Specifies the type of collective operation.
 * @var allocator Pointer to the memory allocator function.
 * @var function Union of function pointers for allreduce, allgather and reduce scatter.
 * @var gt_check Union of function pointers for ground truth checking functions.
 */
typedef struct {
  coll_t collective; /**< Specifies the type of collective operation. */

  allocator_func_ptr allocator; /**< Pointer to the memory allocator function. */

  /** Union of function pointers for custom collective functions. */
  union {
    allreduce_func_ptr allreduce;
    allgather_func_ptr allgather;
    reduce_scatter_func_ptr reduce_scatter;
  } function;

  /** Union of function pointers for ground truth checking functions. */
  union {
    allreduce_gt_func_ptr allreduce;
    allgather_gt_func_ptr allgather;
    reduce_scatter_gt_func_ptr reduce_scatter;
  } gt_check;
} test_routine_t;


//-----------------------------------------------------------------------------------------------
//                                MAIN BENCHMARK LOOP FUNCTIONS
//-----------------------------------------------------------------------------------------------


int test_loop(test_routine_t test_routine, void *sbuf, void *rbuf, size_t count,
              MPI_Datatype dtype, MPI_Comm comm, int iter, double *times);
/**
 * @brief This fucntion benchmarks the allreduce operation using the selected algorithm.
 */
void allreduce_test_loop(ALLREDUCE_ARGS, int iter, double *times, test_routine_t test_routine);

/**
 * @brief This fucntion benchmarks the allgather operation using the selected algorithm.
 */
void allgather_test_loop(ALLGATHER_ARGS, int iter, double *times, test_routine_t test_routine);

/**
 * @brief This fucntion benchmarks the reduce scatter operation using the selected algorithm.
 */
void reduce_scatter_test_loop(REDUCE_SCATTER_ARGS, int iter, double *times, test_routine_t test_routine);

//-----------------------------------------------------------------------------------------------
//                                   GROUND TRUTH CHECK FUNCTIONS
//-----------------------------------------------------------------------------------------------

int ground_truth_check(test_routine_t test_routine, void *sbuf, void *rbuf, void *rbuf_gt,
                       size_t count, MPI_Datatype dtype, MPI_Comm comm);
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
//                     SELECT ALGORITHM AND COMMAND LINE PARSING FUNCTIONS
//-----------------------------------------------------------------------------------------------

/**
 * @brief Populates a `test_routine_t` structure based on environment variables
 * and command-line arguments.
 *
 * This function reads the `COLLECTIVE_TYPE` environment variable and reads the 
 * `algorithm` command-line argument to popuplate the `test_routine_t` structure
 * with the appropriate collective type and function pointers.
 *
 * @param test_routine Pointer to a `test_routine_t` structure to populate.
 * @param algorithm The algorithm name as a string.
 *
 * @return `0` on success, `-1` on error.
 */
int get_routine(test_routine_t *test_routine, const char *algorithm);


/**
 * @brief Parses command-line arguments and extracts parameters.
 *
 * @param argc Number of arguments.
 * @param argv Argument vector.
 * @param[out] array_count Size of the array.
 * @param[out] iter Number of iterations.
 * @param[out] algprithm Algorithm name.
 * @param[out] type_string Data type as a string.
 * @return 0 on success, -1 on error.
 */
int get_command_line_arguments(int argc, char** argv, size_t *array_count,
                               int* iter, const char **algorithm, const
                               char **type_string);


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
 * @param dtype Datatype of the sendbuffer (MPI Datatype).
 * @param array_size Number of elements in the array.
 * @param comm MPI communicator.
 * @param test_routine Routine decision structure.
 *
 * @return 0 on success, -1 if the data type is unsupported.
 */
int rand_sbuf_generator(void *sbuf, MPI_Datatype dtype, size_t array_size,
                         MPI_Comm comm, test_routine_t test_routine);


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

//-----------------------------------------------------------------------------------------------
//                          DEBUGGING FUNCTIONS
//-----------------------------------------------------------------------------------------------
#ifdef DEBUG
/**
 * @brief Generates the send buffer with a sequence of powers of 10^rank.
 *
 * @param sbuf Pointer to the sbuf to fill with random values.
 * @param dtype Datatype of the sendbuffer (MPI Datatype).
 * @param count Number of elements in the array.
 * @param comm MPI communicator.
 * @param test_routine Routine decision structure.
 *
 * @return 0 on success, -1 if the data type is unsupported.
 */
int debug_sbuf_generator(void *sbuf, MPI_Datatype dtype, size_t count,
                    MPI_Comm comm, test_routine_t test_routine);

/**
 * @brief Prints the contents of two buffers for debugging purposes.
 *
 * @param rbuf The buffer to print.
 * @param rbuf_gt The ground-truth buffer to print.
 * @param count The number of elements in the buffer.
 * @param dtype The MPI datatype of the buffer.
 * @param comm The MPI communicator.
 */
void debug_print_buffers(void *rbuf, void *rbuf_gt, size_t count, MPI_Datatype dtype, MPI_Comm comm);
#endif

#endif // TEST_TOOLS_H

