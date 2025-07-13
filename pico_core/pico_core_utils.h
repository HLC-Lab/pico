/*
 * Copyright (c) 2025 Saverio Pasqualoni
 * Licensed under the MIT License
 */

#ifndef PICO_CORE_UTILS_H
#define PICO_CORE_UTILS_H

#include <mpi.h>
#include <stdio.h>
#ifdef CUDA_AWARE
#include <cuda_runtime.h>
#endif

#include "libbine.h"

#if defined(__GNUC__) || defined(__clang__)
  #define PICO_CORE_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
  #define PICO_CORE_UNLIKELY(x) (x)
#endif // defined(__GNUC__) || defined(__clang__)

// Used to print algorithm and collective when in debug mode
#ifndef DEBUG
  #define PICO_CORE_DEBUG_PRINT_STR(name)
  #define PICO_CORE_DEBUG_PRINT_BUFFERS(result, expected, count, dtype, comm, use_barrier) do {} while(0)
#else
  #define PICO_CORE_DEBUG_PRINT_STR(name)           \
    do{                                         \
      int my_r;                                 \
      MPI_Comm_rank(MPI_COMM_WORLD, &my_r);     \
      if(my_r == 0){ printf("%s\n", name); }    \
    } while(0)

  #define PICO_CORE_DEBUG_PRINT_BUFFERS(result, expected, count, dtype, comm, use_barrier)      \
    do {                                                                                    \
    print_buffers(NULL, (result), (expected), 0, (count), (dtype), (comm), (use_barrier));  \
    } while(0)
#endif // DEBUG

#define CHECK_STR(var, name, ret)              \
  if(strcmp(var, name) == 0) {                 \
    PICO_CORE_DEBUG_PRINT_STR(name);               \
    return ret;                                \
  }

#define PICO_CORE_MAX_PATH_LENGTH 512
#define PICO_CORE_BASE_EPSILON_FLOAT 1e-6    // Base epsilon for float
#define PICO_CORE_BASE_EPSILON_DOUBLE 1e-15  // Base epsilon for double

#define PICO_CORE_MAX_ALLOC_NAME_LEN ( MPI_MAX_PROCESSOR_NAME + 64 )
#define PICO_CORE_HEADER_LUMI "MPI_Rank,allocation,xname\n"
#define PICO_CORE_HEADER_DEFAULT "MPI_Rank,allocation\n"

extern size_t bine_allreduce_segsize;

//-----------------------------------------------------------------------------------------------
//                        ENUM FOR TEST DESCRIPTION
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
  ALLTOALL,
  BCAST,
  GATHER,
  REDUCE,
  REDUCE_SCATTER,
  SCATTER,
  COLL_UNKNOWN
}coll_t;

/**
 * @enum output_level_t
 *
 * @brief Defines the output level of data saving for pico_coremark's results.
 * */
typedef enum{
  ALL = 0,
  STATISTICS,
  SUMMARIZED
}output_level_t;


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
int alltoall_allocator(ALLOCATOR_ARGS);
int bcast_allocator(ALLOCATOR_ARGS);
int gather_allocator(ALLOCATOR_ARGS);
int reduce_allocator(ALLOCATOR_ARGS);
int reduce_scatter_allocator(ALLOCATOR_ARGS);
int scatter_allocator(ALLOCATOR_ARGS);

//-----------------------------------------------------------------------------------------------
//                               FUNCTION POINTER AND WRAPPER
//                       (for specific collective function and gt_check)
//-----------------------------------------------------------------------------------------------
typedef int (*allreduce_func_ptr)(ALLREDUCE_ARGS);
typedef int (*allgather_func_ptr)(ALLGATHER_ARGS);
typedef int (*alltoall_func_ptr)(ALLTOALL_ARGS);
typedef int (*bcast_func_ptr)(BCAST_ARGS);
typedef int (*gather_func_ptr)(GATHER_ARGS);
typedef int (*reduce_func_ptr)(REDUCE_ARGS);
typedef int (*reduce_scatter_func_ptr)(REDUCE_SCATTER_ARGS);
typedef int (*scatter_func_ptr)(SCATTER_ARGS);

static inline int allreduce_wrapper(ALLREDUCE_ARGS){
  return MPI_Allreduce(sbuf, rbuf, (int)count, dtype, op, comm);
}
static inline int allgather_wrapper(ALLGATHER_ARGS){
  return MPI_Allgather(sbuf, (int)scount, sdtype, rbuf, (int)rcount, rdtype, comm);
}
static inline int alltoall_wrapper(ALLTOALL_ARGS){
  return MPI_Alltoall(sbuf, (int)scount, sdtype, rbuf, (int)rcount, rdtype, comm);
}
static inline int bcast_wrapper(BCAST_ARGS){
  return MPI_Bcast(buf, (int)count, dtype, root, comm);
}
static inline int gather_wrapper(GATHER_ARGS){
  return MPI_Gather(sbuf, (int)scount, sdtype, rbuf, (int)rcount, rdtype, root, comm);
}
static inline int reduce_wrapper(REDUCE_ARGS){
  return MPI_Reduce(sbuf, rbuf, (int)count, dtype, op, root, comm);
}
static inline int scatter_wrapper(SCATTER_ARGS){
  return MPI_Scatter(sbuf, (int)scount, sdtype, rbuf, (int)rcount, rdtype, root, comm);
}


//-----------------------------------------------------------------------------------------------
//                                TEST ROUTINE STRUCTURE
//-----------------------------------------------------------------------------------------------

/**
 * @struct test_routine_t
 * @brief Structure to hold collective type and function pointers
 * for collective specific allocator, custom collective and 
 * ground truth functions pointers.
 * It also holds the output level, the output folder path and the data folder path.
 *
 * @var collective Specifies the type of collective operation.
 * @var allocator Pointer to the memory allocator function.
 * @var function Union of function pointers for allreduce, allgather and reduce scatter.
 */
typedef struct {
  coll_t collective; /**< Specifies the type of collective operation. */
  allocator_func_ptr allocator; /**< Pointer to the memory allocator function. */
#ifdef CUDA_AWARE
  allocator_func_ptr allocator_cuda; /**< Pointer to the CUDA memory allocator function. */
#endif // CUDA_AWARE
  size_t segsize; /**< Size of the segment for segmented collectives. */

  /** Union of function pointers for custom collective functions. */
  union {
    allreduce_func_ptr allreduce;
    allgather_func_ptr allgather;
    alltoall_func_ptr alltoall;
    bcast_func_ptr bcast;
    gather_func_ptr gather;
    reduce_func_ptr reduce;
    reduce_scatter_func_ptr reduce_scatter;
    scatter_func_ptr scatter;
  } function;

  output_level_t output_level;    /**< Specifies the output level for data saving. */
  char output_data_file[PICO_CORE_MAX_PATH_LENGTH];   /**< Path to the output directory. */
  char alloc_file[PICO_CORE_MAX_PATH_LENGTH];         /**< Path to the data directory. */
} test_routine_t;


// ----------------------------------------------------------------------------------------------
//                                CUDA FUNCTIONS
// ----------------------------------------------------------------------------------------------

#ifdef CUDA_AWARE

#define PICO_CORE_CUDA_CHECK(cmd, err) do {                 \
  err = cmd;                                            \
  if( err != cudaSuccess ) {                            \
    fprintf(stderr, "Failed: Cuda error %s:%d '%s'\n",  \
        __FILE__, __LINE__, cudaGetErrorString(err));   \
    return -1;                                          \
  }                                                     \
} while(0)

int allreduce_allocator_cuda(ALLOCATOR_ARGS);
int allgather_allocator_cuda(ALLOCATOR_ARGS);
int alltoall_allocator_cuda(ALLOCATOR_ARGS);
int bcast_allocator_cuda(ALLOCATOR_ARGS);
int gather_allocator_cuda(ALLOCATOR_ARGS);
int reduce_allocator_cuda(ALLOCATOR_ARGS);
int reduce_scatter_allocator_cuda(ALLOCATOR_ARGS);
int scatter_allocator_cuda(ALLOCATOR_ARGS);

int coll_memcpy_host_to_device(void** d_buf, void** buf, size_t count, size_t type_size, coll_t coll) {
int coll_memcpy_device_to_host(void** d_buf, void** buf, size_t count, size_t type_size, coll_t coll) {

#endif


//-----------------------------------------------------------------------------------------------
//                                MAIN PICO_COREMARK LOOP FUNCTIONS
//-----------------------------------------------------------------------------------------------


 /**
 * @brief Test loop interface that select the appropriate collective operation
 * test loop based on the collective type and algorithm specified in the test_routine.
 *
 * @return MPI_SUCCESS on success, an MPI_ERR code on error.
 */
int test_loop(test_routine_t test_routine, void *sbuf, void *rbuf, size_t count,
              MPI_Datatype dtype, MPI_Comm comm, int iter, double *times);

/**
 * @macro TEST_LOOP
 * @brief Macro to generate a test loop for a given collective operation.
 *
 * @param OP_NAME Name of the operation.
 * @param ARGS Arguments for the operation.
 * @param COLLECTIVE Collective operation to perform.
 */
#define DEFINE_TEST_LOOP(OP_NAME, ARGS, COLLECTIVE)                  \
static inline int OP_NAME##_test_loop(ARGS, int iter, double *times, \
                                   test_routine_t test_routine) {    \
  int ret = MPI_SUCCESS;                                             \
  double start_time, end_time;                                       \
  MPI_Barrier(comm);                                                 \
  for(int i = 0; i < iter; i++) {                                    \
    start_time = MPI_Wtime();                                        \
    ret = test_routine.function.COLLECTIVE;                          \
    end_time = MPI_Wtime();                                          \
    times[i] = end_time - start_time;                                \
    if(PICO_CORE_UNLIKELY(ret != MPI_SUCCESS)) {                         \
      fprintf(stderr, "Error: " #OP_NAME " failed. Aborting...");    \
      return ret;                                                    \
    }                                                                \
    MPI_Barrier(comm);                                               \
  }                                                                  \
  return ret;                                                        \
}

DEFINE_TEST_LOOP(allreduce, ALLREDUCE_ARGS, allreduce(sbuf, rbuf, count, dtype, MPI_SUM, comm))
DEFINE_TEST_LOOP(allgather, ALLGATHER_ARGS, allgather(sbuf, scount, sdtype, rbuf, rcount, rdtype, comm))
DEFINE_TEST_LOOP(alltoall, ALLTOALL_ARGS, alltoall(sbuf, scount, sdtype, rbuf, rcount, rdtype, comm))
DEFINE_TEST_LOOP(bcast, BCAST_ARGS, bcast(buf, count, dtype, 0, comm))
DEFINE_TEST_LOOP(gather, GATHER_ARGS, gather(sbuf, scount, sdtype, rbuf, rcount, rdtype, 0, comm))
DEFINE_TEST_LOOP(reduce, REDUCE_ARGS, reduce(sbuf, rbuf, count, dtype, MPI_SUM, 0, comm))
DEFINE_TEST_LOOP(reduce_scatter, REDUCE_SCATTER_ARGS, reduce_scatter(sbuf, rbuf, rcounts, dtype, MPI_SUM, comm))
DEFINE_TEST_LOOP(scatter, SCATTER_ARGS, scatter(sbuf, scount, sdtype, rbuf, rcount, rdtype, 0, comm))


//-----------------------------------------------------------------------------------------------
//                                   GROUND TRUTH CHECK FUNCTIONS
//-----------------------------------------------------------------------------------------------

/**
 * @brief Compares two buffers with an epsilon tolerance for float or double datatypes.
 *
 * @param buf_1 First buffer.
 * @param buf_2 Second buffer.
 * @param count Size of the buffers in number of elements.
 * @param dtype MPI_Datatype of the recvbuf.
 * @param comm Communicator.
 * @return 0 if buffers are equal within tolerance, -1 otherwise.
 */
int are_equal_eps(const void *buf_1, const void *buf_2, size_t count,
                  MPI_Datatype dtype, MPI_Comm comm);


/**
 * @macro GT_CHECK_BUFFER
 * @brief Macro to check the result of an MPI operation against the ground truth.
 *
 * It is used inside the ground truth check functions to compare the result of an MPI operation
 * against the ground truth. It checks if the result is equal to the expected value within an
 * epsilon tolerance for float and double datatypes, and uses `memcmp` for other datatypes.
 */
#define GT_CHECK_BUFFER(result, expected, count, dtype, comm)                 \
  do {                                                                        \
    if(dtype != MPI_DOUBLE && dtype != MPI_FLOAT) {                          \
      if(memcmp((result), (expected), (count) * type_size) != 0) {           \
        PICO_CORE_DEBUG_PRINT_BUFFERS((result), (expected), (count), (dtype), (comm), (1));  \
        fprintf(stderr, "Error: results are not valid. Aborting...");       \
        ret = -1;                                                             \
      }                                                                       \
    } else {                                                                  \
      if(are_equal_eps((result), (expected), (count), dtype, comm) == -1) {  \
        PICO_CORE_DEBUG_PRINT_BUFFERS((result), (expected), (count), (dtype), (comm), (1));  \
        fprintf(stderr, "Error: results are not valid. Aborting...");       \
        ret = -1;                                                             \
      }                                                                       \
    }                                                                         \
  } while(0)


/**
 * @brief Interface for ground-truth check functions.
 * This function selects the appropriate ground-truth check function based on the
 * collective type specified in the test_routine.
 *
 * @return 0 on success, an -1 on error.
 */
int ground_truth_check(test_routine_t test_routine, void *sbuf, void *rbuf, void *rbuf_gt,
                       size_t count, MPI_Datatype dtype, MPI_Comm comm);


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
 * @brief Retrieves the data saving options based on environment variables.
 *
 * This function reads the `OUTPUT_LEVEL` environment variable to determine
 * the data saving options for the pico_coremark results as well as the
 * `OUTPUT_DIR` and `DATA_DIR` environment variables to set the output
 * directory and data directory respectively.
 *
 * @param test_routine Pointer to a `test_routine_t` structure to populate.
 *
 * @return `0` on success, `-1` on error.
 */
int get_data_saving_options(test_routine_t *test_routine, size_t count,
                            const char *algorithm, const char *type_string);

/**
 * @brief Retrieves the MPI datatype and size based on a string identifier utilizing `type_map`.
 *
 * @param type_string String representation of the data type.
 * @param[out] dtype MPI datatype corresponding to the string.
 * @param[out] type_size Size of the datatype in bytes.
 * @return 0 on success, -1 if the data type is invalid.
 */
int get_data_type(const char *type_string, MPI_Datatype *dtype, size_t *type_size);


/**
 * @brief Splits the MPI communicator into inter and intra communicators.
 *
 * This function splits the MPI_COMM_WORLD communicator into two communicators:
 * - `intra_comm`: for communication within a node.
 * - `inter_comm`: for communication between nodes.
 *
 * @param[out] inter_comm Pointer to the inter communicator.
 * @param[out] intra_comm Pointer to the intra communicator.
 *
 * @return MPI_SUCCESS on success, an MPI_ERR code on error.
 *
 * @note The number of tasks per node is determined by the `CURRENT_TASKS_PER_NODE`
 * environment variable. The function checks if the number of ranks is divisible by this value.
 */
int split_communicator(MPI_Comm *inter_comm, MPI_Comm *intra_comm);

//-----------------------------------------------------------------------------------------------
//                                  I/O FUNCTIONS
//-----------------------------------------------------------------------------------------------

/**
 * @brief Writes the timing results to a specified output file in CSV format.
 *
 * The ammount of data to save is determined by the `output_level` parameter.
 * If `output_level` is set to "all", the timing results for all ranks across all
 * iterations will be saved. If `output_level` is set to "summarized", only the
 * highest timing value for each iteration will be saved.
 *
 * @param test_routine The test routine structure containing the output level and file path
 * @param highest An array containing the highest timing values for each iteration.
 * @param all_times A 2D array flattened into 1D containing timing values for all ranks 
 *                  across all iterations.
 * @param iter The number of iterations.
 *
 * @return int Returns 0 on success, or -1 if an error occurs.
 *
 * @note Time is saved in ns (i.e. 10^-9 s).
 */
int write_output_to_file(test_routine_t test_routine, double *highest, double *all_times, int iter);

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
 * @brief Prints the contents of two buffers for debugging purposes.
 *
 * @param rbuf The buffer to print.
 * @param rbuf_gt The ground-truth buffer to print.
 * @param count The number of elements in the buffer.
 * @param dtype The MPI datatype of the buffer.
 * @param comm The MPI communicator.
 * @param use_barrier Flag to indicate if a barrier should be used.
 */
void print_buffers(const void *sbuf, const void *rbuf, const void *rbuf_gt,
                   size_t sbuf_count, size_t rbuf_count, MPI_Datatype dtype,
                   MPI_Comm comm, int use_barrier);

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

#endif // DEBUG

#endif // PICO_CORE_TOOLS_H
