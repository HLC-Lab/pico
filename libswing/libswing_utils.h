#ifndef LIBSWING_UTILS_H
#define LIBSWING_UTILS_H

#define SWING_MAX_STEPS 20

#include <mpi.h>
#include <string.h>
#include <stdlib.h>

static int rhos[SWING_MAX_STEPS] = {1, -1, 3, -5, 11, -21, 43, -85, 171, -341,
          683, -1365, 2731, -5461, 10923, -21845, 43691, -87381, 174763, -349525};


/**
 * This macro gives a generic way to compute the well distributed block counts
 * when the count and number of blocks are fixed.
 * Macro returns "early-block" count, "late-block" count, and "split-index"
 * which is the block at which we switch from "early-block" count to
 * the "late-block" count.
 * count = split_index * early_block_count +
 *         (block_count - split_index) * late_block_count
 * We do not perform ANY error checks - make sure that the input values
 * make sense (eg. count > num_blocks).
 */
#define COLL_BASE_COMPUTE_BLOCKCOUNT( COUNT, NUM_BLOCKS, SPLIT_INDEX,       \
                                       EARLY_BLOCK_COUNT, LATE_BLOCK_COUNT ) \
    EARLY_BLOCK_COUNT = LATE_BLOCK_COUNT = COUNT / NUM_BLOCKS;               \
    SPLIT_INDEX = COUNT % NUM_BLOCKS;                                        \
    if (0 != SPLIT_INDEX) {                                                  \
        EARLY_BLOCK_COUNT = EARLY_BLOCK_COUNT + 1;                           \
    }                                                                        \


/**
 * @brief Computes the destination rank for a given process in a swing
 * algorithm step.
 *
 * This function calculates the rank to which a process will communicate
 * based on the swing algorithm, ensuring the result is within the valid
 * range of ranks.
 *
 * @param rank The rank of the current process.
 * @param step The current step in the swing algorithm.
 * @param comm_sz The total number of processes in the communicator.
 * @return The destination rank after applying the swing algorithm, a
 *         value in [0, comm_sz - 1].
 */
static inline int pi(int rank, int step, int comm_sz) {
  int dest;

  if ((rank & 1) == 0) dest = (rank + rhos[step]) % comm_sz;  // Even rank
  else dest = (rank - rhos[step]) % comm_sz;                  // Odd rank

  if (dest < 0) dest += comm_sz;                              // Adjust for negative ranks

  return dest;
}

/**
 * @brief Copies data from an input buffer to an output buffer.
 *
 * This function validates the input parameters and performs a memory copy
 * of `count` elements from the input buffer to the output buffer, using
 * the size of the specified MPI datatype.
 *
 * @param input_buffer Pointer to the source buffer.
 * @param output_buffer Pointer to the destination buffer.
 * @param count Number of elements to copy.
 * @param datatype The MPI datatype of each element.
 * @return MPI_SUCCESS on success, or MPI_ERR_UNKNOWN.
 */
static inline int copy_buffer(const void *input_buffer, void *output_buffer,
                              size_t count, const MPI_Datatype datatype) {
  if (input_buffer == NULL || output_buffer == NULL || count <= 0) {
    return MPI_ERR_UNKNOWN;
  }

  int datatype_size;
  MPI_Type_size(datatype, &datatype_size);                // Get the size of the MPI datatype

  size_t total_size = count * (size_t)datatype_size;

  memcpy(output_buffer, input_buffer, total_size);        // Perform the memory copy

  return MPI_SUCCESS;
}


/**
 * @brief Copies data from an input buffer to an output buffer with
 * different datatypes.
 *
 * Similar to copy_buffer but handles the case with different dtypes.
 *
 * @param input_buffer Pointer to the source buffer.
 * @param scount Number of elements to copy from source buffer.
 * @param sdtype The MPI datatype of elements in input_buffer.
 * @param output_buffer Pointer to the destination buffer.
 * @param rcount Number of elements to copy to dest buffer.
 * @param rdtype The MPI datatype of elements in destination buffer.
 * @return MPI_SUCCESS on success, or MPI_ERR.
 */
static inline int copy_buffer_different_dt (const void *input_buffer, size_t scount,
                                            const MPI_Datatype sdtype, void *output_buffer,
                                            size_t rcount, const MPI_Datatype rdtype) {
  if (input_buffer == NULL || output_buffer == NULL || scount <= 0 || rcount <= 0) {
    return MPI_ERR_UNKNOWN;
  }

  int sdtype_size;
  MPI_Type_size(sdtype, &sdtype_size);
  int rdtype_size;
  MPI_Type_size(rdtype, &rdtype_size);

  size_t s_size = (size_t) sdtype_size * scount;
  size_t r_size = (size_t) rdtype_size * rcount;

  if (r_size < s_size) {
    memcpy(output_buffer, input_buffer, r_size); // Copy as much as possible
    return MPI_ERR_TRUNCATE;      // Indicate truncation
  }

  memcpy(output_buffer, input_buffer, s_size);        // Perform the memory copy

  return MPI_SUCCESS;
}


/**
 * @brief Computes the memory span for `count` repetitions of the given
 * MPI_datatype `dtype`.
 *
 * This function calculates the total memory required for `count` repetitions
 * of `dtype`, including the gap at the beginning (true lower bound) and
 * excluding padding at the end.
 *
 * @param dtype The MPI datatype.
 * @param count Number of repetitions of the datatype.
 * @param gap Pointer to store the gap (true lower bound) at the beginning.
 * @return The total memory span required for `count` repetitions of the datatype.
 */
static inline ptrdiff_t datatype_span(MPI_Datatype dtype, size_t count, ptrdiff_t *gap) {
  if (count == 0) {
    *gap = 0;
    return 0;                                 // No memory span required for zero repetitions
  }

  MPI_Aint lb, extent;
  MPI_Aint true_lb, true_extent;
  
  // Get extend and true extent (true extent does not include padding)
  MPI_Type_get_extent(dtype, &lb, &extent);
  MPI_Type_get_true_extent(dtype, &true_lb, &true_extent);

  *gap = true_lb;                             // Store the true lower bound

  return true_extent + extent * (count - 1);  // Calculate the total memory span
}


/**
 * @brief Returns log_2(value). Value must be a positive integer.
 *
 * @param value The **POSITIVE** integer value to return its log_2.
 *
 * @returns The log_2 of value or -1 for negative value.
 */
static inline int log_2(int value) {
  if (1 > value) {
    return -1;
  }
  return sizeof(int)*8 - 1 - __builtin_clz(value);
}


/**
 * @brief Returns next power-of-two greater of the given value.
 *
 * @param value The **POSITIVE** integer value to return power of 2.
 *
 * @returns The next power of two or -1 for negative values.
 */
static inline int next_poweroftwo(int value)
{
  if (0 > value) {
    return -1;
  }

  if (0 == value) {
    return 1;
  }

  return 1 << (8 * sizeof(int) - __builtin_clz(value));
}


/**
 * Calculates the highest bit in an integer
 *
 * @param value The integer value to examine
 * @param start Position to start looking
 *
 * @returns pos Position of highest-set integer or -1 if none are set.
 *
 * Look at the integer "value" starting at position "start", and move
 * to the right.  Return the index of the highest bit that is set to
 * 1.
 *
 * WARNING: *NO* error checking is performed.  This is meant to be a
 * fast inline function.
 */
static inline int hibit(int value, int start)
{
  unsigned int mask;

  /* Only look at the part that the caller wanted looking at */
  mask = value & ((1 << start) - 1);

  if (0 == mask) {
    return -1;
  }

  start = (8 * sizeof(int) - 1) - __builtin_clz(mask);

  return start;
}


typedef struct {
  MPI_Request* reqs;    // Pointer to the array of requests
  int num_reqs;         // Current size of the array
} request_manager_t;

/**
 * Ensures the array of MPI_Request is properly initialized and large enough.
 * If the array is not initialized or is too small, it will be resized to fit the requested size.
 *
 * @param manager  Pointer to the mpi_request_manager_t structure managing the requests.
 * @param nreqs    The desired number of requests in the array.
 * @return         Pointer to the array of MPI_Request, or NULL if allocation failed.
 */
MPI_Request* alloc_reqs(request_manager_t* manager, int nreqs) {
  if (nreqs == 0) {
    return NULL;
  }

  // If the current array is too small, resize it
  if (manager->num_reqs < nreqs) {
    MPI_Request* new_reqs = realloc(manager->reqs, sizeof(MPI_Request) * nreqs);
    if (new_reqs == NULL) {
      // Allocation failed, reset the manager
      manager->reqs = NULL;
      manager->num_reqs = 0;
      return NULL;
    }

    // Update the pointer to the new array
    manager->reqs = new_reqs;

    // Initialize new entries to MPI_REQUEST_NULL
    for (int i = manager->num_reqs; i < nreqs; i++) {
      manager->reqs[i] = MPI_REQUEST_NULL;
    }

    // Update the current size of the array
    manager->num_reqs = nreqs;
  }

  return manager->reqs;
}

/**
 * Cleans up the request_manager_t structure by freeing the array of requests.
 *
 * @param manager  Pointer to the request_manager_t structure to clean up.
 */
void cleanup_reqs(request_manager_t* manager) {
  if (manager->reqs != NULL) {
    free(manager->reqs);
    manager->reqs = NULL;
  }
  manager->num_reqs = 0;
}

#endif

