#ifndef LIBSWING_UTILS_H
#define LIBSWING_UTILS_H

#define SWING_MAX_STEPS 20

#ifdef CUDA_AWARE
#include <cuda_runtime.h>
#endif

#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <limits.h>
#include <stdint.h>
#include <assert.h>
#include <stddef.h>

#ifdef DEBUG
#define SWING_DEBUG_PRINT(fmt, ...) \
  do { fprintf(stderr, fmt, ##__VA_ARGS__); } while (0)
#else
#define SWING_DEBUG_PRINT(fmt, ...) \
  do {} while (0)
#endif
#if defined(__GNUC__) || defined(__clang__)

#define SWING_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define SWING_UNLIKELY(x) (x)
#endif // defined(__GNUC__) || defined(__clang__)

#ifdef CUDA_AWARE
#define COPY_BUFF_DIFF_DT(...) copy_buffer_different_dt_cuda(__VA_ARGS__)
#else
#define COPY_BUFF_DIFF_DT(...) copy_buffer_different_dt(__VA_ARGS__)
#endif

static int rhos[SWING_MAX_STEPS] = {1, -1, 3, -5, 11, -21, 43, -85, 171, -341,
          683, -1365, 2731, -5461, 10923, -21845, 43691, -87381, 174763, -349525};

static int smallest_negabinary[SWING_MAX_STEPS] = {0, 0, -2, -2, -10, -10, -42, -42,
          -170, -170, -682, -682, -2730, -2730, -10922, -10922, -43690, -43690, -174762, -174762};
static int largest_negabinary[SWING_MAX_STEPS] = {0, 1, 1, 5, 5, 21, 21, 85, 85,
          341, 341, 1365, 1365, 5461, 5461, 21845, 21845, 87381, 87381, 349525};

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
    if(0 != SPLIT_INDEX) {                                                  \
        EARLY_BLOCK_COUNT = EARLY_BLOCK_COUNT + 1;                           \
    }                                                                        \

// ----------------------------------------------------------------------------------------------
//                                MACRO FOR CUDA FUNCTION CALLS
// ----------------------------------------------------------------------------------------------

#ifdef CUDA_AWARE

#define SWING_CUDA_CHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    fprintf(stderr, "Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


static inline int copy_buffer_different_dt_cuda(const void *input_buffer, size_t scount,
  const MPI_Datatype sdtype, void *output_buffer,
  size_t rcount, const MPI_Datatype rdtype) {
  if(SWING_UNLIKELY(input_buffer == NULL || output_buffer == NULL || scount <= 0 || rcount <= 0)) {
  return MPI_ERR_UNKNOWN;
  }

  int sdtype_size;
  MPI_Type_size(sdtype, &sdtype_size);
  int rdtype_size;
  MPI_Type_size(rdtype, &rdtype_size);

  size_t s_size = (size_t) sdtype_size * scount;
  size_t r_size = (size_t) rdtype_size * rcount;

  if(r_size < s_size) {
    SWING_CUDA_CHECK(cudaMemcpy(output_buffer, input_buffer, r_size, cudaMemcpyDeviceToDevice));
    return MPI_ERR_TRUNCATE;      // Indicate truncation
  }

  SWING_CUDA_CHECK(cudaMemcpy(output_buffer, input_buffer, s_size, cudaMemcpyDeviceToDevice));   // Perform the memory copy

  return MPI_SUCCESS;
}

#endif // CUDA_AWARE


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

  if((rank & 1) == 0) dest = (rank + rhos[step]) % comm_sz;  // Even rank
  else dest = (rank - rhos[step]) % comm_sz;                  // Odd rank

  if(dest < 0) dest += comm_sz;                              // Adjust for negative ranks

  return dest;
}



static inline void get_indexes_aux(int rank, int step, const int n_steps, const int adj_size, int *bitmap){
  if (step >= n_steps) return;

  int peer;
  
  for (int s = step; s < n_steps; s++){
    peer = pi(rank, s, adj_size);
    *(bitmap + peer) = 0x1;
    get_indexes_aux(peer, s + 1, n_steps, adj_size, bitmap);
  }
}


static inline void get_indexes(int rank, int step, const int n_steps, const int adj_size, int *bitmap){
  if (step >= n_steps) return;
  
  int peer = pi(rank, step, adj_size);
  *(bitmap + peer) = 0x1;
  get_indexes_aux(peer, step + 1, n_steps, adj_size, bitmap);
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
  if(SWING_UNLIKELY(input_buffer == NULL || output_buffer == NULL || count <= 0)) {
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
  if(SWING_UNLIKELY(input_buffer == NULL || output_buffer == NULL || scount <= 0 || rcount <= 0)) {
    return MPI_ERR_UNKNOWN;
  }

  int sdtype_size;
  MPI_Type_size(sdtype, &sdtype_size);
  int rdtype_size;
  MPI_Type_size(rdtype, &rdtype_size);

  size_t s_size = (size_t) sdtype_size * scount;
  size_t r_size = (size_t) rdtype_size * rcount;

  if(r_size < s_size) {
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
  if(count == 0) {
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
  if(SWING_UNLIKELY(1 > value)) {
    return -1;
  }
  return sizeof(int)*8 - 1 - __builtin_clz(value);
}


/**
 * @brief Returns if the given value is a power of two.
 */
static inline int is_power_of_two(int value) {
    return (value & (value - 1)) == 0;
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
  if(SWING_UNLIKELY(0 > value)) {
    return -1;
  }

  if(0 == value) {
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

  if(0 == mask) {
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
static inline MPI_Request* alloc_reqs(request_manager_t* manager, int nreqs) {
  if(nreqs == 0) {
    return NULL;
  }

  // If the current array is too small, resize it
  if(manager->num_reqs < nreqs) {
    MPI_Request* new_reqs = realloc(manager->reqs, sizeof(MPI_Request) * nreqs);
    if(new_reqs == NULL) {
      // Allocation failed, reset the manager
      manager->reqs = NULL;
      manager->num_reqs = 0;
      return NULL;
    }

    // Update the pointer to the new array
    manager->reqs = new_reqs;

    // Initialize new entries to MPI_REQUEST_NULL
    for(int i = manager->num_reqs; i < nreqs; i++) {
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
static inline void cleanup_reqs(request_manager_t* manager) {
  if(manager->reqs != NULL) {
    free(manager->reqs);
    manager->reqs = NULL;
  }
  manager->num_reqs = 0;
}

/*
 * sum_counts: Returns sum of counts [lo, hi]
 *                  lo, hi in {0, 1, ..., nprocs_pof2 - 1}
 */
static inline size_t sum_counts(const int counts[], ptrdiff_t *displs, int nprocs_rem, int lo, int hi)
{
    /* Adjust lo and hi for taking into account blocks of excluded processes */
    lo = (lo < nprocs_rem) ? lo * 2 : lo + nprocs_rem;
    hi = (hi < nprocs_rem) ? hi * 2 + 1 : hi + nprocs_rem;
    return displs[hi] + counts[hi] - displs[lo];
}

/*
 * mirror_perm: Returns mirror permutation of nbits low-order bits
 *                   of x [*].
 * [*] Warren Jr., Henry S. Hacker's Delight (2ed). 2013.
 *     Chapter 7. Rearranging Bits and Bytes.
 */
static inline unsigned int mirror_perm(unsigned int x, int nbits)
{
    x = (((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1));
    x = (((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2));
    x = (((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4));
    x = (((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8));
    x = ((x >> 16) | (x << 16));
    return x >> (sizeof(x) * CHAR_BIT - nbits);
}

/**
 * @brief Reorders blocks in a buffer according to a given permutation.
 *
 * @param buffer The buffer containing the blocks to reorder.
 * @param block_size The size of each block in bytes.
 * @param block_permutation The permutation of the blocks.
 * @param num_blocks The number of blocks in the buffer.
 *
 * @return MPI_SUCCESS on success, or an error code.
 */
static inline int reorder_blocks(void *buffer, size_t block_size,
                                  int *block_permutation, int num_blocks) {
  if(SWING_UNLIKELY(buffer == NULL || block_permutation == NULL || num_blocks <= 0)) {
    return MPI_ERR_ARG;
  }

  char *buf = (char *)buffer;
  void *temp = malloc(block_size);
  char *visited = (char *)calloc(num_blocks, sizeof(int));
  if(temp == NULL || visited == NULL) {
    return MPI_ERR_NO_MEM;
  }

  for(int i = 0; i < num_blocks; ++i) {
    // Skip if the block is already in its correct position or visited
    if(visited[i] == 1 || block_permutation[i] == i) {
      continue;
    }

    int current = i;
    // Save the current block to temp (start of the cycle)
    memcpy(temp, buf + current * block_size, block_size);


    // Follow the cycle and place each block in its final position
    while (visited[block_permutation[current]] != 1) {
    int next = block_permutation[current];
      memcpy(buf + current * block_size, buf + next * block_size, block_size);
      visited[current] = 1;
      current = next;
    }

    // Place the saved block in its final position
    memcpy(buf + current * block_size, temp, block_size);
    visited[current] = 1; // Mark the last block as visited
  }

  free(temp);
  free(visited);

  return MPI_SUCCESS;
}

/**
 * @brief Get the sender of a message based on the permutation.
 *
 * @param p The permutation array.
 * @param n The size of the permutation array (comm_sz).
 * @param i The receiver index.
 *
 * @return the sender on success, or -1 if error.
 */
static inline int get_sender(const int *p, int n, int i) {
  // Iterate over the array to find the index j for which p[j] == i.
  for(int j = 0; j < n; j++) {
      if(p[j] == i) {
          return j;  // Found the sender.
      }
  }
  return -1;
}

/*
 * rounddown: Rounds a number down to nearest multiple.
 *     rounddown(10,4) = 8, rounddown(6,3) = 6, rounddown(14,3) = 12
 */
static inline int rounddown(int num, int factor)
{
    num /= factor;
    return num * factor;    /* floor(num / factor) * factor */
}
static uint32_t binary_to_negabinary(int32_t bin) {
    if(SWING_UNLIKELY(bin > 0x55555555)) return -1;
    const uint32_t mask = 0xAAAAAAAA;
    return (mask + bin) ^ mask;
}


static inline int in_range(int x, uint32_t nbits){
    return x >= smallest_negabinary[nbits] && x <= largest_negabinary[nbits];
}

static inline uint32_t reverse(uint32_t x){
    x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
    x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
    x = ((x >> 4) & 0x0f0f0f0fu) | ((x & 0x0f0f0f0fu) << 4);
    x = ((x >> 8) & 0x00ff00ffu) | ((x & 0x00ff00ffu) << 8);
    x = ((x >> 16) & 0xffffu) | ((x & 0xffffu) << 16);
    return x;
}

static inline uint32_t get_rank_negabinary_representation(uint32_t num_ranks, uint32_t rank){
    binary_to_negabinary(rank);
    uint32_t nba = UINT32_MAX, nbb = UINT32_MAX;
    size_t num_bits = log_2(num_ranks);
    if(rank % 2){
        if(in_range(rank, num_bits)){
            nba = binary_to_negabinary(rank);
        }
        if(in_range(rank - num_ranks, num_bits)){
            nbb = binary_to_negabinary(rank - num_ranks);
        }
    }else{
        if(in_range(-rank, num_bits)){
            nba = binary_to_negabinary(-rank);
        }
        if(in_range(-rank + num_ranks, num_bits)){
            nbb = binary_to_negabinary(-rank + num_ranks);
        }
    }

    assert(nba != UINT32_MAX || nbb != UINT32_MAX);

    if(nba == UINT32_MAX && nbb != UINT32_MAX){
        return nbb;
    }else if(nba != UINT32_MAX && nbb == UINT32_MAX){
        return nba;
    }else{ // Check MSB
        if(nba & (80000000 >> (32 - num_bits))){
            return nba;
        }else{
            return nbb;
        }
    }
}

static inline uint32_t remap_rank(uint32_t num_ranks, uint32_t rank){
    uint32_t remap_rank = get_rank_negabinary_representation(num_ranks, rank);    
    remap_rank = remap_rank ^ (remap_rank >> 1);
    size_t num_bits = log_2(num_ranks);
    remap_rank = reverse(remap_rank) >> (32 - num_bits);
    return remap_rank;
}

static inline uint32_t inverse_rank(uint32_t num_ranks, uint32_t rank){
    size_t num_bits = log_2(num_ranks);
    return reverse(rank) >> (32 - num_bits);
}

static inline uint32_t get_sender_aux(uint32_t num_ranks, uint32_t rank, uint32_t root){
  uint32_t remap = remap_rank(num_ranks, rank);

  if (remap == root)  return rank;
  else                return get_sender_aux(num_ranks, remap, root);
}

static inline uint32_t get_sender_rec(uint32_t num_ranks, uint32_t rank){
  return get_sender_aux(num_ranks, rank, rank);
}

// NOTE: Commented since at the moment not used in the code
//
// /**
//  * This macro gives a generic way to compute the best count of
//  * the segment (i.e. the number of complete datatypes that
//  * can fit in the specified SEGSIZE). Beware, when this macro
//  * is called, the SEGCOUNT should be initialized to the count as
//  * expected by the collective call.
//  */
// define COLL_BASE_COMPUTED_SEGCOUNT(SEGSIZE, TYPELNG, SEGCOUNT)
//     if( ((SEGSIZE) >= (TYPELNG)) &&
//         ((SEGSIZE) < ((TYPELNG) * (SEGCOUNT))) ) {
//         size_t residual;
//         (SEGCOUNT) = (int)((SEGSIZE) / (TYPELNG));
//         residual = (SEGSIZE) - (SEGCOUNT) * (TYPELNG);
//         if( residual > ((TYPELNG) >> 1) )
//             (SEGCOUNT)++;
//     }
//
//
// typedef enum{
//   LIBSWING_DOUBLING = 0,
//   LIBSWING_HALVING
// }swing_direction_t;
//
//
// static inline int build_tree(int *tree, int root, int rank, int size, int *recv_step, swing_direction_t direction) {
//   int step, dest, idx = 0;
//   int steps = log_2(size);
//   char *received = NULL;
//
//   *recv_step = -1;
//   received = calloc(size, sizeof(char));
//   if (received == NULL) {
//     return -1;
//   }
//
//   received[root] = 1;
//   tree[idx++] = root;
//
//   for (step = 0; step < steps; step++) {
//     for (int proc = 0; proc < size; proc++) {
//       if (!received[proc]) continue;
//
//       dest = (direction == LIBSWING_DOUBLING) ? pi(proc, step, size) : pi(proc, steps - step - 1, size);
//       received[dest] = 1;
//       tree[idx++] = dest;
//       if (dest == rank) {
//         *recv_step = step;
//       }
//     }
//   }
//
//   free(received);
//   return 0;
// }
//
//
// static inline int build_both_trees(int *tree, int *halv_tree, int root, int rank, int size,
//                                    int *recv_step, int *halv_recv_step, int *pos, int *halv_pos) {
//   int step, dest, idx = 0, halv_idx = 0;
//   int steps = log_2(size);
//   char *received = NULL, *received_halv = NULL;
//
//   received = calloc(size, sizeof(char));
//   received_halv = calloc(size, sizeof(char));
//   if (received == NULL || received_halv == NULL) {
//     return -1;
//   }
//
//   *recv_step = -1;
//   *halv_recv_step = -1;
//   *pos = 0;
//   *halv_pos = 0;
//   received[root] = 1;
//   received_halv[root] = 1;
//   tree[idx++] = root;
//   halv_tree[halv_idx++] = root;
//
//   for (step = 0; step < steps; step++) {
//     // swing doubling tree
//     for (int proc = 0; proc < size; proc++) {
//       if (!received[proc]) continue;
//
//       dest = pi(proc, step, size);
//       if(received[dest]) continue;
//
//       received[dest] = 1;
//       tree[idx++] = dest;
//       if (dest == rank) {
//         *recv_step = step;
//         *pos = idx - 1;
//       }
//     }
//     // swing halving tree
//     for (int proc = 0; proc < size; proc++) {
//       if (!received_halv[proc]) continue;
//
//       dest = pi(proc, steps - step - 1, size);
//       if(received_halv[dest]) continue;
//
//       received_halv[dest] = 1;
//       halv_tree[halv_idx++] = dest;
//       if (dest == rank) {
//         *halv_recv_step = step;
//         *halv_pos = halv_idx - 1;
//       }
//     }
//   }
//
//   free(received);
//   free(received_halv);
//
//   return 0;
// }
//
//
// static inline int libswing_indexed_datatype(MPI_Datatype *new_dtype, const int *bitmap, int adj_size, int w_size,
//                                              const size_t small_block_count, const int split_rank,
//                                              MPI_Datatype old_dtype, int *block_len, int *disp){
//   int index = 0, disp_counter = 0;
//   for (int i = 0; i < adj_size; i++){
//     if (bitmap[i] != 0){
//       block_len[index] =  i < split_rank ? (int) (small_block_count + 1) : (int) small_block_count;
//       disp[index] = disp_counter;
//       index++;
//     }
//     disp_counter += i < split_rank ? (int) (small_block_count + 1): (int) small_block_count;
//   }
//
//   if (index != w_size){
//     return MPI_ERR_UNKNOWN;
//   }
//
//   MPI_Type_indexed(w_size, block_len, disp, old_dtype, new_dtype);
//   MPI_Type_commit(new_dtype);
//
//   return MPI_SUCCESS;
// }

#endif // LIBSWING_UTILS_H

