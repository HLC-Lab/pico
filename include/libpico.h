/*
 * Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
 * Licensed under the MIT License
 */

#ifndef LIBPICO_H
#define LIBPICO_H

#if defined PICO_INSTRUMENT && !defined PICO_NCCL && !defined PICO_MPI_CUDA_AWARE
#include <stdio.h>
#endif

#include <stddef.h>
#include <mpi.h>

#define ALLREDUCE_MPI_ARGS        const void *sbuf, void *rbuf, size_t count, \
                                  MPI_Datatype dtype, MPI_Op op, MPI_Comm comm
#define ALLGATHER_MPI_ARGS        const void *sbuf, size_t scount, MPI_Datatype sdtype, \
                                  void* rbuf, size_t rcount, MPI_Datatype rdtype, MPI_Comm comm
#define ALLTOALL_MPI_ARGS         const void *sbuf, size_t scount, MPI_Datatype sdtype, \
                                  void *rbuf, size_t rcount, MPI_Datatype rdtype, MPI_Comm comm
#define BCAST_MPI_ARGS            void *buf, size_t count, MPI_Datatype dtype, int root, MPI_Comm comm
#define GATHER_MPI_ARGS           const void *sbuf, size_t scount, MPI_Datatype sdtype, \
                                  void *rbuf, size_t rcount, MPI_Datatype rdtype, int root, MPI_Comm comm
#define REDUCE_MPI_ARGS           const void *sbuf, void *rbuf, size_t count, \
                                  MPI_Datatype dtype, MPI_Op op, int root, MPI_Comm comm
#define REDUCE_SCATTER_MPI_ARGS   const void *sbuf, void *rbuf, const int rcounts[], \
                                  MPI_Datatype dtype, MPI_Op op, MPI_Comm comm
#define SCATTER_MPI_ARGS          const void *sbuf, size_t scount, MPI_Datatype sdtype, \
                                  void *rbuf, size_t rcount, MPI_Datatype rdtype, int root, MPI_Comm comm
#ifdef PICO_NCCL
#include <nccl.h>
#include <cuda_runtime.h>

#define ALLREDUCE_NCCL_ARGS       const void *sbuf, void *rbuf, size_t count, ncclDataType_t dtype, \
                                  ncclRedOp_t op, ncclComm_t nccl_comm, cudaStream_t stream
#define ALLGATHER_NCCL_ARGS       const void *sbuf, void* rbuf, size_t count, \
                                  ncclDataType_t dtype, ncclComm_t nccl_comm, cudaStream_t stream
#define ALLTOALL_NCCL_ARGS        const void *sbuf, void* rbuf, size_t count,\
                                  ncclDataType_t dtype, ncclComm_t nccl_comm, cudaStream_t stream
#define BCAST_NCCL_ARGS           void *buf, size_t count, ncclDataType_t dtype, \
                                  int root, ncclComm_t nccl_comm, cudaStream_t stream
#define GATHER_NCCL_ARGS          const void *sbuf, void *rbuf, size_t count, ncclDataType_t dtype, \
                                  int root, ncclComm_t nccl_comm, cudaStream_t stream
#define REDUCE_NCCL_ARGS          const void *sbuf, void *rbuf, size_t count, ncclDataType_t dtype, \
                                  ncclRedOp_t op, int root, ncclComm_t nccl_comm, cudaStream_t stream
#define REDUCE_SCATTER_NCCL_ARGS  const void *sbuf, void *rbuf, size_t rcount, ncclDataType_t dtype, \
                                  ncclRedOp_t op, ncclComm_t nccl_comm, cudaStream_t stream
#define SCATTER_NCCL_ARGS         const void *sbuf, void *rbuf, size_t count, ncclDataType_t dtype, \
                                  int root, ncclComm_t nccl_comm, cudaStream_t stream
#endif

extern size_t bine_allreduce_segsize;

int allreduce_recursivedoubling(ALLREDUCE_MPI_ARGS);
int allreduce_ring(ALLREDUCE_MPI_ARGS);
int allreduce_rabenseifner(ALLREDUCE_MPI_ARGS);
int allreduce_bine_lat(ALLREDUCE_MPI_ARGS);
int allreduce_bine_bdw_remap(ALLREDUCE_MPI_ARGS);
int allreduce_bine_bdw_remap_segmented(ALLREDUCE_MPI_ARGS);
int allreduce_bine_block_by_block_any_even(ALLREDUCE_MPI_ARGS);

int allgather_k_bruck(ALLGATHER_MPI_ARGS);
int allgather_recursivedoubling(ALLGATHER_MPI_ARGS);
int allgather_ring(ALLGATHER_MPI_ARGS);
int allgather_sparbit(ALLGATHER_MPI_ARGS);
int allgather_bine_block_by_block(ALLGATHER_MPI_ARGS);
int allgather_bine_block_by_block_any_even(ALLGATHER_MPI_ARGS);
int allgather_bine_send_remap(ALLGATHER_MPI_ARGS);
int allgather_bine_2_blocks(ALLGATHER_MPI_ARGS);
int allgather_bine_2_blocks_dtype(ALLGATHER_MPI_ARGS);

int alltoall_pairwise_ompi(ALLTOALL_MPI_ARGS);
int alltoall_bine(ALLTOALL_MPI_ARGS);

int bcast_linear(BCAST_MPI_ARGS);
int bcast_binomial_halving(BCAST_MPI_ARGS);
int bcast_binomial_doubling(BCAST_MPI_ARGS);
int bcast_scatter_allgather(BCAST_MPI_ARGS);
int bcast_bine_lat(BCAST_MPI_ARGS);
int bcast_bine_lat_reversed(BCAST_MPI_ARGS);
int bcast_bine_lat_new(BCAST_MPI_ARGS);
int bcast_bine_lat_i_new(BCAST_MPI_ARGS);
int bcast_bine_bdw_remap(BCAST_MPI_ARGS);

int gather_linear(GATHER_MPI_ARGS);
int gather_bine(GATHER_MPI_ARGS);

int reduce_bine_lat(REDUCE_MPI_ARGS);
int reduce_bine_bdw(REDUCE_MPI_ARGS);

int reduce_scatter_recursivehalving(REDUCE_SCATTER_MPI_ARGS);
int reduce_scatter_recursive_distance_doubling(REDUCE_SCATTER_MPI_ARGS);
int reduce_scatter_ring(REDUCE_SCATTER_MPI_ARGS);
int reduce_scatter_butterfly(REDUCE_SCATTER_MPI_ARGS);
int reduce_scatter_bine_send_remap(REDUCE_SCATTER_MPI_ARGS);
int reduce_scatter_bine_permute_remap(REDUCE_SCATTER_MPI_ARGS);
int reduce_scatter_bine_block_by_block(REDUCE_SCATTER_MPI_ARGS);
int reduce_scatter_bine_block_by_block_any_even(REDUCE_SCATTER_MPI_ARGS);

int scatter_linear(SCATTER_MPI_ARGS);
int scatter_bine(SCATTER_MPI_ARGS);

/**
 * Instrumentation support
 */

/** Maximum number of distinct active tags that can be created in a run. */
#ifndef LIBPICO_MAX_TAGS
#define LIBPICO_MAX_TAGS 32
#endif

/** Maximum tag name length including the NUL terminator. */
#ifndef LIBPICO_TAG_NAME_MAX
#define LIBPICO_TAG_NAME_MAX 32
#endif

#ifndef LIBPICO_NAME_POOL_BYTES
#define LIBPICO_NAME_POOL_BYTES (LIBPICO_MAX_TAGS * LIBPICO_TAG_NAME_MAX)
#endif

/* ---- dispatcher: 1 arg vs 2 args ---- */
#define _PICO_TAG_PICK(_1,_2,NAME,...) NAME

// ----------------------------------------------------------------------------------------------
//                        PUBLIC API Maros for instrumentation
// ----------------------------------------------------------------------------------------------

#if defined PICO_INSTRUMENT && !defined PICO_NCCL


/**
 * @brief Format "<base>:<idx>" into @p dst without allocating or interning.
 *        This is a pure formatter for the 2-arg macro path.
 *
 * @param dst  Destination buffer.
 * @param cap  Capacity of @p dst in bytes (must be >= LIBPICO_TAG_NAME_MAX).
 * @param base Base tag name (may be NULL, treated as "").
 * @param v    Integer suffix (the 2nd arg of the macro).
 *
 * @return The number of characters written (excluding the NUL) on success,
 *         or -1 if truncated or any snprintf error occurred.
 *
 * @note Maximum tag length is limited by LIBPICO_TAG_NAME_MAX (includes NUL).
 *       If "<base>:<idx>" does not fit, this returns -1 and the macros abort.
 */
static inline int libpico_format_tag(char *dst, size_t cap, const char *base, int v) {
  int n = snprintf(dst, cap, "%s:%d", base ? base : "", v);
  return (n < 0 || (size_t)n >= cap) ? -1 : n;
}

/* Back-end calls; return 0 on success, -1 on error. */
int libpico_tag_begin(const char *tag);
int libpico_tag_end(const char *tag);

/**
 * PICO_TAG_BEGIN(name) / PICO_TAG_END(name)
 *   - Start/stop a tag by string name. Tag is created on first use.
 *
 * PICO_TAG_BEGIN(name, idx:int) / PICO_TAG_END(name, idx:int)
 *   - Derived tag form: formats "<name>:<idx>" on a local stack buffer, then
 *     calls the 1-arg path. The caller must ensure idx is an int (cast if needed).
 *
 * Error semantics:
 *   - On any failure, these macros execute `return -1;` in the CALLER.
 *     Use only in functions that return int.
 */


/* API macros: expand to 1-arg or 2-arg form */
#define PICO_TAG_BEGIN(...) _PICO_TAG_PICK(__VA_ARGS__, PICO_TAG_BEGIN2, PICO_TAG_BEGIN1)(__VA_ARGS__)
#define PICO_TAG_END(...)   _PICO_TAG_PICK(__VA_ARGS__, PICO_TAG_END2,   PICO_TAG_END1  )(__VA_ARGS__)

/* 1-arg forms: pass the provided name through to the backend. */
#define PICO_TAG_BEGIN1(TAG) do { if (libpico_tag_begin((TAG)) != 0) return -1; } while (0)
#define PICO_TAG_END1(TAG)   do { if (libpico_tag_end((TAG))   != 0) return -1; } while (0)

/* 2-arg forms: compose "<base>:<int>" on the stack; backend will intern it. */
#define PICO_TAG_BEGIN2(BASE, IDX) do {                                                     \
  char _pico_name_[LIBPICO_TAG_NAME_MAX];                                                   \
  if (libpico_format_tag(_pico_name_, sizeof _pico_name_, (BASE), (int)(IDX)) < 0) {        \
    fprintf(stderr, "PICO_TAG_BEGIN error: tag too long at %s:%d: PICO_TAG_BEGIN(%s,%d)\n", \
            __FILE__, __LINE__, (BASE) ? (BASE) : "(null)", (int)(IDX));                    \
    return -1;                                                                              \
  }                                                                                         \
  if (libpico_tag_begin(_pico_name_) != 0) return -1;                                       \
} while (0)

#define PICO_TAG_END2(BASE, IDX) do {                                                   \
  char _pico_name_[LIBPICO_TAG_NAME_MAX];                                               \
  if (libpico_format_tag(_pico_name_, sizeof _pico_name_, (BASE), (int)(IDX)) < 0){     \
    fprintf(stderr, "PICO_TAG_END error: tag too long at %s:%d: PICO_TAG_END(%s,%d)\n", \
            __FILE__, __LINE__, (BASE) ? (BASE) : "(null)", (int)(IDX));                \
    return -1;                                                                          \
  }                                                                                     \
  if (libpico_tag_end(_pico_name_) != 0) return -1;                                     \
} while (0)

#else /* instrumentation disabled: macros are no-ops for zero overhead */

#define PICO_TAG_BEGIN(...) do {} while (0)
#define PICO_TAG_END(...)   do {} while (0)

#endif


// ----------------------------------------------------------------------------------------------
//                   Functions for managing tags, used in pico core
// ----------------------------------------------------------------------------------------------

/**
 * @brief Init all tags and bindings to unused state. Pico core call this once 
 *        at the start of the benchmarking run.
 */
void libpico_init_tags(void);

/**
 * @brief Count of currenly used tags for sizing arrays in pico core
 *
 * @return number of tags in use (<= LIBPICO_MAX_TAGS)
 */
int libpico_count_tags(void);

/**
 * @brief Writes the names of the currently used tags into the provided array.
 *
 * @param names Array to write the tag names into.
 * @param max Current number of tags in use (must be retrieved via libpico_count_tags()).
 *
 * @return 0 on success, -1 on error.
 */
int libpico_get_tag_names(const char **names, int count);

/**
 * @brief Binds the provided buffers to the currently active tags.
 *
 * @param bufs Array of buffers, one for each active tag.
 * @param k Number of active tags (must be retrieved via libpico_count_tags()).
 * @param out_len Length of each buffer (must be = benchmaking iter).
 *
 * @return 0 on success, -1 on error.
 */
int libpico_build_handles(double **bufs, int k, int out_len);

/**
 * @brief Clear all active tag accum and last start values.
 *
 * @return 0 on success, -1 on error.
 */
int libpico_clear_tags(void);

/**
 * @brief Store the current accum values of all active tags into the provided buffers.
 *
 * @param iter_idx Current benchmarking iteration index.
 * @param k Number of active tags (must be retrieved via libpico_count_tags()).
 *
 * @return 0 on success, -1 on error.
 */
int libpico_snapshot_store(int iter_idx);

#endif // LIBPICO_H
