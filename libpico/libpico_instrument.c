/*
 * Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
 * Licensed under the MIT License
 */
#include <mpi.h>
#include "libpico.h"

#if defined PICO_INSTRUMENT && !defined PICO_NCCL && !defined PICO_MPI_CUDA_AWARE
#include <string.h>

/**
 * @brief Enable/disable tracking of nesting structure.
 * 
 * 0 = disabled (no stack, no parent→child edges)
 * 1 = enabled (record structure only; no extra timing overhead)
 *
 * Default: 1
 */
#ifndef LIBPICO_NESTING
#define LIBPICO_NESTING 1
#endif

#if LIBPICO_NESTING
  /* Max nesting depth; safe upper bound is number of tags */
  #ifndef LIBPICO_MAX_NESTING
  #define LIBPICO_MAX_NESTING LIBPICO_MAX_TAGS
  #endif

  /* Children adjacency bitset: for each parent, a set of child indices */
  enum { LIBPICO_CHILD_MASK_WORDS = (LIBPICO_MAX_TAGS + 63) / 64 };

  static unsigned long long pico_children[LIBPICO_MAX_TAGS][LIBPICO_CHILD_MASK_WORDS];
  static int  pico_stack[LIBPICO_MAX_NESTING];
  static int  pico_sp = 0;  /* stack pointer (0 = empty) */

  /* Set parent->child edge (idempotent) */
  static inline void pico_set_edge(int parent, int child) {
    pico_children[parent][(unsigned)child >> 6] |= (1ull << ((unsigned)child & 63));
  }
#endif /* LIBPICO_NESTING */

// ------------------------------------------------------------------------------------------------
//                                   INTERNAL DATA STRUCTURES
// ------------------------------------------------------------------------------------------------
typedef struct {
  const char *tag_name;  /* interned, stable pointer (lifetime = process) */
  double      accum;     /* accumulated elapsed time while depth==0 */
  double      last_start;/* last start timestamp (MPI_Wtime) when depth transitions 0->1 */
  int         depth;     /* re-entrant begin/end depth */
  int         active;    /* slot in use */
} libpico_tag_t;

typedef struct {
  libpico_tag_t *tag;    /* bound tag */
  int            out_len;/* length of out_buf */
  double        *out_buf;/* output buffer (owned by caller) */
} libpico_tag_handler_t;

// ------------------------------------------------------------------------------------------------
//                                      STATIC STATE
// ------------------------------------------------------------------------------------------------

/**
 * Name pool: fixed-size, bump-pointer arena for interned tag strings.
 * Size is derived from header config (LIBPICO_NAME_POOL_BYTES). Reset in libpico_init_tags().
 */
static char   libpico_name_pool[LIBPICO_NAME_POOL_BYTES];
static size_t libpico_name_pool_off = 0;

/* Tag table and handle table (bounded by LIBPICO_MAX_TAGS). */
static libpico_tag_t pico_tags[LIBPICO_MAX_TAGS];
static libpico_tag_handler_t pico_handles[LIBPICO_MAX_TAGS];
static int libpico_handles_built = 0;

// ----------------------------------------------------------------------------------------------
//                    Functions behind the LIBPICO_TAG_BEGIN/END macros
// ----------------------------------------------------------------------------------------------

/**
 * @brief Intern a string into the name pool.
 *
 * @param s The string to intern.
 * @return A pointer to the interned string, or NULL if the pool is full.
 *
 * @note This function is not to be called directly.
 */
static inline const char *_libpico_intern(const char *s) {
  size_t len = strlen(s) + 1;
  if (len > LIBPICO_NAME_POOL_BYTES - libpico_name_pool_off) return NULL;
  char *dst = &libpico_name_pool[libpico_name_pool_off];
  memcpy(dst, s, len);
  libpico_name_pool_off += len;
  return dst;
}

/**
* @brief Find the index of a tag by name.
*
* @param tag The name of the tag to find.
* @return The index of the tag, or -1 if not found.
*
* @note This function is not to be called directly.
*/
static inline int _libpico_find_tag(const char *tag) {
  for (int i = 0; i < LIBPICO_MAX_TAGS; ++i) {
    if (pico_tags[i].active && strcmp(pico_tags[i].tag_name, tag) == 0) 
      return i;
  }
  return -1;
}

/**
 * @brief Ensure that a tag exists, creating it if necessary.
 *
 * @param tag The name of the tag to ensure.
 * @return The index of the tag, or -1 on error (e.g., maximum tags exceeded).
 *
 * @note This function is not to be called directly.
 */
static inline int _libpico_ensure_tag(const char *tag) {
  int idx = _libpico_find_tag(tag);
  if (idx >= 0) return idx;
  for (int i = 0; i < LIBPICO_MAX_TAGS; ++i) {
    if (pico_tags[i].active) continue;

    const char *interned = _libpico_intern(tag);
    if (!interned) return -1;

    pico_tags[i].active     = 1;
    pico_tags[i].tag_name   = interned;
    pico_tags[i].accum      = 0.0;
    pico_tags[i].last_start = 0.0;
    pico_tags[i].depth      = 0;
    return i;
  }
  return -1;
}


// ----------------------------------------------------------------------------------------------
//                   Tag begin/end functions called by the macros
// ----------------------------------------------------------------------------------------------

/**
 * Contract:
 *  - Tag is created on first use (libpico_ensure_tag).
 *  - Tag depth is checked for consistency (>=0 before begin, >0 before end).
 *  - Returns 0 on success, -1 on error (stderr message emitted).
 *  - Uses MPI_Wtime() only when depth transitions 0->1 (begin) and 1->0 (end).
 */

int libpico_tag_begin(const char *tag) {
  if (!tag) {
    fprintf(stderr, "Error: NULL tag passed to libpico_tag_begin.\n");
    return -1;
  }
  int idx = _libpico_ensure_tag(tag);
  if (idx < 0) {
    fprintf(stderr, "Error: Maximum number of tags (%d) exceeded.\n", LIBPICO_MAX_TAGS);
    return -1;
  }

#if LIBPICO_NESTING
  /* Record structure: if there is a current parent, add parent->child edge */
  if (pico_sp > 0) {
    int parent = pico_stack[pico_sp - 1];
    pico_set_edge(parent, idx);
  }
  /* Push this tag on the call stack */
  if (pico_sp >= LIBPICO_MAX_NESTING) {
    fprintf(stderr, "Error: nesting exceeds LIBPICO_MAX_NESTING=%d\n", LIBPICO_MAX_NESTING);
    return -1;
  }
  pico_stack[pico_sp++] = idx;
#endif

  if (pico_tags[idx].depth < 0) {
    fprintf(stderr, "Error: Tag '%s' has invalid depth '%d' before beginning.\n", 
            tag, pico_tags[idx].depth);
    return -1;
  }

  if (pico_tags[idx].depth == 0){
    pico_tags[idx].last_start = MPI_Wtime();
  }
  pico_tags[idx].depth++;
  return 0;
}

int libpico_tag_end(const char *tag) {
  if (!tag) {
    fprintf(stderr, "Error: NULL tag passed to libpico_tag_end.\n");
    return -1;
  }

  int idx = _libpico_find_tag(tag);
  if (idx < 0) {
    fprintf(stderr, "Error: Tag '%s' was not initialized before ending.\n", tag);
    return -1;
  }

  if (pico_tags[idx].depth <= 0) {
    fprintf(stderr, "Error: Tag '%s' was not properly begun before ending.\n", tag);
    return -1;
  }

  pico_tags[idx].depth -= 1;
  if (pico_tags[idx].depth == 0){
    pico_tags[idx].accum += MPI_Wtime() - pico_tags[idx].last_start;
  }

#if LIBPICO_NESTING
  if (pico_sp <= 0 || pico_stack[pico_sp - 1] != idx) {
    fprintf(stderr, "Error: unbalanced nesting for tag '%s'\n", tag);
    return -1;
  }
  pico_sp--;
#endif
  return 0;
}


// ----------------------------------------------------------------------------------------------
//                    Tag handle management functions (used in pico core)
// ----------------------------------------------------------------------------------------------

/**
 * @brief Initialize all tags to unused state.
 *
 * @note This function is not to be called directly; use libpico_init_tags() instead.
 */
static inline void libpico_initialize_all_tags(void) {
  for (int i = 0; i < LIBPICO_MAX_TAGS; ++i) {
    pico_tags[i].tag_name   = NULL;
    pico_tags[i].accum  = 0.0;
    pico_tags[i].last_start = 0.0;
    pico_tags[i].depth = 0;
    pico_tags[i].active = 0;
  }
}

/**
 * @brief Initialize all bindings to unused state.
 *
 *  @note This function is not to be called directly; use libpico_init_tags() instead.
 */
static inline void libpico_initialize_all_bindings(void) {
  for (int i = 0; i < LIBPICO_MAX_TAGS; ++i) {
    pico_handles[i].tag     = NULL;
    pico_handles[i].out_len = 0;
    pico_handles[i].out_buf = NULL;
  }
}

void libpico_init_tags(void) {
  libpico_initialize_all_tags();
  libpico_initialize_all_bindings();
  libpico_handles_built = 0;
  libpico_name_pool_off = 0;
#if LIBPICO_NESTING
  pico_sp = 0;
  memset(pico_children, 0, sizeof pico_children);
#endif
}


int libpico_count_tags(void) {
  int n = 0;
  for (int i = 0; i < LIBPICO_MAX_TAGS; ++i) {
    if (pico_tags[i].active) ++n;
  }
  return n;
}

int libpico_get_tag_names(const char **names, int count) {
  if (names == NULL || count <= 0) {
    fprintf(stderr, "Error: Invalid arguments to libpico_get_tag_names.\n");
    return -1;
  }

  int written = 0;
  for (int i = 0; i < LIBPICO_MAX_TAGS && written < count; ++i) {
    if (!pico_tags[i].active) continue;

    if (pico_tags[i].tag_name == NULL) {
      fprintf(stderr, "Error: Inconsistent state: active tag with NULL name.\n");
      return -1;
    }
    names[written++] = pico_tags[i].tag_name;
  }

  if (written != count) {
    fprintf(stderr, "Error: Mismatch in tag count. Expected %d, found %d.\n", count, written);
    return -1;
  }
  return 0;
}

int libpico_build_handles(double **bufs, int k, int out_len) {
  if (!bufs || k <= 0 || out_len <= 0) {
    fprintf(stderr, "Error: Invalid arguments to libpico_build_handles.\n");
    return -1;
  }

  int tag_cnt = 0;
  for (int i = 0; i < LIBPICO_MAX_TAGS; ++i)
    if (pico_tags[i].active) ++tag_cnt;

  if (tag_cnt != k){
    fprintf(stderr, "Error: Number of active tags (%d) does not match number of buffers (%d).\n",
            tag_cnt, k);
    return -1;
  }

  int seen = 0;
  for (int i = 0; i < LIBPICO_MAX_TAGS && seen < k; ++i) {
    if (!pico_tags[i].active) continue;

    if (!bufs[seen]){
      fprintf(stderr, "Error: NULL buffer provided for tag '%s'.\n", pico_tags[i].tag_name);
      return -1;
    }

    pico_handles[seen].tag = &pico_tags[i];
    pico_handles[seen].out_buf = bufs[seen];
    pico_handles[seen].out_len = out_len;
    ++seen;
  }
  if (seen != k){
    fprintf(stderr, "Error: Mismatch in tag count. Expected %d, found %d.\n", k, seen);
    return -1;
  }

  libpico_handles_built = k;
  for (int i = k; i < LIBPICO_MAX_TAGS; ++i) {
    pico_handles[i].tag     = NULL;
    pico_handles[i].out_buf = NULL;
    pico_handles[i].out_len = 0;
  }

  return 0;
}


int libpico_clear_tags(void) {
  for (int i = 0; i < LIBPICO_MAX_TAGS; ++i) {
    if (!pico_tags[i].active) continue;

    if (pico_tags[i].depth != 0) {
      fprintf(stderr, "Error: Tag '%s' was not properly ended before clearing.\n", pico_tags[i].tag_name);
      return -1;
    }
    pico_tags[i].last_start = 0.0;
    pico_tags[i].accum = 0.0;
  }
  return 0;
}

int libpico_snapshot_store(int iter_idx) {
  int k = libpico_handles_built;
  if (iter_idx < 0 || k <= 0) {
    fprintf(stderr, "Error: Invalid arguments to libpico_snapshot_store (iter_idx=%d, k=%d).\n", iter_idx, k);
    return -1;
  }

  for (int i = 0; i < k; ++i) {
    libpico_tag_t *tag = pico_handles[i].tag;
    if (!tag || !tag->active || !pico_handles[i].out_buf) {
      fprintf(stderr, "Error: Invalid handle at index %d.\n", i);
      return -1;
    }
    if (iter_idx >= pico_handles[i].out_len) {
      fprintf(stderr, "Error: Iteration index %d out of bounds for handle %d (length %d).\n",
              iter_idx, i, pico_handles[i].out_len);
      return -1;
    }
    if (tag->depth != 0) {
      fprintf(stderr, "Error: Tag '%s' must be closed (depth==0) before snapshot.\n", tag->tag_name);
      return -1;
    }
    double v = tag->accum;
    if (v < 0.0) {
      fprintf(stderr, "Error: Inconsistent state: tag '%s' has negative accumulated time.\n", tag->tag_name);
      return -1;
    }
    pico_handles[i].out_buf[iter_idx] = v;
  }
  return 0;
}

#else

// ----------------------------------------------------------------------------------------------
//                         Stubs when PICO_INSTRUMENT is not defined
// ----------------------------------------------------------------------------------------------

int libpico_tag_begin(const char *tag) { (void)tag; return 0; }

int libpico_tag_end(const char *tag) { (void)tag; return 0; }

void libpico_init_tags(void) { }

int libpico_count_tags(void) { return 0; }

int libpico_get_tag_names(const char **names, int count) { if (names || count) {} return 0; }

int libpico_build_handles(double **bufs, int k, int out_len) { if (bufs || k || out_len) {} return 0; }

int libpico_clear_tags(void) { return 0; }

int libpico_snapshot_store(int iter_idx) { (void)iter_idx; return 0; }

#endif // PICO_INSTRUMENT && !PICO_NCCL && !PICO_MPI_CUDA_AWARE

