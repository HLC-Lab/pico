/*
 * Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
 * Licensed under the MIT License
 */

#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <unistd.h>  /* usleep, for the optional TWILL_SKEW_US skew injection */

#include "libpico.h"
#include "libpico_utils.h"  /* includes the TWILL schedule core (twill_* helpers) */


/* Alltoall pairwise implementation from Open MPI 5.0.1 base module.
 * Original file: ompi/mca/coll/base/coll_base_alltoall.c
 * Original function: ompi_coll_base_alltoall_intra_pairwise
 */
int alltoall_pairwise_ompi(const void *sbuf, size_t scount, MPI_Datatype sdtype, 
                           void* rbuf, size_t rcount, MPI_Datatype rdtype, MPI_Comm comm)
{
  int line = -1, err = 0, rank, size, step, sendto, recvfrom;
  void * tmpsend, *tmprecv;
  ptrdiff_t lb, sext, rext;

  if (MPI_IN_PLACE == sbuf) {
    err = MPI_ERR_ARG;
    line = __LINE__;
    goto err_hndl;
  }

  MPI_Comm_rank (comm, &rank);
  MPI_Comm_size (comm, &size);

  err = MPI_Type_get_extent(sdtype, &lb, &sext);
  if (err != MPI_SUCCESS) { line = __LINE__; goto err_hndl; }
  err = MPI_Type_get_extent(rdtype, &lb, &rext);
  if (err != MPI_SUCCESS) { line = __LINE__; goto err_hndl; }


  /* Perform pairwise exchange - starting from 1 so the local copy is last */
  for (step = 1; step < size + 1; step++) {

    /* Determine sender and receiver for this step. */
    sendto  = (rank + step) % size;
    recvfrom = (rank + size - step) % size;

    /* Determine sending and receiving locations */
    tmpsend = (char*)sbuf + (ptrdiff_t)sendto * sext * (ptrdiff_t)scount;
    tmprecv = (char*)rbuf + (ptrdiff_t)recvfrom * rext * (ptrdiff_t)rcount;

    /* send and receive */
    err = MPI_Sendrecv(tmpsend, scount, sdtype, sendto, 0,
                       tmprecv, rcount, rdtype, recvfrom, 0,
                       comm, MPI_STATUS_IGNORE);
    if (err != MPI_SUCCESS) { line = __LINE__; goto err_hndl;  }
  }

  return MPI_SUCCESS;

 err_hndl:
  fprintf(stderr, "\n%s:%4d\tRank %d Error occurred %d\n\n", __FILE__, line, rank, err);
  (void)line;  // silence compiler warning
  return err;
}

int alltoall_bine(const void *sendbuf, size_t s_count, MPI_Datatype s_dtype,
                   void *recvbuf, size_t r_count, MPI_Datatype r_dtype, MPI_Comm comm)
{
  assert(s_count == r_count);
  assert(s_dtype == r_dtype);
  int rank, size, dtsize, err = MPI_SUCCESS;
  int inverse_mask, mask = 0x1, block_first_mask;
  size_t num_resident_blocks, num_resident_blocks_next, min_block_s, max_block_s;
  size_t sbuf_size, tmpbuf_size, tmpbuf_size_real;
  char *tmpbuf = NULL;
  uint *resident_block, *resident_block_next;
  // resident_block[i] contains the id of a block that is resident in the current rank (for i < num_resident_blocks)
  // resident_block_next[i] contains the id of a block that is resident in the current rank in the next step (for i < num_resident_blocks_next)
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  MPI_Type_size(s_dtype, &dtsize);

  num_resident_blocks = size;
  num_resident_blocks_next = 0;
  sbuf_size = s_count * dtsize;
  tmpbuf_size = sbuf_size * size;
  tmpbuf_size_real = tmpbuf_size + sizeof(uint) * size + sizeof(uint) * size;

  tmpbuf = (char *) malloc(tmpbuf_size_real);
  if(tmpbuf == NULL){
    err = MPI_ERR_NO_MEM;
    goto err_hndl;
  }
  resident_block = (uint *) (tmpbuf + tmpbuf_size);
  resident_block_next = (uint *) (tmpbuf + tmpbuf_size + sizeof(uint) * size);

  // At the beginning I only have my blocks
  for(size_t i = 0; i < size; i++){
    resident_block[i] = i;
  }

  memcpy(tmpbuf, sendbuf, tmpbuf_size);

  // We use recvbuf to receive/send the data, and tmpbuf to organize the data to send at the next step
  // By doing so, we avoid a copy form tmpbuf to recvbuf at the end
  inverse_mask = 0x1 << (int) (log_2(size) - 1);
  block_first_mask = ~(inverse_mask - 1);

  while(mask < size){
    int partner;
    int ntbn = negabinary_to_binary((mask << 1) -1);
    if(rank % 2 == 0){
      partner = mod(rank + ntbn, size); 
    } else {
      partner = mod(rank - ntbn, size); 
    }
    min_block_s = remap_rank(size, partner) & block_first_mask;
    max_block_s = min_block_s + inverse_mask - 1;

    size_t block_recvd_cnt = 0, block_send_cnt = 0;
    size_t offset_send = 0, offset_keep = 0;
    num_resident_blocks_next = 0;
    for(size_t i = 0; i < size; i++){
      uint block = resident_block[i % num_resident_blocks];
      // Shall I send this block? Check the negabinary thing  
      uint remap_block = remap_rank(size, block);
      size_t offset = i * sbuf_size;

      // I move to the beginning of tmpbuf the blocks I want to keep,
      // and I move to recvbuf the blocks I want to send.
      if(remap_block >= min_block_s && remap_block <= max_block_s){
        memcpy((char*) recvbuf + offset_send, tmpbuf + offset, sbuf_size);
        offset_send += sbuf_size;
        block_send_cnt++;
      }else{
        // Copy the blocks we are not sending to the second half of recvbuf
        if(offset != offset_keep){
          memcpy(tmpbuf + offset_keep, tmpbuf + offset, sbuf_size);
        }
        offset_keep += sbuf_size;
        block_recvd_cnt++;

        resident_block_next[num_resident_blocks_next] = block;
        num_resident_blocks_next++;
      }
    }
    assert(block_recvd_cnt == size/2);
    assert(block_send_cnt == size/2);
    num_resident_blocks /= 2;

    // I receive data in the second half of tmpbuf (the first half contains the blocks I am keeping from previous iteration)
    err = MPI_Sendrecv((char*) recvbuf, s_count * block_send_cnt, s_dtype, partner, 0,
                       tmpbuf + (size / 2) * sbuf_size, s_count * block_send_cnt, s_dtype, partner, 0, 
                       comm, MPI_STATUS_IGNORE);
    if(err != MPI_SUCCESS) { goto err_hndl; }

    // Update resident blocks
    memcpy(resident_block, resident_block_next, sizeof(uint) * num_resident_blocks);

    mask <<= 1;
    inverse_mask >>= 1;
    block_first_mask >>= 1;
  }

  // Now I need to permute tmpbuf into recvbuf
  // Since I always received the new block on the right, and moved the blocks
  // I wanted to keep to the left, they are now sorted in the same order they reached this
  // rank from their corresponding source ranks. 
  // I.e., I should consider the "reverse tree" (with this rank at the bottom and all the other ranks on top),
  // which represent how the data arrived here.
  // This tree is basically the opposite that I used to send the data
  // I should consider the decreasing tree, and viceversa.
  for(size_t i = 0; i < size; i++){
    int rotated_i = 0;
    if((rank % 2) == 0){
      rotated_i = mod(i - rank, size);
    } else {
      rotated_i = mod(rank - i, size);
    }
    int repr = 0;
    if(in_range(rotated_i, log_2(size))){
      repr = binary_to_negabinary(rotated_i);
    }else{
      repr = binary_to_negabinary(rotated_i - size);
    }
    int index = remap_distance_doubling(repr);

    size_t offset_src = index * sbuf_size;
    size_t offset_dst = i * sbuf_size;
    memcpy((char*) recvbuf + offset_dst, tmpbuf + offset_src, sbuf_size);
  }

  free(tmpbuf);
  return MPI_SUCCESS;

err_hndl:
  if(tmpbuf != NULL) free(tmpbuf);
  return err;
}


/* ===========================================================================
 * TWILL — Tapered-Weight Interleaved Linear aLLtoall.
 *
 * External alltoall algorithms that keep instantaneous per-group load
 * proportional to group capacity on tapered topologies, while every step is a
 * perfect matching (no rank-level incast, analytically invertible receives).
 *
 * Three variants share one windowed nonblocking engine; they differ only in the
 * relabeling rho (see the twill_* schedule helpers in libpico_utils.h):
 *   - alltoall_twill_group  : smooth weighted round-robin over groups (contender)
 *   - alltoall_twill_random : pseudorandom permutation from a shared seed
 *   - alltoall_twill_shift  : identity relabeling -> classic (s+t) mod P
 *
 * Environment knobs (read once per context build, like SEGSIZE):
 *   TWILL_WINDOW  outstanding-op window           (default 32, clamped >= 1)
 *   TWILL_SEED    seed for twill_random           (default fixed constant)
 *   TWILL_CACHE   1: cache schedule across calls   (default 1)
 *                 0: rebuild/free schedule per call (map discovery still once)
 *   TWILL_MAP     path to rank->group id map file (one id per rank, any ids)
 *   TWILL_GROUP   when no TWILL_MAP: an integer N = synthetic uniform groups of
 *                 size N (rank i -> i/N), or "node" (default) = node = group
 *   TWILL_SKEW_US per-call random pre-exchange sleep, microseconds (default 0)
 *
 * v1 scope: CPU / host MPI buffers. The self-block copy is a host memcpy (as in
 * alltoall_bine); do not rely on this on device buffers. Not wired into NCCL.
 * See libpico/twill.md for details.
 * =========================================================================== */

#define TWILL_TAG          0x7711
#define TWILL_DEFAULT_SEED 0x9e3779b97f4a7c15ULL

/* ---------------------------------------------------------------------------
 * Cached, lazily-initialized context (PICO's fixed signature has no room for a
 * context argument; this mirrors the module-level state idiom of libpico).
 * ------------------------------------------------------------------------- */
typedef struct {
  int       valid;            /* context initialized for (comm, P)            */
  int       P;                /* comm size at build                           */
  MPI_Comm  comm;             /* comm handle at build (cache validation)      */

  int       window;          /* TWILL_WINDOW                                  */
  int       cache;           /* TWILL_CACHE                                   */
  uint64_t  seed;            /* TWILL_SEED                                    */
  int       skew_us;         /* TWILL_SKEW_US (0 = off): max pre-exchange skew */
  uint64_t  calls;           /* per-context call counter (skew randomization) */

  int      *grp;             /* dense group id per rank (group variant only)  */
  int       G;               /* number of groups                             */
  int       map_ready;       /* map discovered (retained even if cache==0)    */
  int       checked;         /* cross-rank consistency verified once          */

  int      *rho[3];          /* per-variant relabeling (indexed by kind)      */
  int      *rho_inv[3];

  int          ntotal_cap;   /* capacity of the engine scratch arrays         */
  MPI_Request *reqs;
  int         *slot_kind;    /* 0 = recv slot, 1 = send slot                  */
  int         *done_idx;
  MPI_Status  *statuses;
} twill_ctx_t;

static twill_ctx_t g_twill_ctx;               /* zero-initialized */
static int         g_twill_atexit_registered; /* register the host-memory free once */
static int         g_twill_warned_single;     /* "single group" warning, rank 0 once */

/* ---------------------------------------------------------------------------
 * Host-memory teardown. No MPI calls (may run after MPI_Finalize via atexit).
 * Every pointer is NULLed after free so the per-call (cache==0) free path and
 * the atexit path cannot double-free.
 * ------------------------------------------------------------------------- */
static void twill_free_compute(twill_ctx_t *c) {
  for (int k = 0; k < 3; k++) {
    free(c->rho[k]);     c->rho[k]     = NULL;
    free(c->rho_inv[k]); c->rho_inv[k] = NULL;
  }
  free(c->reqs);      c->reqs      = NULL;
  free(c->slot_kind); c->slot_kind = NULL;
  free(c->done_idx);  c->done_idx  = NULL;
  free(c->statuses);  c->statuses  = NULL;
  c->ntotal_cap = 0;
}

static void twill_free_all(twill_ctx_t *c) {
  twill_free_compute(c);
  free(c->grp); c->grp = NULL;
  c->G = 0; c->map_ready = 0; c->checked = 0; c->valid = 0;
}

static void twill_atexit(void) { twill_free_all(&g_twill_ctx); }

/* ---- env helpers ---------------------------------------------------------- */
static int twill_env_int(const char *name, int def) {
  const char *v = getenv(name);
  if (v == NULL || *v == '\0') { return def; }
  return (int)strtol(v, NULL, 10);
}
static uint64_t twill_env_u64(const char *name, uint64_t def) {
  const char *v = getenv(name);
  if (v == NULL || *v == '\0') { return def; }
  return (uint64_t)strtoull(v, NULL, 0);  /* base 0: accept 0x.. hex */
}

/* ---------------------------------------------------------------------------
 * Group map discovery (evaluated once per context), in priority order:
 *   1. TWILL_MAP=<path> : every rank reads the file (one group id per rank).
 *   2. TWILL_GROUP=<N>  : synthetic uniform contiguous groups of size N
 *                         (rank i -> i/N), independent of task placement.
 *   3. TWILL_GROUP=node : node = group via MPI_COMM_TYPE_SHARED + allgather
 *                         (the default when TWILL_GROUP is unset).
 *   4. single group     : fallback (warn once on rank 0).
 * ------------------------------------------------------------------------- */
static int twill_discover_map(MPI_Comm comm, int rank, int P,
                              int **grp_out, int *G_out) {
  int *grp = (int *)malloc((size_t)P * sizeof(int));
  if (grp == NULL) { return MPI_ERR_NO_MEM; }

  const char *mapfile = getenv("TWILL_MAP");
  if (mapfile != NULL && *mapfile != '\0') {
    long *raw = (long *)malloc((size_t)P * sizeof(long));
    if (raw == NULL) { free(grp); return MPI_ERR_NO_MEM; }
    FILE *f = fopen(mapfile, "r");
    int ok = (f != NULL);
    if (ok) {
      for (int i = 0; i < P; i++) {
        if (fscanf(f, "%ld", &raw[i]) != 1) { ok = 0; break; }
      }
      fclose(f);
    }
    if (ok) {
      int rc = twill_densify(raw, P, grp, G_out);
      free(raw);
      if (rc != 0) { free(grp); return MPI_ERR_NO_MEM; }
      *grp_out = grp;
      return MPI_SUCCESS;
    }
    if (rank == 0) {
      fprintf(stderr, "TWILL: TWILL_MAP='%s' unreadable or has fewer than %d "
                      "entries; falling back to node=group.\n", mapfile, P);
    }
    free(raw);
    /* fall through to TWILL_GROUP / node derivation */
  }

  /* TWILL_GROUP=<N>: synthetic uniform contiguous groups of size N (rank i -> i/N),
   * independent of placement. "node" (or unset) falls through to node=group. */
  {
    const char *grpenv = getenv("TWILL_GROUP");
    if (grpenv != NULL && *grpenv != '\0' && strcmp(grpenv, "node") != 0) {
      char *end = NULL;
      long n = strtol(grpenv, &end, 10);
      if (end != grpenv && *end == '\0' && n >= 1) {
        int gsize = (n > (long)P) ? P : (int)n;        /* N >= P -> a single group */
        for (int i = 0; i < P; i++) { grp[i] = i / gsize; }
        *G_out = (P + gsize - 1) / gsize;
        if (rank == 0) {
          BINE_DEBUG_PRINT("TWILL: TWILL_GROUP=%ld -> %d synthetic group(s) of size %d\n",
                           n, *G_out, gsize);
        }
        *grp_out = grp;
        return MPI_SUCCESS;
      }
      if (rank == 0) {
        fprintf(stderr, "TWILL: ignoring TWILL_GROUP='%s' (expected a positive integer "
                        "group size or 'node'); using node=group.\n", grpenv);
      }
      /* fall through to node = group */
    }
  }

  /* node = group (default) */
  {
    MPI_Comm shmcomm;
    int err = MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank,
                                  MPI_INFO_NULL, &shmcomm);
    if (err == MPI_SUCCESS) {
      int leader_global = rank;          /* each rank's own global rank      */
      err = MPI_Bcast(&leader_global, 1, MPI_INT, 0, shmcomm); /* -> node id */
      MPI_Comm_free(&shmcomm);
      if (err == MPI_SUCCESS) {
        int  *tmp = (int  *)malloc((size_t)P * sizeof(int));
        long *raw = (long *)malloc((size_t)P * sizeof(long));
        if (tmp == NULL || raw == NULL) { free(tmp); free(raw); free(grp); return MPI_ERR_NO_MEM; }
        err = MPI_Allgather(&leader_global, 1, MPI_INT, tmp, 1, MPI_INT, comm);
        if (err == MPI_SUCCESS) {
          for (int i = 0; i < P; i++) { raw[i] = tmp[i]; }
          int rc = twill_densify(raw, P, grp, G_out);
          free(tmp); free(raw);
          if (rc != 0) { free(grp); return MPI_ERR_NO_MEM; }
          *grp_out = grp;
          return MPI_SUCCESS;
        }
        free(tmp); free(raw);
      }
    }
    /* fall through to single group on any MPI failure */
  }

  /* single group */
  for (int i = 0; i < P; i++) { grp[i] = 0; }
  *G_out = 1;
  if (rank == 0 && !g_twill_warned_single) {
    fprintf(stderr, "TWILL: no usable group map; using a single group "
                    "(twill_group degenerates to a plain shift).\n");
    g_twill_warned_single = 1;
  }
  *grp_out = grp;
  return MPI_SUCCESS;
}

/* ---------------------------------------------------------------------------
 * Cross-rank consistency guard (once per context): all ranks must derive the
 * same (kind, window, seed, map). Abort the collective with a clear message on
 * mismatch rather than deadlocking later on divergent schedules.
 * ------------------------------------------------------------------------- */
static inline uint64_t twill_hash_u64(uint64_t h, uint64_t x) {
  h ^= x; h *= 1099511628211ULL; return h;  /* FNV-1a style mix */
}

static int twill_consistency_check(MPI_Comm comm, int rank, int P,
                                   twill_rho_kind_t kind) {
  uint64_t h = 1469598103934665603ULL;
  h = twill_hash_u64(h, (uint64_t)kind);
  h = twill_hash_u64(h, (uint64_t)g_twill_ctx.window);
  h = twill_hash_u64(h, g_twill_ctx.seed);
  if (kind == TWILL_RHO_GROUP && g_twill_ctx.map_ready) {
    h = twill_hash_u64(h, (uint64_t)g_twill_ctx.G);
    for (int i = 0; i < P; i++) { h = twill_hash_u64(h, (uint64_t)g_twill_ctx.grp[i]); }
  }

  uint64_t root_h = h;
  int err = MPI_Bcast(&root_h, 1, MPI_UINT64_T, 0, comm);
  if (err != MPI_SUCCESS) { return err; }
  int agree = (h == root_h) ? 1 : 0, all_agree = 0;
  err = MPI_Allreduce(&agree, &all_agree, 1, MPI_INT, MPI_MIN, comm);
  if (err != MPI_SUCCESS) { return err; }
  if (!all_agree) {
    if (rank == 0) {
      fprintf(stderr, "TWILL: inconsistent schedule inputs (map/seed/window) "
                      "across ranks; aborting collective.\n");
    }
    return MPI_ERR_OTHER;
  }
  return MPI_SUCCESS;
}

/* Grow the engine scratch arrays to hold at least ntotal outstanding ops. */
static int twill_ensure_bufs(twill_ctx_t *c, int ntotal) {
  if (c->ntotal_cap >= ntotal) { return MPI_SUCCESS; }
  MPI_Request *r = (MPI_Request *)realloc(c->reqs,      (size_t)ntotal * sizeof(MPI_Request));
  int         *k = (int         *)realloc(c->slot_kind, (size_t)ntotal * sizeof(int));
  int         *d = (int         *)realloc(c->done_idx,  (size_t)ntotal * sizeof(int));
  MPI_Status  *s = (MPI_Status  *)realloc(c->statuses,  (size_t)ntotal * sizeof(MPI_Status));
  if (r != NULL) { c->reqs = r; }
  if (k != NULL) { c->slot_kind = k; }
  if (d != NULL) { c->done_idx = d; }
  if (s != NULL) { c->statuses = s; }
  if (r == NULL || k == NULL || d == NULL || s == NULL) { return MPI_ERR_NO_MEM; }
  c->ntotal_cap = ntotal;
  return MPI_SUCCESS;
}

/* ---------------------------------------------------------------------------
 * Ensure the cached context is valid for (comm, P) and that rho for the
 * requested variant is built. Rebuilds everything on a comm/size change.
 * Map discovery and the consistency check run at most once per context build
 * (both are collective); they are NOT repeated under TWILL_CACHE=0.
 * ------------------------------------------------------------------------- */
static int twill_ensure_ctx(MPI_Comm comm, int rank, int P, twill_rho_kind_t kind) {
  if (!g_twill_atexit_registered) { atexit(twill_atexit); g_twill_atexit_registered = 1; }

  if (!g_twill_ctx.valid || g_twill_ctx.P != P || g_twill_ctx.comm != comm) {
    twill_free_all(&g_twill_ctx);
    g_twill_ctx.valid  = 1;
    g_twill_ctx.P      = P;
    g_twill_ctx.comm   = comm;
    g_twill_ctx.window = twill_env_int("TWILL_WINDOW", 32);
    if (g_twill_ctx.window < 1) { g_twill_ctx.window = 1; }
    g_twill_ctx.cache  = twill_env_int("TWILL_CACHE", 1);
    g_twill_ctx.seed   = twill_env_u64("TWILL_SEED", TWILL_DEFAULT_SEED);
    g_twill_ctx.skew_us = twill_env_int("TWILL_SKEW_US", 0);
    if (g_twill_ctx.skew_us < 0) { g_twill_ctx.skew_us = 0; }
    g_twill_ctx.calls  = 0;
  }

  int err;
  if (kind == TWILL_RHO_GROUP && !g_twill_ctx.map_ready) {
    err = twill_discover_map(comm, rank, P, &g_twill_ctx.grp, &g_twill_ctx.G);
    if (err != MPI_SUCCESS) { return err; }
    g_twill_ctx.map_ready = 1;
  }

  if (!g_twill_ctx.checked) {
    err = twill_consistency_check(comm, rank, P, kind);
    if (err != MPI_SUCCESS) { return err; }
    g_twill_ctx.checked = 1;
  }

  if (g_twill_ctx.rho[kind] == NULL) {
    g_twill_ctx.rho[kind]     = (int *)malloc((size_t)P * sizeof(int));
    g_twill_ctx.rho_inv[kind] = (int *)malloc((size_t)P * sizeof(int));
    if (g_twill_ctx.rho[kind] == NULL || g_twill_ctx.rho_inv[kind] == NULL) { return MPI_ERR_NO_MEM; }
    if (twill_build_rho(kind, g_twill_ctx.grp, P, g_twill_ctx.G, g_twill_ctx.seed,
                        g_twill_ctx.rho[kind], g_twill_ctx.rho_inv[kind]) != 0) {
      return MPI_ERR_NO_MEM;
    }
  }
  return MPI_SUCCESS;
}

/* ---------------------------------------------------------------------------
 * Shared windowed nonblocking engine. No barriers, no per-step synchronization:
 * the schedule is only an ordering consumed by a bounded window of outstanding
 * Isend/Irecv. Receives land directly at rbuf[sigma(r,t)] (analytic invert).
 * ------------------------------------------------------------------------- */
static int twill_run(twill_rho_kind_t kind,
                     const void *sbuf, size_t scount, MPI_Datatype sdtype,
                     void *rbuf, size_t rcount, MPI_Datatype rdtype, MPI_Comm comm) {
  int rank = -1, P = -1, err = MPI_SUCCESS, line = -1;
  ptrdiff_t lb, sext, rext;

  if (sbuf == MPI_IN_PLACE) { err = MPI_ERR_ARG; line = __LINE__; goto err_hndl; }

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &P);

  if (scount > (size_t)INT_MAX || rcount > (size_t)INT_MAX) {
    err = MPI_ERR_COUNT; line = __LINE__; goto err_hndl;
  }

  err = MPI_Type_get_extent(sdtype, &lb, &sext);
  if (err != MPI_SUCCESS) { line = __LINE__; goto err_hndl; }
  err = MPI_Type_get_extent(rdtype, &lb, &rext);
  if (err != MPI_SUCCESS) { line = __LINE__; goto err_hndl; }

  /* ---- setup (cached, amortized over PICO's many iterations) ---- */
  PICO_TAG_BEGIN("twill_setup");
  err = twill_ensure_ctx(comm, rank, P, kind);
  PICO_TAG_END("twill_setup");
  if (err != MPI_SUCCESS) { line = __LINE__; goto err_hndl; }

  const int *rho  = g_twill_ctx.rho[kind];
  const int *rinv = g_twill_ctx.rho_inv[kind];

  /* ---- self block (t = 0): local copy, no MPI ---- */
  PICO_TAG_BEGIN("twill_self");
  err = copy_buffer_different_dt(
            (const char *)sbuf + (ptrdiff_t)rank * sext * (ptrdiff_t)scount, scount, sdtype,
            (char *)rbuf       + (ptrdiff_t)rank * rext * (ptrdiff_t)rcount, rcount, rdtype);
  PICO_TAG_END("twill_self");
  if (err != MPI_SUCCESS) { line = __LINE__; goto err_hndl; }

  /* Optional skew injection (off by default): a per-rank, per-call random sleep
   * before the exchange, to probe skew tolerance. Outside the twill_exchange
   * tag (counted in total iter time, not in the exchange breakdown). */
  if (g_twill_ctx.skew_us > 0) {
    uint64_t s = g_twill_ctx.seed ^ ((uint64_t)rank << 32) ^ g_twill_ctx.calls;
    useconds_t us = (useconds_t)(twill_splitmix64(&s) % (uint64_t)g_twill_ctx.skew_us);
    usleep(us);
  }
  g_twill_ctx.calls++;

  /* ---- windowed exchange over steps t = 1 .. P-1 ---- */
  int steps = P - 1;
  if (steps > 0) {
    int win   = g_twill_ctx.window;
    int nsend = (win >= steps) ? steps : win;             /* min(W,   P-1) */
    long winr = (long)win + 8;                            /* recv runs ahead */
    int nrecv = (winr >= (long)steps) ? steps : (int)winr;/* min(W+8, P-1) */
    int ntotal = nrecv + nsend;

    err = twill_ensure_bufs(&g_twill_ctx, ntotal);
    if (err != MPI_SUCCESS) { line = __LINE__; goto err_hndl; }

    MPI_Request *reqs = g_twill_ctx.reqs;
    int        *skind = g_twill_ctx.slot_kind;
    int         *idxs = g_twill_ctx.done_idx;
    MPI_Status   *sts = g_twill_ctx.statuses;

    int send_next = 1, recv_next = 1, sdone = 0, rdone = 0;

    PICO_TAG_BEGIN("twill_exchange");

    /* recv window slightly ahead of the send window */
    for (int i = 0; i < nrecv && err == MPI_SUCCESS; i++) {
      int t = recv_next++;
      int s = twill_sigma(rho, rinv, P, rank, t);
      skind[i] = 0;
      err = MPI_Irecv((char *)rbuf + (ptrdiff_t)s * rext * (ptrdiff_t)rcount,
                      (int)rcount, rdtype, s, TWILL_TAG, comm, &reqs[i]);
    }
    for (int i = 0; i < nsend && err == MPI_SUCCESS; i++) {
      int slot = nrecv + i;
      int t = send_next++;
      int d = twill_pi(rho, rinv, P, rank, t);
      skind[slot] = 1;
      err = MPI_Isend((const char *)sbuf + (ptrdiff_t)d * sext * (ptrdiff_t)scount,
                      (int)scount, sdtype, d, TWILL_TAG, comm, &reqs[slot]);
    }

    while (err == MPI_SUCCESS && (sdone < steps || rdone < steps)) {
      int outcount = 0;
      err = MPI_Waitsome(ntotal, reqs, &outcount, idxs, sts);
      if (err != MPI_SUCCESS || outcount == MPI_UNDEFINED) { break; }

      for (int c = 0; c < outcount && err == MPI_SUCCESS; c++) {
        int slot = idxs[c];
        if (skind[slot] == 0) {                 /* a receive completed */
          rdone++;
          if (recv_next <= steps) {
            int t = recv_next++;
            int s = twill_sigma(rho, rinv, P, rank, t);
            err = MPI_Irecv((char *)rbuf + (ptrdiff_t)s * rext * (ptrdiff_t)rcount,
                            (int)rcount, rdtype, s, TWILL_TAG, comm, &reqs[slot]);
          } else {
            reqs[slot] = MPI_REQUEST_NULL;
          }
        } else {                                 /* a send completed */
          sdone++;
          if (send_next <= steps) {
            int t = send_next++;
            int d = twill_pi(rho, rinv, P, rank, t);
            err = MPI_Isend((const char *)sbuf + (ptrdiff_t)d * sext * (ptrdiff_t)scount,
                            (int)scount, sdtype, d, TWILL_TAG, comm, &reqs[slot]);
          } else {
            reqs[slot] = MPI_REQUEST_NULL;
          }
        }
      }
    }

    PICO_TAG_END("twill_exchange");
    if (err != MPI_SUCCESS) { line = __LINE__; goto err_hndl; }
  }

  /* Per-call schedule teardown so its cost shows up in every timed iteration. */
  if (g_twill_ctx.cache == 0) { twill_free_compute(&g_twill_ctx); }

  return MPI_SUCCESS;

 err_hndl:
  fprintf(stderr, "\n%s:%4d\tRank %d TWILL error %d\n\n", __FILE__, line, rank, err);
  (void)line;
  return err;
}

/* ---------------------------------------------------------------------------
 * PICO entry points: thin wrappers selecting rho and calling the shared engine.
 * ------------------------------------------------------------------------- */
int alltoall_twill_group(const void *sbuf, size_t scount, MPI_Datatype sdtype,
                         void *rbuf, size_t rcount, MPI_Datatype rdtype, MPI_Comm comm) {
  return twill_run(TWILL_RHO_GROUP, sbuf, scount, sdtype, rbuf, rcount, rdtype, comm);
}

int alltoall_twill_random(const void *sbuf, size_t scount, MPI_Datatype sdtype,
                          void *rbuf, size_t rcount, MPI_Datatype rdtype, MPI_Comm comm) {
  return twill_run(TWILL_RHO_RANDOM, sbuf, scount, sdtype, rbuf, rcount, rdtype, comm);
}

int alltoall_twill_shift(const void *sbuf, size_t scount, MPI_Datatype sdtype,
                         void *rbuf, size_t rcount, MPI_Datatype rdtype, MPI_Comm comm) {
  return twill_run(TWILL_RHO_SHIFT, sbuf, scount, sdtype, rbuf, rcount, rdtype, comm);
}
