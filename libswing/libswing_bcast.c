#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>

#include "libswing.h"
#include "libswing_utils.h"
#include "libswing_utils_bitmaps.h"

/*
 * NOTE: Taken from Open MPI base module and rewritten using MPI API for benchmarking
 * reasons.
 *
 * ompi_coll_base_bcast_intra_scatter_allgather
 *
 * Function:  Bcast using a binomial tree scatter followed by a recursive
 *            doubling allgather.
 * Accepts:   Same arguments as MPI_Bcast
 * Returns:   MPI_SUCCESS or error code
 *
 * Limitations: count >= comm_size
 * Time complexity: O(\alpha\log(p) + \beta*m((p-1)/p))
 *   Binomial tree scatter: \alpha\log(p) + \beta*m((p-1)/p)
 *   Recursive doubling allgather: \alpha\log(p) + \beta*m((p-1)/p)
 *
 * Example, p=8, count=8, root=0
 *    Binomial tree scatter      Recursive doubling allgather
 * 0: --+  --+  --+  [0*******]  <-+ [01******]  <--+   [0123****] <--+
 * 1:   |   2|  <-+  [*1******]  <-+ [01******]  <--|-+ [0123****] <--+-+
 * 2:  4|  <-+  --+  [**2*****]  <-+ [**23****]  <--+ | [0123****] <--+-+-+
 * 3:   |       <-+  [***3****]  <-+ [**23****]  <----+ [0123****] <--+-+-+-+
 * 4: <-+  --+  --+  [****4***]  <-+ [****45**]  <--+   [****4567] <--+ | | |
 * 5:       2|  <-+  [*****5**]  <-+ [****45**]  <--|-+ [****4567] <----+ | |
 * 6:      <-+  --+  [******6*]  <-+ [******67]  <--+ | [****4567] <------+ |
 * 7:           <-+  [*******7]  <-+ [******67]  <--|-+ [****4567] <--------+
 */
int bcast_scatter_allgather(void *buf, size_t count, MPI_Datatype dtype, int root, MPI_Comm comm)
{
  int rank, comm_size, err = MPI_SUCCESS, dtype_size_int;
  ptrdiff_t lb, extent;
  MPI_Status status;
  MPI_Type_get_extent(dtype, &lb, &extent);
  MPI_Type_size(dtype, &dtype_size_int);
  size_t dtype_size = (size_t)dtype_size_int;

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &comm_size);

  if (comm_size < 2 || dtype_size == 0)
    return MPI_SUCCESS;

  if (count < (size_t)comm_size) {
    if (rank == 0) {
      SWING_DEBUG_PRINT("Error: count < comm_size");
    }
    return MPI_ERR_COUNT;
  }

  int vrank = (rank - root + comm_size) % comm_size;
  size_t recv_count = 0, send_count = 0;
  size_t scatter_count = (count + comm_size - 1) / comm_size; /* ceil(count / comm_size) */
  size_t curr_count = (rank == root) ? count : 0;
  int tmp_count; // to silence compiler warning for MPI_Get_count

  /* Scatter by binomial tree: receive data from parent */
  int mask = 0x1;
  while (mask < comm_size) {
    if (vrank & mask) {
      int parent = (rank - mask + comm_size) % comm_size;
      /* Compute an upper bound on recv block size */
      recv_count = count - vrank * scatter_count;
      if (recv_count <= 0) {
        curr_count = 0;
      } else {
        /* Recv data from parent */
        err = MPI_Recv((char *)buf + (ptrdiff_t)vrank * scatter_count * extent,
                    recv_count, dtype, parent, 0, comm, &status);
        if (MPI_SUCCESS != err) { goto cleanup_and_return; }
        /* Get received count */
        MPI_Get_count(&status, dtype, &tmp_count);
        curr_count = (size_t) tmp_count;
      }
      break;
    }
    mask <<= 1;
  }

  /* Scatter by binomial tree: send data to child processes */
  mask >>= 1;
  while (mask > 0) {
    if (vrank + mask < comm_size) {
      send_count = curr_count - scatter_count * mask;
      if (send_count > 0) {
        int child = (rank + mask) % comm_size;
        err = MPI_Send((char *)buf + (ptrdiff_t)scatter_count * (vrank + mask) * extent,
                    send_count, dtype, child, 0, comm);
        if (MPI_SUCCESS != err) { goto cleanup_and_return; }
        curr_count -= send_count;
      }
    }
    mask >>= 1;
  }

  /*
   * Allgather by recursive doubling
   * Each process has the curr_count elems in the buf[vrank * scatter_count, ...]
   */
  size_t rem_count = count - vrank * scatter_count;
  curr_count = (scatter_count < rem_count) ? scatter_count : rem_count;
  if (curr_count < 0)
    curr_count = 0;

  mask = 0x1;
  while (mask < comm_size) {
    int vremote = vrank ^ mask;
    int remote = (vremote + root) % comm_size;

    int vrank_tree_root = rounddown(vrank, mask);
    int vremote_tree_root = rounddown(vremote, mask);

    if (vremote < comm_size) {
      ptrdiff_t send_offset = vrank_tree_root * scatter_count * extent;
      ptrdiff_t recv_offset = vremote_tree_root * scatter_count * extent;
      recv_count = count - vremote_tree_root * scatter_count;
      if (recv_count < 0)
        recv_count = 0;
      err = MPI_Sendrecv((char *)buf + send_offset, curr_count, dtype, remote, 0,
                         (char *)buf + recv_offset, recv_count, dtype, remote, 0,
                          comm, &status);
      if (MPI_SUCCESS != err) { goto cleanup_and_return; }
      MPI_Get_count(&status, dtype, &tmp_count);
      recv_count = (size_t) tmp_count;
      curr_count += recv_count;
    }

    /*
     * Non-power-of-two case: if process did not have destination process
     * to communicate with, we need to send him the current result.
     * Recursive halving algorithm is used for search of process.
     */
    if (vremote_tree_root + mask > comm_size) {
      int nprocs_alldata = comm_size - vrank_tree_root - mask;
      ptrdiff_t offset = scatter_count * (vrank_tree_root + mask);
      for (int rhalving_mask = mask >> 1; rhalving_mask > 0; rhalving_mask >>= 1) {
        vremote = vrank ^ rhalving_mask;
        remote = (vremote + root) % comm_size;
        int tree_root = rounddown(vrank, rhalving_mask << 1);
        /*
         * Send only if:
         * 1) current process has data: (vremote > vrank) && (vrank < tree_root + nprocs_alldata)
         * 2) remote process does not have data at any step: vremote >= tree_root + nprocs_alldata
         */
        if ((vremote > vrank) && (vrank < tree_root + nprocs_alldata)
          && (vremote >= tree_root + nprocs_alldata)) {
          err = MPI_Send((char *)buf + (ptrdiff_t)offset * extent,
                      recv_count, dtype, remote, 0, comm);
          if (MPI_SUCCESS != err) { goto cleanup_and_return; }

        } else if ((vremote < vrank) && (vremote < tree_root + nprocs_alldata)
               && (vrank >= tree_root + nprocs_alldata)) {
          err = MPI_Recv((char *)buf + (ptrdiff_t)offset * extent,
                      count, dtype, remote, 0, comm, &status);
          if (MPI_SUCCESS != err) { goto cleanup_and_return; }
          MPI_Get_count(&status, dtype, &tmp_count);
          recv_count = (size_t) tmp_count;
          curr_count += recv_count;
        }
      }
    }
    mask <<= 1;
  }

cleanup_and_return:
  return err;
}

/*
 * @brief bcast_swing_lat: broadcast buf from root to all processes using 
 * a binomial tree communication pattern with swing `pi` peer selection.
 *
 * For now only works with comm_sz = 2^k and root = 0, but logic will be
 * extended to work with any root.
 */
int bcast_swing_lat(void *buf, size_t count, MPI_Datatype dtype, int root, MPI_Comm comm)
{
  int size, rank, steps, recv_step = -1, line, err = MPI_SUCCESS;
  char *received = NULL;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  // Check if the number of processes is a power of 2
  steps = log_2(size);
  if (size != (1 << steps)) {
    line = __LINE__;
    err = MPI_ERR_SIZE;
    goto cleanup_and_return;
  }
  // Only root = 0 logic is done
  if (root != 0){
    line = __LINE__;
    err = MPI_ERR_ROOT;
    goto cleanup_and_return;
  }

  // TODO: CHANGE THIS
  // Use an auxiliary array to record visited node in order
  // to calculate at which step node is gonna receive the message.
  received = calloc(size, sizeof(char));
  if (received == NULL) {
    line = __LINE__;
    err = MPI_ERR_NO_MEM;
    goto cleanup_and_return;
  }
  received[root] = 1;

  for (int step = 0; step < steps && !received[rank]; step++) {
    for (int proc = 0; proc < size; proc++) {
      if (!received[proc]) continue;

      int dest = pi(proc, step, size);
      received[dest] = 1;
      if (dest == rank) {
        recv_step = step;
        break;
      }
    }
  }

  /* Main loop.
   *
   * At each step s:
   * - if rank r has the data it sends it to dest = pi(r, s)
   * - if rank r does not have the data:
   *   - if recv_step ==s, it receives the data from the parent
   *   - otherwise it does nothing in this iteration
   */
  for (int s = 0; s < steps; s++) {
    int dest;
    // If I don't have the data and I am scheduled to receive it, wait for it.
    if (rank != root && recv_step == s) {
      dest = pi(rank, s, size);
      err = MPI_Recv(buf, count, dtype, dest, s, comm, MPI_STATUS_IGNORE);
      if (MPI_SUCCESS != err) { line = __LINE__; goto cleanup_and_return; }
      continue;
    }

    // If I already have the message, send the data.
    if (recv_step < s) {
      dest = pi(rank, s, size);
      err = MPI_Send(buf, count, dtype, dest, s, comm);
      if (MPI_SUCCESS != err) { line = __LINE__; goto cleanup_and_return; }
      continue;
    }
  }

  free(received);

  return MPI_SUCCESS;

cleanup_and_return:
  SWING_DEBUG_PRINT("\n%s:%4d\tRank %d Error occurred %d\n\n", __FILE__, line, rank, err);
  (void)line;  // silence compiler warning
  if (NULL!= received)     free(received);

  return err;
}

int bcast_swing_lat_reversed(void *buf, size_t count, MPI_Datatype dtype, int root, MPI_Comm comm)
{
  int size, rank, steps, recv_step = -1, line, err = MPI_SUCCESS;
  char *received = NULL;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  // Check if the number of processes is a power of 2
  steps = log_2(size);
  if (size != (1 << steps)) {
    line = __LINE__;
    err = MPI_ERR_SIZE;
    goto cleanup_and_return;
  }
  // Only root = 0 logic is done
  if (root != 0){
    line = __LINE__;
    err = MPI_ERR_ROOT;
    goto cleanup_and_return;
  }

  // TODO: CHANGE THIS
  // Use an auxiliary array to record visited node in order
  // to calculate at which step node is gonna receive the message.
  received = calloc(size, sizeof(char));
  if (received == NULL) {
    line = __LINE__;
    err = MPI_ERR_NO_MEM;
    goto cleanup_and_return;
  }
  received[root] = 1;

  for (int step = 0; step < steps && !received[rank]; step++) {
    for (int proc = 0; proc < size; proc++) {
      if (!received[proc]) continue;

      int dest = pi(proc, steps - step - 1, size);
      received[dest] = 1;
      if (dest == rank) {
        recv_step = step;
        break;
      }
    }
  }

  /* Main loop.
   *
   * At each step s:
   * - if rank r has the data it sends it to dest = pi(r, s)
   * - if rank r does not have the data:
   *   - if recv_step ==s, it receives the data from the parent
   *   - otherwise it does nothing in this iteration
   */
  for (int s = 0; s < steps; s++) {
    int dest;
    // If I don't have the data and I am scheduled to receive it, wait for it.
    if (rank != root && recv_step == s) {
      dest = pi(rank, steps - s - 1, size);
      err = MPI_Recv(buf, count, dtype, dest, s, comm, MPI_STATUS_IGNORE);
      if (MPI_SUCCESS != err) { line = __LINE__; goto cleanup_and_return; }
      continue;
    }

    // If I already have the message, send the data.
    if (recv_step < s) {
      dest = pi(rank, steps - s - 1, size);
      err = MPI_Send(buf, count, dtype, dest, s, comm);
      if (MPI_SUCCESS != err) { line = __LINE__; goto cleanup_and_return; }
      continue;
    }
  }

  free(received);

  return MPI_SUCCESS;

cleanup_and_return:
  SWING_DEBUG_PRINT("\n%s:%4d\tRank %d Error occurred %d\n\n", __FILE__, line, rank, err);
  (void)line;  // silence compiler warning
  if (NULL!= received)     free(received);

  return err;
}

 /*
 * @brief bcast_swing_bdw_static: it's composed by a scatter and allgather phases.
 * Both phases utilizes swing communication pattern. The scatter phase is done using
 * a binomial tree scatter and the allgather phase is done using a recursive doubling.
 *
 * For now only works with size = 2^k,<=512, root = 0, and count <= comm_sz.
 * Logic will be extended to work with any root.
 */
int bcast_swing_bdw_static(void *buf, size_t count, MPI_Datatype dtype, int root, MPI_Comm comm)
{
  int size, rank, step, steps, recv_step = -1;
  int dtype_size, line, err = MPI_SUCCESS;
  int dest, rdest, sdest, w_size, split_rank;
  size_t small_blocks, big_blocks, r_count = 0, s_count = 0;
  ptrdiff_t r_offset = 0, s_offset = 0, lb, extent;
  const int *s_bitmap, *r_bitmap;
  char *received = NULL;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  // Get datatype informations
  MPI_Type_get_extent(dtype, &lb, &extent);
  MPI_Type_size(dtype, &dtype_size);

  // Trivial case
  if (size < 2 || dtype_size == 0)
    return MPI_SUCCESS;
  
  // Check if count is less than the number of processes
  if (count < (size_t)size) {
    line = __LINE__;
    err = MPI_ERR_COUNT;
    goto cleanup_and_return;
  }

  // Check if the number of processes is a power of 2
  steps = log_2(size);
  if (size != (1 << steps)) {
    line = __LINE__;
    err = MPI_ERR_SIZE;
    goto cleanup_and_return;
  }

  // Only root = 0 logic is done
  if (root != 0){
    line = __LINE__;
    err = MPI_ERR_ROOT;
    goto cleanup_and_return;
  }

  // Use an auxiliary array to record visited node in order
  // to calculate at which step node is gonna receive the message.
  received = calloc(size, sizeof(char));
  if (received == NULL) {
    line = __LINE__;
    err = MPI_ERR_NO_MEM;
    goto cleanup_and_return;
  }
  received[root] = 1;
  for (step = 0; step < steps && !received[rank]; step++) {
    for (int proc = 0; proc < size; proc++) {
      if (!received[proc]) continue;

      dest = pi(proc, step, size);
      received[dest] = 1;
      if (dest == rank) {
        recv_step = step;
        break;
      }
    }
  }
  
  // Compute the size of each block, dividing in small and big blocks
  // where:
  // - small_blocks = count / size
  // - big_blocks = count / size + count % size
  // - split_rank = count % size
  COLL_BASE_COMPUTE_BLOCKCOUNT(count, size, split_rank,
                               big_blocks, small_blocks);

  
  // Gets rearranged contiguous bitmaps from "libswing_utils_bitmaps.c"
  if (get_static_bitmap(&s_bitmap, &r_bitmap, steps, size, rank) == -1){
    line = __LINE__;
    err = MPI_ERR_UNKNOWN;
    goto cleanup_and_return;
  }

  /* Scatter phase.
   *
   * At each step s:
   * - if rank r has the data it sends `w_size` blocks to dest = pi(r, s)
   * - if rank r does not have the data:
   *   - if recv_step ==s, it receives the data from the parent
   *   - otherwise it does nothing in this iteration
   */
  w_size = size >> 1;
  for (step = 0; step < steps; step++) {
    // If I don't have the data and I am scheduled to receive it, wait for it.
    if (rank != root && recv_step == step) {
      dest = pi(rank, step, size);
      r_count = (r_bitmap[step] + w_size <= split_rank) ?
                  (size_t)w_size * big_blocks :
                    (r_bitmap[step] >= split_rank) ?
                      (size_t)w_size * small_blocks :
                      (size_t)w_size * small_blocks + (size_t)(split_rank - r_bitmap[step]);
      r_offset = (r_bitmap[step] <= split_rank) ?
                  (ptrdiff_t) r_bitmap[step] * (ptrdiff_t)(big_blocks * extent) :
                  (ptrdiff_t)(r_bitmap[step] * (int) small_blocks + split_rank) * (ptrdiff_t) extent;
      err = MPI_Recv((char*)buf + r_offset, r_count, dtype, dest, 0, comm, MPI_STATUS_IGNORE);
      if (MPI_SUCCESS != err) { line = __LINE__; goto cleanup_and_return; }
    }

    // If I already have the message, send the data.
    if (recv_step < step) {
      dest = pi(rank, step, size);
      s_count = (s_bitmap[step] + w_size <= split_rank) ?
                  (size_t)w_size * big_blocks :
                    (s_bitmap[step] >= split_rank) ?
                      (size_t)w_size * small_blocks :
                      (size_t)w_size * small_blocks + (size_t)(split_rank - s_bitmap[step]);
      s_offset = (s_bitmap[step] <= split_rank) ?
                  (ptrdiff_t) s_bitmap[step] * (ptrdiff_t)(big_blocks * extent) :
                  (ptrdiff_t)(s_bitmap[step] * (int) small_blocks + split_rank) * (ptrdiff_t) extent;
      err = MPI_Send((char *)buf + s_offset, s_count, dtype, dest, 0, comm);
      if (MPI_SUCCESS != err) { line = __LINE__; goto cleanup_and_return; }
    }
    w_size >>= 1;
  }

  /* Allgather phase.
   *
   * At each step s, each rank r:
   * - if `dest=pi(r, s)` has not the blocks that r has, it sends `w_size` blocks to dest,
   * - if r does not have the blocks that `dest=pi(r, s)` has, it recv `w_size` blocks from dest.
   *
   * To know if r has the data of dest and viceversa, recv_step is used:
   * - if recv_step == s, r has the data of dest
   * - if recv_step < s, r does not have the data of dest
   * in all other cases r and dest do a complete sendrecv of `w_size` blocks.
   */
  w_size = 0x1;
  for(step = steps - 1; step >= 0; step--) {
    dest = pi(rank, step, size);
    rdest = (recv_step < step) ? MPI_PROC_NULL : dest;
    sdest = (recv_step == step) ? MPI_PROC_NULL : dest;

    r_count = (s_bitmap[step] + w_size <= split_rank) ?
                (size_t)w_size * big_blocks :
                  (s_bitmap[step] >= split_rank) ?
                    (size_t)w_size * small_blocks :
                    (size_t)w_size * small_blocks + (size_t)(split_rank - s_bitmap[step]);
    r_offset = (s_bitmap[step] <= split_rank) ?
                (ptrdiff_t) s_bitmap[step] * (ptrdiff_t)(big_blocks * extent) :
                (ptrdiff_t)(s_bitmap[step] * (int) small_blocks + split_rank) * (ptrdiff_t) extent;

    s_count = (r_bitmap[step] + w_size <= split_rank) ?
                (size_t)w_size * big_blocks :
                  (r_bitmap[step] >= split_rank) ?
                    (size_t)w_size * small_blocks :
                    (size_t)w_size * small_blocks + (size_t)(split_rank - r_bitmap[step]);
    s_offset = (r_bitmap[step] <= split_rank) ?
                (ptrdiff_t) r_bitmap[step] * (ptrdiff_t)(big_blocks * extent) :
                (ptrdiff_t)(r_bitmap[step] * (int)small_blocks + split_rank) * (ptrdiff_t) extent;
    
    err = MPI_Sendrecv((char *) buf + s_offset, s_count, dtype, sdest, 0,
                 (char *) buf + r_offset, r_count, dtype, rdest, 0,
                 comm, MPI_STATUS_IGNORE);
    if (MPI_SUCCESS != err) { line = __LINE__; goto cleanup_and_return; }
    
    w_size <<= 1;
  }

  free(received);

  return MPI_SUCCESS;

cleanup_and_return:
  SWING_DEBUG_PRINT("\n%s:%4d\tRank %d Error occurred %d\n\n", __FILE__, line, rank, err);
  (void) line; // silence compiler warning
  if (NULL!= received)     free(received);

  return err;
}


// int bcast_swing_bdw_static_reversed(void *buf, size_t count, MPI_Datatype dtype, int root, MPI_Comm comm)
// {
//   int size, rank, step, steps, recv_step = -1;
//   int dtype_size, line, err = MPI_SUCCESS;
//   int dest, rdest, sdest, w_size, split_rank;
//   size_t small_blocks, big_blocks, r_count = 0, s_count = 0;
//   ptrdiff_t r_offset = 0, s_offset = 0, lb, extent;
//   const int *s_bitmap, *r_bitmap;
//   char *received = NULL;
//   MPI_Comm_size(comm, &size);
//   MPI_Comm_rank(comm, &rank);
//
//   // Get datatype informations
//   MPI_Type_get_extent(dtype, &lb, &extent);
//   MPI_Type_size(dtype, &dtype_size);
//
//   // Trivial case
//   if (size < 2 || dtype_size == 0)
//     return MPI_SUCCESS;
//   
//   // Check if count is less than the number of processes
//   if (count < (size_t)size) {
//     line = __LINE__;
//     err = MPI_ERR_COUNT;
//     goto cleanup_and_return;
//   }
//
//   // Check if the number of processes is a power of 2
//   steps = log_2(size);
//   if (size != (1 << steps)) {
//     line = __LINE__;
//     err = MPI_ERR_SIZE;
//     goto cleanup_and_return;
//   }
//
//   // Only root = 0 logic is done
//   if (root != 0){
//     line = __LINE__;
//     err = MPI_ERR_ROOT;
//     goto cleanup_and_return;
//   }
//
//   // Use an auxiliary array to record visited node in order
//   // to calculate at which step node is gonna receive the message.
//   received = calloc(size, sizeof(char));
//   if (received == NULL) {
//     line = __LINE__;
//     err = MPI_ERR_NO_MEM;
//     goto cleanup_and_return;
//   }
//   received[root] = 1;
//   for (step = 0; step < steps && !received[rank]; step++) {
//     for (int proc = 0; proc < size; proc++) {
//       if (!received[proc]) continue;
//
//       dest = pi(proc, steps - step - 1, size);
//       received[dest] = 1;
//       if (dest == rank) {
//         recv_step = step;
//         break;
//       }
//     }
//   }
//   
//   // Compute the size of each block, dividing in small and big blocks
//   // where:
//   // - small_blocks = count / size
//   // - big_blocks = count / size + count % size
//   // - split_rank = count % size
//   COLL_BASE_COMPUTE_BLOCKCOUNT(count, size, split_rank,
//                                big_blocks, small_blocks);
//
//   
//   // Gets rearranged contiguous bitmaps from "libswing_utils_bitmaps.c"
//   if (get_static_bitmap(&s_bitmap, &r_bitmap, steps, size, rank) == -1){
//     line = __LINE__;
//     err = MPI_ERR_UNKNOWN;
//     goto cleanup_and_return;
//   }
//
//   /* Scatter phase.
//    *
//    * At each step s:
//    * - if rank r has the data it sends `w_size` blocks to dest = pi(r, s)
//    * - if rank r does not have the data:
//    *   - if recv_step ==s, it receives the data from the parent
//    *   - otherwise it does nothing in this iteration
//    */
//   w_size = size >> 1;
//   for (step = 0; step < steps; step++) {
//     // If I don't have the data and I am scheduled to receive it, wait for it.
//     if (rank != root && recv_step == step) {
//       dest = pi(rank, steps - step - 1, size);
//       r_count = (r_bitmap[step] + w_size <= split_rank) ?
//                   (size_t)w_size * big_blocks :
//                     (r_bitmap[step] >= split_rank) ?
//                       (size_t)w_size * small_blocks :
//                       (size_t)w_size * small_blocks + (size_t)(split_rank - r_bitmap[step]);
//       r_offset = (r_bitmap[step] <= split_rank) ?
//                   (ptrdiff_t) r_bitmap[step] * (ptrdiff_t)(big_blocks * extent) :
//                   (ptrdiff_t)(r_bitmap[step] * (int) small_blocks + split_rank) * (ptrdiff_t) extent;
//       err = MPI_Recv((char*)buf + r_offset, r_count, dtype, dest, 0, comm, MPI_STATUS_IGNORE);
//       if (MPI_SUCCESS != err) { line = __LINE__; goto cleanup_and_return; }
//     }
//
//     // If I already have the message, send the data.
//     if (recv_step < step) {
//       dest = pi(rank, steps - step - 1, size);
//       s_count = (s_bitmap[step] + w_size <= split_rank) ?
//                   (size_t)w_size * big_blocks :
//                     (s_bitmap[step] >= split_rank) ?
//                       (size_t)w_size * small_blocks :
//                       (size_t)w_size * small_blocks + (size_t)(split_rank - s_bitmap[step]);
//       s_offset = (s_bitmap[step] <= split_rank) ?
//                   (ptrdiff_t) s_bitmap[step] * (ptrdiff_t)(big_blocks * extent) :
//                   (ptrdiff_t)(s_bitmap[step] * (int) small_blocks + split_rank) * (ptrdiff_t) extent;
//       err = MPI_Send((char *)buf + s_offset, s_count, dtype, dest, 0, comm);
//       if (MPI_SUCCESS != err) { line = __LINE__; goto cleanup_and_return; }
//     }
//     w_size >>= 1;
//   }
//
//   /* Allgather phase.
//    *
//    * At each step s, each rank r:
//    * - if `dest=pi(r, s)` has not the blocks that r has, it sends `w_size` blocks to dest,
//    * - if r does not have the blocks that `dest=pi(r, s)` has, it recv `w_size` blocks from dest.
//    *
//    * To know if r has the data of dest and viceversa, recv_step is used:
//    * - if recv_step == s, r has the data of dest
//    * - if recv_step < s, r does not have the data of dest
//    * in all other cases r and dest do a complete sendrecv of `w_size` blocks.
//    */
//   w_size = 0x1;
//   for(step = steps - 1; step >= 0; step--) {
//     dest = pi(rank, steps - step - 1, size);
//     rdest = (recv_step < step) ? MPI_PROC_NULL : dest;
//     sdest = (recv_step == step) ? MPI_PROC_NULL : dest;
//
//     r_count = (s_bitmap[step] + w_size <= split_rank) ?
//                 (size_t)w_size * big_blocks :
//                   (s_bitmap[step] >= split_rank) ?
//                     (size_t)w_size * small_blocks :
//                     (size_t)w_size * small_blocks + (size_t)(split_rank - s_bitmap[step]);
//     r_offset = (s_bitmap[step] <= split_rank) ?
//                 (ptrdiff_t) s_bitmap[step] * (ptrdiff_t)(big_blocks * extent) :
//                 (ptrdiff_t)(s_bitmap[step] * (int) small_blocks + split_rank) * (ptrdiff_t) extent;
//
//     s_count = (r_bitmap[step] + w_size <= split_rank) ?
//                 (size_t)w_size * big_blocks :
//                   (r_bitmap[step] >= split_rank) ?
//                     (size_t)w_size * small_blocks :
//                     (size_t)w_size * small_blocks + (size_t)(split_rank - r_bitmap[step]);
//     s_offset = (r_bitmap[step] <= split_rank) ?
//                 (ptrdiff_t) r_bitmap[step] * (ptrdiff_t)(big_blocks * extent) :
//                 (ptrdiff_t)(r_bitmap[step] * (int)small_blocks + split_rank) * (ptrdiff_t) extent;
//     
//     err = MPI_Sendrecv((char *) buf + s_offset, s_count, dtype, sdest, 0,
//                  (char *) buf + r_offset, r_count, dtype, rdest, 0,
//                  comm, MPI_STATUS_IGNORE);
//     if (MPI_SUCCESS != err) { line = __LINE__; goto cleanup_and_return; }
//     
//     w_size <<= 1;
//   }
//
//   free(received);
//
//   return MPI_SUCCESS;
//
// cleanup_and_return:
//   SWING_DEBUG_PRINT("\n%s:%4d\tRank %d Error occurred %d\n\n", __FILE__, line, rank, err);
//   (void) line; // silence compiler warning
//   if (NULL!= received)     free(received);
//
//   return err;
// }
