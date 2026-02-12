/*
 * Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
 * Licensed under the MIT License
 */

#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>

#include "libpico.h"
#include "libpico_utils.h"


/**
 * Open MPI linear bcast function copied from
 * https://github.com/open-mpi/ompi/blob/3a2e90895e15822912002be1e9aea8032c4c0bae/ompi/mca/coll/base/coll_base_bcast.c#L638
 */
int
bcast_linear(void *buff, size_t count, MPI_Datatype datatype, int root,
             MPI_Comm comm) 
{
    int i, size, rank, err;
    request_manager_t req_manager = {NULL, 0};
    MPI_Request *reqs = NULL;

    MPI_Comm_rank (comm, &rank);
    MPI_Comm_size (comm, &size);

    if (1 == size) return MPI_SUCCESS;

    /* Non-root receive the data. */

    if (rank != root) {
        return MPI_Recv(buff, count, datatype, root, 0, comm, MPI_STATUS_IGNORE);
    }

    /* Root sends data to all others. */
    reqs = alloc_reqs(&req_manager, size - 1);
    if( NULL == reqs ) {
        return MPI_ERR_NO_MEM;
    }

    for (i = 0; i < size; ++i) {
        if (i == rank) {
            continue;
        }

        err = MPI_Isend(buff, count, datatype, i, 0, comm, &reqs[i < rank ? i : i - 1]);
        if (MPI_SUCCESS != err) { goto err_hndl; }
    }
    --i;

    /* Wait for them all.  If there's an error, note that we don't
     * care what the error was -- just that there *was* an error.  The
     * PML will finish all requests, even if one or more of them fail.
     * i.e., by the end of this call, all the requests are free-able.
     * So free them anyway -- even if there was an error. 
     * Note we still need to get the actual error, as collective 
     * operations cannot return MPI_ERR_IN_STATUS.
     */

    err = MPI_Waitall(i, reqs, MPI_STATUSES_IGNORE);
 err_hndl:
    if( NULL != reqs ) {
      cleanup_reqs(&req_manager);
    }
    /* All done */
    return err;
}


/**
 * MPICH recursive halving (binomial) bcast function copied from
 * https://github.com/pmodels/mpich/blob/6e5a2adfeb8a37a89a96bc646e375062c15dc9cd/src/mpi/coll/bcast/bcast_intra_binomial.c
 */
int bcast_binomial_halving(void *buffer, size_t count, MPI_Datatype datatype, int root, MPI_Comm comm_ptr)
{
    int rank, comm_size, src, dst;
    int relative_rank, mask;
    int mpi_errno = MPI_SUCCESS;
    MPI_Aint nbytes = 0, lb;
    MPI_Status *status_p;
    status_p = MPI_STATUS_IGNORE;
    MPI_Aint type_size;

    MPI_Comm_size(comm_ptr, &comm_size);
    MPI_Comm_rank(comm_ptr, &rank);


    MPI_Type_get_extent(datatype, &lb, &type_size);

    nbytes = type_size * count;
    if (nbytes == 0)
        goto fn_exit;   /* nothing to do */



    relative_rank = (rank >= root) ? rank - root : rank - root + comm_size;

    /* Use short message algorithm, namely, binomial tree */

    /* Algorithm:
     * This uses a fairly basic recursive subdivision algorithm.
     * The root sends to the process comm_size/2 away; the receiver becomes
     * a root for a subtree and applies the same process.
     *
     * So that the new root can easily identify the size of its
     * subtree, the (subtree) roots are all powers of two (relative
     * to the root) If m = the first power of 2 such that 2^m >= the
     * size of the communicator, then the subtree at root at 2^(m-k)
     * has size 2^k (with special handling for subtrees that aren't
     * a power of two in size).
     *
     * Do subdivision.  There are two phases:
     * 1. Wait for arrival of data.  Because of the power of two nature
     * of the subtree roots, the source of this message is always the
     * process whose relative rank has the least significant 1 bit CLEARED.
     * That is, process 4 (100) receives from process 0, process 7 (111)
     * from process 6 (110), etc.
     * 2. Forward to my subtree
     *
     * Note that the process that is the tree root is handled automatically
     * by this code, since it has no bits set.  */

    mask = 0x1;
    while (mask < comm_size) {
        if (relative_rank & mask) {
            src = rank - mask;
            if (src < 0)
                src += comm_size;
            mpi_errno = MPI_Recv(buffer, count, datatype, src, 0, comm_ptr, status_p);
            if (mpi_errno != MPI_SUCCESS) {
                goto fn_fail;
            }
            break;
        }
        mask <<= 1;
    }

    /* This process is responsible for all processes that have bits
     * set from the LSB up to (but not including) mask.  Because of
     * the "not including", we start by shifting mask back down one.
     *
     * We can easily change to a different algorithm at any power of two
     * by changing the test (mask > 1) to (mask > block_size)
     *
     * One such version would use non-blocking operations for the last 2-4
     * steps (this also bounds the number of MPI_Requests that would
     * be needed).  */

    mask >>= 1;
    while (mask > 0) {
        if (relative_rank + mask < comm_size) {
            dst = rank + mask;
            if (dst >= comm_size)
                dst -= comm_size;
            mpi_errno = MPI_Send(buffer, count, datatype, dst, 0, comm_ptr);
            if (mpi_errno != MPI_SUCCESS) {
                goto fn_fail;
            }
        }
        mask >>= 1;
    }


  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/**
 * Binomial (recursive doubling) bcast function.
 *
 * Processes propagate the message by doubling the number of informed ranks
 * at each step. On iteration k, ranks with relative_rank in [0, 2^k) send to
 * the partner at +2^k; ranks with relative_rank in [2^k, 2^(k+1)) receive.
 */
int bcast_binomial_doubling(void *buffer, size_t count, MPI_Datatype datatype, int root, MPI_Comm comm_ptr)
{
    int rank, comm_size, src, dst;
    int relative_rank, mask;
    int mpi_errno = MPI_SUCCESS;
    MPI_Aint nbytes = 0, lb;
    MPI_Status *status_p;
    status_p = MPI_STATUS_IGNORE;
    MPI_Aint type_size;

    MPI_Comm_size(comm_ptr, &comm_size);
    MPI_Comm_rank(comm_ptr, &rank);

    MPI_Type_get_extent(datatype, &lb, &type_size);

    nbytes = type_size * count;
    if (nbytes == 0)
        goto fn_exit;   /* nothing to do */

    relative_rank = (rank >= root) ? rank - root : rank - root + comm_size;

    mask = 0x1;
    while (mask < comm_size) {
        if (relative_rank < mask) {
            dst = rank + mask;
            if (dst >= comm_size)
                dst -= comm_size;
            mpi_errno = MPI_Send(buffer, count, datatype, dst, 0, comm_ptr);
            if (mpi_errno != MPI_SUCCESS) {
                goto fn_fail;
            }
        }
        else if (relative_rank < (mask << 1)) {
            src = rank - mask;
            if (src < 0)
                src += comm_size;
            mpi_errno = MPI_Recv(buffer, count, datatype, src, 0, comm_ptr, status_p);
            if (mpi_errno != MPI_SUCCESS) {
                goto fn_fail;
            }
        }
        mask <<= 1;
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

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

  if(comm_size < 2 || dtype_size == 0)
    return MPI_SUCCESS;

  if(count < (size_t)comm_size) {
    if(rank == 0) {
      BINE_DEBUG_PRINT("Error: count < comm_size");
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
    if(vrank & mask) {
      int parent = (rank - mask + comm_size) % comm_size;
      /* Compute an upper bound on recv block size */
      recv_count = count - vrank * scatter_count;
      if(recv_count <= 0) {
        curr_count = 0;
      } else {
        /* Recv data from parent */
        err = MPI_Recv((char *)buf + (ptrdiff_t)vrank * scatter_count * extent,
                    recv_count, dtype, parent, 0, comm, &status);
        if(MPI_SUCCESS != err) { goto cleanup_and_return; }
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
    if(vrank + mask < comm_size) {
      send_count = curr_count - scatter_count * mask;
      if(send_count > 0) {
        int child = (rank + mask) % comm_size;
        err = MPI_Send((char *)buf + (ptrdiff_t)scatter_count * (vrank + mask) * extent,
                    send_count, dtype, child, 0, comm);
        if(MPI_SUCCESS != err) { goto cleanup_and_return; }
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
  if(curr_count < 0)
    curr_count = 0;

  mask = 0x1;
  while (mask < comm_size) {
    int vremote = vrank ^ mask;
    int remote = (vremote + root) % comm_size;

    int vrank_tree_root = rounddown(vrank, mask);
    int vremote_tree_root = rounddown(vremote, mask);

    if(vremote < comm_size) {
      ptrdiff_t send_offset = vrank_tree_root * scatter_count * extent;
      ptrdiff_t recv_offset = vremote_tree_root * scatter_count * extent;
      recv_count = count - vremote_tree_root * scatter_count;
      if(recv_count < 0)
        recv_count = 0;
      err = MPI_Sendrecv((char *)buf + send_offset, curr_count, dtype, remote, 0,
                         (char *)buf + recv_offset, recv_count, dtype, remote, 0,
                          comm, &status);
      if(MPI_SUCCESS != err) { goto cleanup_and_return; }
      MPI_Get_count(&status, dtype, &tmp_count);
      recv_count = (size_t) tmp_count;
      curr_count += recv_count;
    }

    /*
     * Non-power-of-two case: if process did not have destination process
     * to communicate with, we need to send him the current result.
     * Recursive halving algorithm is used for search of process.
     */
    if(vremote_tree_root + mask > comm_size) {
      int nprocs_alldata = comm_size - vrank_tree_root - mask;
      ptrdiff_t offset = scatter_count * (vrank_tree_root + mask);
      for(int rhalving_mask = mask >> 1; rhalving_mask > 0; rhalving_mask >>= 1) {
        vremote = vrank ^ rhalving_mask;
        remote = (vremote + root) % comm_size;
        int tree_root = rounddown(vrank, rhalving_mask << 1);
        /*
         * Send only if:
         * 1) current process has data: (vremote > vrank) && (vrank < tree_root + nprocs_alldata)
         * 2) remote process does not have data at any step: vremote >= tree_root + nprocs_alldata
         */
        if((vremote > vrank) && (vrank < tree_root + nprocs_alldata)
          && (vremote >= tree_root + nprocs_alldata)) {
          err = MPI_Send((char *)buf + (ptrdiff_t)offset * extent,
                      recv_count, dtype, remote, 0, comm);
          if(MPI_SUCCESS != err) { goto cleanup_and_return; }

        } else if((vremote < vrank) && (vremote < tree_root + nprocs_alldata)
               && (vrank >= tree_root + nprocs_alldata)) {
          err = MPI_Recv((char *)buf + (ptrdiff_t)offset * extent,
                      count, dtype, remote, 0, comm, &status);
          if(MPI_SUCCESS != err) { goto cleanup_and_return; }
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
 * @brief bcast_bine_lat: broadcast buf from root to all processes using 
 * a binomial tree communication pattern with bine `pi` peer selection.
 *
 * For now only works with comm_sz = 2^k and root = 0, but logic will be
 * extended to work with any root.
 */
int bcast_bine_lat(void *buf, size_t count, MPI_Datatype dtype, int root, MPI_Comm comm)
{
  int size, rank, steps, recv_step = -1, line, err = MPI_SUCCESS;
  char *received = NULL;
  MPI_Request requests[BINE_MAX_STEPS];
  int request_count = 0;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  // Check if the number of processes is a power of 2
  steps = log_2(size);
  if(size != (1 << steps)) {
    line = __LINE__;
    err = MPI_ERR_SIZE;
    goto cleanup_and_return;
  }
  // Only root = 0 logic is done
  if(root != 0){
    line = __LINE__;
    err = MPI_ERR_ROOT;
    goto cleanup_and_return;
  }

  // TODO: CHANGE THIS
  // Use an auxiliary array to record visited node in order
  // to calculate at which step node is gonna receive the message.
  received = calloc(size, sizeof(char));
  if(received == NULL) {
    line = __LINE__;
    err = MPI_ERR_NO_MEM;
    goto cleanup_and_return;
  }
  received[root] = 1;

  for(int step = 0; step < steps && !received[rank]; step++) {
    for(int proc = 0; proc < size; proc++) {
      if(!received[proc]) continue;

      int dest = pi(proc, step, size);
      received[dest] = 1;
      if(dest == rank) {
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
  for(int s = 0; s < steps; s++) {
    int dest;
    // If I don't have the data and I am scheduled to receive it, wait for it.
    if(rank != root && recv_step == s) {
      dest = pi(rank, s, size);
      err = MPI_Recv(buf, count, dtype, dest, s, comm, MPI_STATUS_IGNORE);
      if(MPI_SUCCESS != err) { line = __LINE__; goto cleanup_and_return; }
      continue;
    }

    // If I already have the message, send the data.
    if(recv_step < s) {
      dest = pi(rank, s, size);
      err = MPI_Isend(buf, count, dtype, dest, s, comm, &requests[request_count]);
      if(MPI_SUCCESS != err) { line = __LINE__; goto cleanup_and_return; }
      request_count++;
      continue;
    }
  }

  if(request_count > 0) {
    err = MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);
    if(MPI_SUCCESS != err) { line = __LINE__; goto cleanup_and_return; }
  }

  free(received);

  return MPI_SUCCESS;

cleanup_and_return:
  BINE_DEBUG_PRINT("\n%s:%4d\tRank %d Error occurred %d\n\n", __FILE__, line, rank, err);
  (void)line;  // silence compiler warning
  if(NULL!= received)     free(received);

  return err;
}

int bcast_bine_lat_reversed(void *buf, size_t count, MPI_Datatype dtype, int root, MPI_Comm comm)
{
  int size, rank, steps, recv_step = -1, line, err = MPI_SUCCESS;
  char *received = NULL;
  MPI_Request requests[BINE_MAX_STEPS];
  int request_count = 0;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  // Check if the number of processes is a power of 2
  steps = log_2(size);
  if(size != (1 << steps)) {
    line = __LINE__;
    err = MPI_ERR_SIZE;
    goto cleanup_and_return;
  }
  // Only root = 0 logic is done
  if(root != 0){
    line = __LINE__;
    err = MPI_ERR_ROOT;
    goto cleanup_and_return;
  }

  // TODO: CHANGE THIS
  // Use an auxiliary array to record visited node in order
  // to calculate at which step node is gonna receive the message.
  received = calloc(size, sizeof(char));
  if(received == NULL) {
    line = __LINE__;
    err = MPI_ERR_NO_MEM;
    goto cleanup_and_return;
  }
  received[root] = 1;

  for(int step = 0; step < steps && !received[rank]; step++) {
    for(int proc = 0; proc < size; proc++) {
      if(!received[proc]) continue;

      int dest = pi(proc, steps - step - 1, size);
      received[dest] = 1;
      if(dest == rank) {
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
  for(int s = 0; s < steps; s++) {
    int dest;
    // If I don't have the data and I am scheduled to receive it, wait for it.
    if(rank != root && recv_step == s) {
      dest = pi(rank, steps - s - 1, size);
      err = MPI_Recv(buf, count, dtype, dest, s, comm, MPI_STATUS_IGNORE);
      if(MPI_SUCCESS != err) { line = __LINE__; goto cleanup_and_return; }
      continue;
    }

    // If I already have the message, send the data.
    if(recv_step < s) {
      dest = pi(rank, steps - s - 1, size);
      err = MPI_Isend(buf, count, dtype, dest, s, comm, &requests[request_count]);
      if(MPI_SUCCESS != err) { line = __LINE__; goto cleanup_and_return; }
      request_count++;
      continue;
    }
  }

  if(request_count > 0) {
    err = MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);
    if(MPI_SUCCESS != err) { line = __LINE__; goto cleanup_and_return; }
  }

  free(received);

  return MPI_SUCCESS;

cleanup_and_return:
  BINE_DEBUG_PRINT("\n%s:%4d\tRank %d Error occurred %d\n\n", __FILE__, line, rank, err);
  (void)line;  // silence compiler warning
  if(NULL!= received)     free(received);

  return err;
}

int bcast_bine_lat_new(void *buf, size_t count, MPI_Datatype dtype, int root, MPI_Comm comm)
{
  int size, rank, dtsize, err = MPI_SUCCESS, btnb_vrank;
  int vrank, mask, recvd;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  MPI_Type_size(dtype, &dtsize);

  if(!is_power_of_two(size)) return MPI_ERR_SIZE;

  vrank = mod(rank - root, size); // mod computes math modulo rather than reminder
  mask = 0x1 << (int) (log_2(size) - 1);
  recvd = (root == rank);
  btnb_vrank = binary_to_negabinary(vrank);
  while(mask > 0){
    int partner = btnb_vrank ^ ((mask << 1) - 1);
    partner = mod(negabinary_to_binary(partner) + root, size);
    int mask_lsbs = (mask << 1) - 1; // Mask with num_steps - step + 1 LSBs set to 1
    int lsbs = btnb_vrank & mask_lsbs; // Extract k LSBs
    int equal_lsbs = (lsbs == 0 || lsbs == mask_lsbs);

    if(recvd){
      err = MPI_Send(buf, count, dtype, partner, 0, comm);
      if(MPI_SUCCESS != err) return err;
    }else if(equal_lsbs){
      err = MPI_Recv(buf, count, dtype, partner, 0, comm, MPI_STATUS_IGNORE);
      if(MPI_SUCCESS != err) return err;
      recvd = 1;
    }
    mask >>= 1;
  }

  return MPI_SUCCESS;
}

int bcast_bine_lat_i_new(void *buf, size_t count, MPI_Datatype dtype, int root, MPI_Comm comm)
{
  int size, rank, dtsize, err = MPI_SUCCESS, btnb_vrank;
  int vrank, mask, recvd, req_count = 0, steps;
  MPI_Request *requests;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  MPI_Type_size(dtype, &dtsize);

  if(!is_power_of_two(size)) return MPI_ERR_SIZE;

  vrank = mod(rank - root, size); // mod computes math modulo rather than reminder
  steps = log_2(size);
  mask = 0x1 << (int) (steps - 1);
  recvd = (root == rank);
  btnb_vrank = binary_to_negabinary(vrank);
  requests = (MPI_Request *) malloc(steps * sizeof(MPI_Request));
  if(requests == NULL) return MPI_ERR_NO_MEM;
  while(mask > 0){
    int partner = btnb_vrank ^ ((mask << 1) - 1);
    partner = mod(negabinary_to_binary(partner) + root, size);
    int mask_lsbs = (mask << 1) - 1; // Mask with num_steps - step + 1 LSBs set to 1
    int lsbs = btnb_vrank & mask_lsbs; // Extract k LSBs
    int equal_lsbs = (lsbs == 0 || lsbs == mask_lsbs);

    if(recvd){
      err = MPI_Isend(buf, count, dtype, partner, 0, comm, &requests[req_count++]);
      if(MPI_SUCCESS != err) { goto err_hndl; }
    }else if(equal_lsbs){
      err = MPI_Recv(buf, count, dtype, partner, 0, comm, MPI_STATUS_IGNORE);
      if(MPI_SUCCESS != err) { goto err_hndl; }
      recvd = 1;
    }
    mask >>= 1;
  }

  MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);

  free(requests);
  return MPI_SUCCESS;

err_hndl:
  if (NULL != requests) free(requests);
  return err;
}


int bcast_bine_bdw_remap(void *buffer, size_t count, MPI_Datatype dt, int root, MPI_Comm comm){
  assert(root == 0); // TODO: Generalize
  int size, rank, dtsize, err = MPI_SUCCESS;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  MPI_Type_size(dt, &dtsize);

  int* displs = (int*) malloc(size*sizeof(int));
  int* recvcounts = (int*) malloc(size*sizeof(int));
  if(displs == NULL || recvcounts == NULL){
    err = MPI_ERR_NO_MEM;
    goto err_hndl;
  }
  int count_per_rank = count / size;
  int rem = count % size;
  for(int i = 0; i < size; i++){
    displs[i] = count_per_rank*i + (i < rem ? i : rem);
    recvcounts[i] = count_per_rank + (i < rem ? 1 : 0);
  }

  int mask = 0x1;
  int inverse_mask = 0x1 << (int) (log_2(size) - 1);
  int block_first_mask = ~(inverse_mask - 1);
  int remapped_rank = remap_rank(size, rank);
  int receiving_mask = inverse_mask << 1; // Root never receives. By having a large mask inverse_mask will always be < receiving_mask
  // I receive in the step corresponding to the position (starting from right)
  // of the first 1 in my remapped rank -- this indicates the step when the data reaches me
  if(rank != root){
    receiving_mask = 0x1 << (ffs(remapped_rank) - 1); // ffs starts counting from 1, thus -1
  }
  
  /***** Scatter *****/
  int recvd = (root == rank);
  while(mask < size){
    int partner;
    if(rank % 2 == 0){
      partner = mod(rank + negabinary_to_binary((mask << 1) - 1), size); 
    }else{
      partner = mod(rank - negabinary_to_binary((mask << 1) - 1), size); 
    }
  
    // For sure I need to send my (remapped) partner's data
    // the actual start block however must be aligned to 
    // the power of two
    int send_block_first = remap_rank(size, partner) & block_first_mask;
    int send_block_last = send_block_first + inverse_mask - 1;
    int send_count = displs[send_block_last] - displs[send_block_first] + recvcounts[send_block_last];
    // Something similar for the block to recv.
    // I receive my block, but aligned to the power of two
    int recv_block_first = remapped_rank & block_first_mask;
    int recv_block_last = recv_block_first + inverse_mask - 1;
    int recv_count = displs[recv_block_last] - displs[recv_block_first] + recvcounts[recv_block_last];
    
    if(recvd){
      err = MPI_Send((char*) buffer + displs[send_block_first]*dtsize, send_count, dt, partner, 0, comm);
      if(MPI_SUCCESS != err) { goto err_hndl; }
    }else if(inverse_mask == receiving_mask || partner == root){
      err = MPI_Recv((char*) buffer + displs[recv_block_first]*dtsize, recv_count, dt, partner, 0, comm, MPI_STATUS_IGNORE);
      if(MPI_SUCCESS != err) { goto err_hndl; }
      recvd = 1;
    }

    mask <<= 1;
    inverse_mask >>= 1;
    block_first_mask >>= 1;
  }

  /***** Allgather *****/  
  mask >>= 1;
  inverse_mask = 0x1;
  block_first_mask = ~0x0;
  while(mask > 0){
    int spartner, rpartner;
    int send_block_first = 0, send_block_last = 0, send_count = 0, recv_block_first = 0, recv_block_last = 0, recv_count = 0;
    int partner;
    if(rank % 2 == 0){
      partner = mod(rank + negabinary_to_binary((mask << 1) - 1), size); 
    }else{
      partner = mod(rank - negabinary_to_binary((mask << 1) - 1), size); 
    }

    rpartner = (inverse_mask < receiving_mask) ? MPI_PROC_NULL : partner;
    spartner = (inverse_mask == receiving_mask) ? MPI_PROC_NULL : partner;

    if(spartner != MPI_PROC_NULL){
      send_block_first = remapped_rank & block_first_mask;
      send_block_last = send_block_first + inverse_mask - 1;
      send_count = displs[send_block_last] - displs[send_block_first] + recvcounts[send_block_last];  
    }
    if(rpartner != MPI_PROC_NULL){
      recv_block_first = remap_rank(size, rpartner) & block_first_mask;
      recv_block_last = recv_block_first + inverse_mask - 1;
      recv_count = displs[recv_block_last] - displs[recv_block_first] + recvcounts[recv_block_last];
    }
    err = MPI_Sendrecv((char*) buffer + displs[send_block_first]*dtsize, send_count, dt, spartner, 0, 
           (char*) buffer + displs[recv_block_first]*dtsize, recv_count, dt, rpartner, 0, comm, MPI_STATUS_IGNORE);
    if(MPI_SUCCESS != err) { goto err_hndl; }

    mask >>= 1;
    inverse_mask <<= 1;
    block_first_mask <<= 1;
  }


  free(displs);
  free(recvcounts);
  return MPI_SUCCESS;

err_hndl:
  if(NULL!= displs)     free(displs);
  if(NULL!= recvcounts) free(recvcounts);
  return err;
}
