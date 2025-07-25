/*
 * Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
 * Licensed under the MIT License
 */

#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>

#include "libbine.h"
#include "libbine_utils.h"
#include "libbine_utils_bitmaps.h"

size_t bine_allreduce_segsize = 0;

int allreduce_recursivedoubling(const void *sbuf, void *rbuf, size_t count,
                                MPI_Datatype dtype, MPI_Op op, MPI_Comm comm)
{
  int ret, line, rank, size, adjsize, remote, distance;
  int newrank, newremote, extra_ranks;
  char *tmpsend = NULL, *tmprecv = NULL, *inplacebuf_free = NULL, *inplacebuf;
  ptrdiff_t span, gap = 0;

  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  /* Special case for size == 1 */
  if(1 == size) {
    if(MPI_IN_PLACE != sbuf) {
      ret = copy_buffer((char *) sbuf, (char *) rbuf, count, dtype);
      if(ret < 0) { line = __LINE__; goto error_hndl; }
    }
    return MPI_SUCCESS;
  }

  /* Allocate and initialize temporary send buffer */
  span = datatype_span(dtype, count, &gap);

  inplacebuf_free = (char*) malloc(span);
  if(NULL == inplacebuf_free) { ret = -1; line = __LINE__; goto error_hndl; }
  inplacebuf = inplacebuf_free - gap;

  if(MPI_IN_PLACE == sbuf) {
      ret = copy_buffer((char*)rbuf, inplacebuf, count, dtype);
      if(ret < 0) { line = __LINE__; goto error_hndl; }
  } else {
      ret = copy_buffer((char*)sbuf, inplacebuf, count, dtype);
      if(ret < 0) { line = __LINE__; goto error_hndl; }
  }

  tmpsend = (char*) inplacebuf;
  tmprecv = (char*) rbuf;

  /* Determine nearest power of two less than or equal to size */
  adjsize = next_poweroftwo(size) >> 1;

  /* Handle non-power-of-two case:
     - Even ranks less than 2 * extra_ranks send their data to (rank + 1), and
     sets new rank to -1.
     - Odd ranks less than 2 * extra_ranks receive data from (rank - 1),
     apply appropriate operation, and set new rank to rank/2
     - Everyone else sets rank to rank - extra_ranks
  */
  extra_ranks = size - adjsize;
  if(rank <  (2 * extra_ranks)) {
    if(0 == (rank % 2)) {
      ret = MPI_Send(tmpsend, count, dtype, (rank + 1), 0, comm);
      if(MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
      newrank = -1;
    } else {
      ret = MPI_Recv(tmprecv, count, dtype, (rank - 1), 0, comm,
                  MPI_STATUS_IGNORE);
      if(MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
      /* tmpsend = tmprecv (op) tmpsend */
      // reduction((int64_t *) tmprecv, (int64_t *) tmpsend, count);
      MPI_Reduce_local((char *) tmprecv, (char *) tmpsend, count, dtype, op);
      newrank = rank >> 1;
    }
  } else {
    newrank = rank - extra_ranks;
  }

  /* Communication/Computation loop
     - Exchange message with remote node.
     - Perform appropriate operation taking in account order of operations:
     result = value (op) result
  */
  for(distance = 0x1; distance < adjsize; distance <<=1) {
    if(newrank < 0) break;
    /* Determine remote node */
    newremote = newrank ^ distance;
    remote = (newremote < extra_ranks) ? (newremote * 2 + 1) : (newremote + extra_ranks);
   
    /* Exchange the data */
    ret = MPI_Sendrecv(tmpsend, count, dtype, remote, 0,
                       tmprecv, count, dtype, remote, 0,
                       comm, MPI_STATUS_IGNORE);
    if(MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }

    // reduction((int64_t *) tmprecv, (int64_t *) tmpsend, count);
    MPI_Reduce_local((char *) tmprecv, (char *) tmpsend, count, dtype, op);
  }

  /* Handle non-power-of-two case:
     - Odd ranks less than 2 * extra_ranks send result from tmpsend to
     (rank - 1)
     - Even ranks less than 2 * extra_ranks receive result from (rank + 1)
  */
  if(rank < (2 * extra_ranks)) {
    if(0 == (rank % 2)) {
      ret = MPI_Recv(rbuf, count, dtype, (rank + 1), 0, comm, MPI_STATUS_IGNORE);
      if(MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
      tmpsend = (char*)rbuf;
    } else {
      ret = MPI_Send(tmpsend, count, dtype, (rank - 1), 0, comm);
      if(MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
    }
  }

  /* Ensure that the final result is in rbuf */
  if(tmpsend != rbuf) {
    ret = copy_buffer(tmpsend, (char*)rbuf, count, dtype);
    if(ret < 0) { line = __LINE__; goto error_hndl; }
  }

  if(NULL != inplacebuf_free) free(inplacebuf_free);
  return MPI_SUCCESS;

  error_hndl:
    BINE_DEBUG_PRINT("\n%s:%4d\tRank %d Error occurred %d\n", __FILE__, line, rank, ret);
    (void)line;  // silence compiler warning
    if(NULL != inplacebuf_free) free(inplacebuf_free);
    return ret;
}


int allreduce_ring(const void *sbuf, void *rbuf, size_t count, MPI_Datatype dtype,
                   MPI_Op op, MPI_Comm comm)
{
  int ret, line, rank, size, k, recv_from, send_to, block_count, inbi;
  int early_segcount, late_segcount, split_rank, max_segcount;
  char *tmpsend = NULL, *tmprecv = NULL, *inbuf[2] = {NULL, NULL};
  ptrdiff_t true_lb, true_extent, lb, extent;
  ptrdiff_t block_offset, max_real_segsize;
  MPI_Request reqs[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};

  ret = MPI_Comm_rank(comm, &rank);
  if(MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
  ret = MPI_Comm_size(comm, &size);
  if(MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }

  // if(rank == 0) {
  //   printf("4: RING\n");
  //   fflush(stdout);
  // }

  /* Special case for size == 1 */
  if(1 == size) {
    if(MPI_IN_PLACE != sbuf) {
      ret = copy_buffer((char *) sbuf, (char *) rbuf, count, dtype);
      if(ret < 0) { line = __LINE__; goto error_hndl; }
    }
    return MPI_SUCCESS;
  }

  /* Special case for count less than size - use recursive doubling */
  if(count < (size_t) size) {
    return (allreduce_recursivedoubling(sbuf, rbuf, count, dtype, op, comm));
  }

  /* Allocate and initialize temporary buffers */
  ret = MPI_Type_get_extent(dtype, &lb, &extent);
  if(MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
  ret = MPI_Type_get_true_extent(dtype, &true_lb, &true_extent);
  if(MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }

  /* Determine the number of elements per block and corresponding
     block sizes.
     The blocks are divided into "early" and "late" ones:
     blocks 0 .. (split_rank - 1) are "early" and
     blocks (split_rank) .. (size - 1) are "late".
     Early blocks are at most 1 element larger than the late ones.
  */
  COLL_BASE_COMPUTE_BLOCKCOUNT( count, size, split_rank,
                   early_segcount, late_segcount );
  max_segcount = early_segcount;
  max_real_segsize = true_extent + (max_segcount - 1) * extent;


  inbuf[0] = (char*)malloc(max_real_segsize);
  if(NULL == inbuf[0]) { ret = -1; line = __LINE__; goto error_hndl; }
  if(size > 2) {
    inbuf[1] = (char*)malloc(max_real_segsize);
    if(NULL == inbuf[1]) { ret = -1; line = __LINE__; goto error_hndl; }
  }

  /* Handle MPI_IN_PLACE */
  if(MPI_IN_PLACE != sbuf) {
    ret = copy_buffer((char *)sbuf, (char *) rbuf, count, dtype);
    if(ret < 0) { line = __LINE__; goto error_hndl; }
  }

  /* Computation loop */

  /*
     For each of the remote nodes:
     - post irecv for block (r-1)
     - send block (r)
     - in loop for every step k = 2 .. n
     - post irecv for block (r + n - k) % n
     - wait on block (r + n - k + 1) % n to arrive
     - compute on block (r + n - k + 1) % n
     - send block (r + n - k + 1) % n
     - wait on block (r + 1)
     - compute on block (r + 1)
     - send block (r + 1) to rank (r + 1)
     Note that we must be careful when computing the beginning of buffers and
     for send operations and computation we must compute the exact block size.
  */
  send_to = (rank + 1) % size;
  recv_from = (rank + size - 1) % size;

  inbi = 0;
  /* Initialize first receive from the neighbor on the left */
  ret = MPI_Irecv(inbuf[inbi], max_segcount, dtype, recv_from, 0, comm, &reqs[inbi]);
  if(MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
  /* Send first block (my block) to the neighbor on the right */
  block_offset = ((rank < split_rank)?
          ((ptrdiff_t)rank * (ptrdiff_t)early_segcount) :
          ((ptrdiff_t)rank * (ptrdiff_t)late_segcount + split_rank));
  block_count = ((rank < split_rank)? early_segcount : late_segcount);
  tmpsend = ((char*)rbuf) + block_offset * extent;
  ret = MPI_Send(tmpsend, block_count, dtype, send_to, 0, comm);
  if(MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }

  for(k = 2; k < size; k++) {
    const int prevblock = (rank + size - k + 1) % size;

    inbi = inbi ^ 0x1;

    /* Post irecv for the current block */
    ret = MPI_Irecv(inbuf[inbi], max_segcount, dtype, recv_from, 0, comm, &reqs[inbi]);
    if(MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }

    /* Wait on previous block to arrive */
    ret = MPI_Wait(&reqs[inbi ^ 0x1], MPI_STATUS_IGNORE);
    if(MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }

    /* Apply operation on previous block: result goes to rbuf
       rbuf[prevblock] = inbuf[inbi ^ 0x1] (op) rbuf[prevblock]
    */
    block_offset = ((prevblock < split_rank)?
            ((ptrdiff_t)prevblock * early_segcount) :
            ((ptrdiff_t)prevblock * late_segcount + split_rank));
    block_count = ((prevblock < split_rank)? early_segcount : late_segcount);
    tmprecv = ((char*)rbuf) + (ptrdiff_t)block_offset * extent;
    MPI_Reduce_local(inbuf[inbi ^ 0x1], tmprecv, block_count, dtype, op);

    /* send previous block to send_to */
    ret = MPI_Send(tmprecv, block_count, dtype, send_to, 0, comm);
    if(MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
  }

  /* Wait on the last block to arrive */
  ret = MPI_Wait(&reqs[inbi], MPI_STATUS_IGNORE);
  if(MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }

  /* Apply operation on the last block (from neighbor (rank + 1)
     rbuf[rank+1] = inbuf[inbi] (op) rbuf[rank + 1] */
  recv_from = (rank + 1) % size;
  block_offset = ((recv_from < split_rank)?
          ((ptrdiff_t)recv_from * early_segcount) :
          ((ptrdiff_t)recv_from * late_segcount + split_rank));
  block_count = ((recv_from < split_rank)? early_segcount : late_segcount);
  tmprecv = ((char*)rbuf) + (ptrdiff_t)block_offset * extent;
  MPI_Reduce_local(inbuf[inbi], tmprecv, block_count, dtype, op);

  /* Distribution loop - variation of ring allgather */
  send_to = (rank + 1) % size;
  recv_from = (rank + size - 1) % size;
  for(k = 0; k < size - 1; k++) {
    const int recv_data_from = (rank + size - k) % size;
    const int send_data_from = (rank + 1 + size - k) % size;
    const int send_block_offset =
      ((send_data_from < split_rank)?
       ((ptrdiff_t)send_data_from * early_segcount) :
       ((ptrdiff_t)send_data_from * late_segcount + split_rank));
    const int recv_block_offset =
      ((recv_data_from < split_rank)?
       ((ptrdiff_t)recv_data_from * early_segcount) :
       ((ptrdiff_t)recv_data_from * late_segcount + split_rank));
    block_count = ((send_data_from < split_rank)?
             early_segcount : late_segcount);

    tmprecv = (char*)rbuf + (ptrdiff_t)recv_block_offset * extent;
    tmpsend = (char*)rbuf + (ptrdiff_t)send_block_offset * extent;

    ret = MPI_Sendrecv(tmpsend, block_count, dtype, send_to, 0,
                       tmprecv, max_segcount, dtype, recv_from,
                       0, comm, MPI_STATUS_IGNORE);
    if(MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl;}

  }

  if(NULL != inbuf[0]) free(inbuf[0]);
  if(NULL != inbuf[1]) free(inbuf[1]);

  return MPI_SUCCESS;

 error_hndl:
  BINE_DEBUG_PRINT("\n%s:%4d\tRank %d Error occurred %d\n\n", __FILE__, line, rank, ret);
  MPI_Request_free(&reqs[0]);
  MPI_Request_free(&reqs[1]);
  (void)line;  // silence compiler warning
  if(NULL != inbuf[0]) free(inbuf[0]);
  if(NULL != inbuf[1]) free(inbuf[1]);
  return ret;
}

int allreduce_bine_lat(const void *sbuf, void *rbuf, size_t count, MPI_Datatype dtype, MPI_Op op, MPI_Comm comm) {
  int rank, size;
  int ret, line; // for error handling
  char *tmpsend, *tmprecv, *inplacebuf_free = NULL;
  ptrdiff_t extent, true_extent, lb,gap, span = 0;
  

  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  // Special case for size == 1
  if(1 == size) {
    if(MPI_IN_PLACE != sbuf) {
      ret = copy_buffer((char *) sbuf, (char *) rbuf, count, dtype);
      if(ret < 0) { line = __LINE__; goto error_hndl; }
    }
    return MPI_SUCCESS;
  }

  // Allocate and initialize temporary send buffer
  MPI_Type_get_extent(dtype, &lb, &extent);
  MPI_Type_get_true_extent(dtype, &gap, &true_extent);
  span = true_extent + extent * (count - 1);
  inplacebuf_free = (char*) malloc(span + gap);
  char *inplacebuf = inplacebuf_free + gap;

  // Copy content from sbuffer to inplacebuf
  if(MPI_IN_PLACE == sbuf) {
      ret = copy_buffer((char*)rbuf, inplacebuf, count, dtype);
      if(ret < 0) { line = __LINE__; goto error_hndl; }
  } else {
      ret = copy_buffer((char*)sbuf, inplacebuf, count, dtype);
      if(ret < 0) { line = __LINE__; goto error_hndl; }
  }

  tmpsend = inplacebuf;
  tmprecv = (char*) rbuf;
  
  // Determine nearest power of two less than or equal to size
  // and return an error if size is 0
  int steps = hibit(size, (int)(sizeof(size) * CHAR_BIT) - 1);
  if(steps == -1) {
      return MPI_ERR_ARG;
  }
  int adjsize = 1 << steps;  // Largest power of two <= size

  // Number of nodes that exceed the largest power of two less than or equal to size
  int extra_ranks = size - adjsize;
  int is_power_of_two = (size & (size - 1)) == 0;


  // First part of computation to get a 2^n number of nodes.
  // What happens is that first #extra_rank even nodes sends their
  // data to the successive node and do not partecipate in the general
  // collective call operation.
  // All the nodes that do not stop their computation will receive an alias
  // called new_node, used to calculate their correct destination wrt this
  // new "cut" topology.
  int new_rank = rank, loop_flag = 0;
  if(rank <  (2 * extra_ranks)) {
    if(0 == (rank % 2)) {
      ret = MPI_Send(tmpsend, count, dtype, (rank + 1), 0, comm);
      if(MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
      loop_flag = 1;
    } else {
      ret = MPI_Recv(tmprecv, count, dtype, (rank - 1), 0, comm, MPI_STATUS_IGNORE);
      if(MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
      MPI_Reduce_local((char *) tmprecv, (char *) tmpsend, count, dtype, op);
      new_rank = rank >> 1;
    }
  } else new_rank = rank - extra_ranks;
  
  
  // Actual allreduce computation for general cases
  int s, vdest, dest;
  for(s = 0; s < steps; s++){
    if(loop_flag) break;
    vdest = pi(new_rank, s, adjsize);

    dest = is_power_of_two ?
              vdest :
              (vdest < extra_ranks) ?
              (vdest << 1) + 1 : vdest + extra_ranks;

    ret = MPI_Sendrecv(tmpsend, count, dtype, dest, 0,
                       tmprecv, count, dtype, dest, 0,
                       comm, MPI_STATUS_IGNORE);
    if(MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
    
    MPI_Reduce_local((char *) tmprecv, (char *) tmpsend, count, dtype, op);
  }
  
  // Final results is sent to nodes that are not included in general computation
  // (general computation loop requires 2^n nodes).
  if(rank < (2 * extra_ranks)){
    if(!loop_flag){
      ret = MPI_Send(tmpsend, count, dtype, (rank - 1), 0, comm);
      if(MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
    } else {
      ret = MPI_Recv(rbuf, count, dtype, (rank + 1), 0, comm, MPI_STATUS_IGNORE);
      if(MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
      tmpsend = (char*)rbuf;
    }
  }

  if(tmpsend != rbuf) {
    ret = copy_buffer(tmpsend, (char*) rbuf, count, dtype);
    if(ret < 0) { line = __LINE__; goto error_hndl; }
  }

  free(inplacebuf_free);
  return MPI_SUCCESS;

  error_hndl:
    BINE_DEBUG_PRINT("\n%s:%4d\tRank %d Error occurred %d\n\n", __FILE__, line, rank, ret);
    (void)line;  // silence compiler warning
    if(NULL != inplacebuf_free) free(inplacebuf_free);
    return ret;
}

int allreduce_rabenseifner(const void *sbuf, void *rbuf, size_t count,
                           MPI_Datatype dtype, MPI_Op op, MPI_Comm comm)
{
  int *rindex = NULL, *rcount = NULL, *sindex = NULL, *scount = NULL;
  int size, rank;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  // if(rank == 0) {
  //   printf("6: RABENSEIFNER\n");
  //   fflush(stdout);
  // }

  // Find number of steps of scatter-reduce and allgather,
  // biggest power of two smaller or equal to size,
  // size of send_window (number of chunks to send/recv at each step)
  // and alias of the rank to be used if size != adj_size
  //Determine nearest power of two less than or equal to size
  int steps = hibit(size, (int) (sizeof(size) * CHAR_BIT) - 1);
  if(-1 == steps){
    return MPI_ERR_ARG;
  }
  int adjsize = 1 << steps;

  int err = MPI_SUCCESS;
  ptrdiff_t lb, extent, gap = 0;
  MPI_Type_get_extent(dtype, &lb, &extent);
  ptrdiff_t buf_size = datatype_span(dtype, count, &gap);

  /* Temporary buffer for receiving messages */
  char *tmp_buf = NULL;
  char *tmp_buf_raw = (char *)malloc(buf_size);
  if(NULL == tmp_buf_raw)
    return MPI_ERR_UNKNOWN;
  tmp_buf = tmp_buf_raw - gap;

  if(sbuf != MPI_IN_PLACE) {
    err = copy_buffer((char *)sbuf, (char *)rbuf, count, dtype);
    if(MPI_SUCCESS != err) { goto cleanup_and_return; }
  }

  /*
   * Step 1. Reduce the number of processes to the nearest lower power of two
   * p' = 2^{\floor{\log_2 p}} by removing r = p - p' processes.
   * 1. In the first 2r processes (ranks 0 to 2r - 1), all the even ranks send
   *  the second half of the input vector to their right neighbor (rank + 1)
   *  and all the odd ranks send the first half of the input vector to their
   *  left neighbor (rank - 1).
   * 2. All 2r processes compute the reduction on their half.
   * 3. The odd ranks then send the result to their left neighbors
   *  (the even ranks).
   *
   * The even ranks (0 to 2r - 1) now contain the reduction with the input
   * vector on their right neighbors (the odd ranks). The first r even
   * processes and the p - 2r last processes are renumbered from
   * 0 to 2^{\floor{\log_2 p}} - 1.
   */

  int vrank, step, wsize;
  int nprocs_rem = size - adjsize;

  if(rank < 2 * nprocs_rem) {
    int count_lhalf = count / 2;
    int count_rhalf = count - count_lhalf;

    if(rank % 2 != 0) {
      /*
       * Odd process -- exchange with rank - 1
       * Send the left half of the input vector to the left neighbor,
       * Recv the right half of the input vector from the left neighbor
       */
      err = MPI_Sendrecv(rbuf, count_lhalf, dtype, rank - 1, 0,
                      (char *)tmp_buf + (ptrdiff_t)count_lhalf * extent,
                      count_rhalf, dtype, rank - 1, 0, comm, MPI_STATUS_IGNORE);
      if(MPI_SUCCESS != err) { goto cleanup_and_return; }

      /* Reduce on the right half of the buffers (result in rbuf) */
      MPI_Reduce_local((char *)tmp_buf + (ptrdiff_t)count_lhalf * extent,
                       (char *)rbuf + count_lhalf * extent, count_rhalf, dtype, op);

      /* Send the right half to the left neighbor */
      err = MPI_Send((char *)rbuf + (ptrdiff_t)count_lhalf * extent,
                     count_rhalf, dtype, rank - 1, 0, comm);
      if(MPI_SUCCESS != err) { goto cleanup_and_return; }

      /* This process does not pariticipate in recursive doubling phase */
      vrank = -1;

    } else {
      /*
       * Even process -- exchange with rank + 1
       * Send the right half of the input vector to the right neighbor,
       * Recv the left half of the input vector from the right neighbor
       */
      err = MPI_Sendrecv((char *)rbuf + (ptrdiff_t)count_lhalf * extent,
                      count_rhalf, dtype, rank + 1, 0,
                      tmp_buf, count_lhalf, dtype, rank + 1, 0, comm,
                      MPI_STATUS_IGNORE);
      if(MPI_SUCCESS != err) { goto cleanup_and_return; }

      /* Reduce on the right half of the buffers (result in rbuf) */
      MPI_Reduce_local(tmp_buf, rbuf, count_lhalf, dtype, op);

      /* Recv the right half from the right neighbor */
      err = MPI_Recv((char *)rbuf + (ptrdiff_t)count_lhalf * extent,
                  count_rhalf, dtype, rank + 1, 0, comm, MPI_STATUS_IGNORE);
      if(MPI_SUCCESS != err) { goto cleanup_and_return; }

      vrank = rank / 2;
    }
  } else { /* rank >= 2 * nprocs_rem */
    vrank = rank - nprocs_rem;
  }

  /*
   * Step 2. Reduce-scatter implemented with recursive vector halving and
   * recursive distance doubling. We have p' = 2^{\floor{\log_2 p}}
   * power-of-two number of processes with new ranks (vrank) and result in rbuf.
   *
   * The even-ranked processes send the right half of their buffer to rank + 1
   * and the odd-ranked processes send the left half of their buffer to
   * rank - 1. All processes then compute the reduction between the local
   * buffer and the received buffer. In the next \log_2(p') - 1 steps, the
   * buffers are recursively halved, and the distance is doubled. At the end,
   * each of the p' processes has 1 / p' of the total reduction result.
   */
  rindex = malloc(sizeof(*rindex) * steps);
  sindex = malloc(sizeof(*sindex) * steps);
  rcount = malloc(sizeof(*rcount) * steps);
  scount = malloc(sizeof(*scount) * steps);
  if(NULL == rindex || NULL == sindex || NULL == rcount || NULL == scount) {
    err = MPI_ERR_UNKNOWN;
    goto cleanup_and_return;
  }

  if(vrank != -1) {
    step = 0;
    wsize = count;
    sindex[0] = rindex[0] = 0;

    for(int mask = 1; mask < adjsize; mask <<= 1) {
      /*
       * On each iteration: rindex[step] = sindex[step] -- beginning of the
       * current window. Length of the current window is storded in wsize.
       */
      int vdest = vrank ^ mask;
      /* Translate vdest virtual rank to real rank */
      int dest = (vdest < nprocs_rem) ? vdest * 2 : vdest + nprocs_rem;

      if(rank < dest) {
        /*
         * Recv into the left half of the current window, send the right
         * half of the window to the peer (perform reduce on the left
         * half of the current window)
         */
        rcount[step] = wsize / 2;
        scount[step] = wsize - rcount[step];
        sindex[step] = rindex[step] + rcount[step];
      } else {
        /*
         * Recv into the right half of the current window, send the left
         * half of the window to the peer (perform reduce on the right
         * half of the current window)
         */
        scount[step] = wsize / 2;
        rcount[step] = wsize - scount[step];
        rindex[step] = sindex[step] + scount[step];
      }

      /* Send part of data from the rbuf, recv into the tmp_buf */
      err = MPI_Sendrecv((char *)rbuf + (ptrdiff_t)sindex[step] * extent,
                      scount[step], dtype, dest, 0,
                      (char *)tmp_buf + (ptrdiff_t)rindex[step] * extent,
                      rcount[step], dtype, dest, 0, comm,
                      MPI_STATUS_IGNORE);
      if(MPI_SUCCESS != err) { goto cleanup_and_return; }

      /* Local reduce: rbuf[] = tmp_buf[] <op> rbuf[] */
      MPI_Reduce_local((char *)tmp_buf + (ptrdiff_t)rindex[step] * extent,
               (char *)rbuf + (ptrdiff_t)rindex[step] * extent,
               rcount[step], dtype, op);

      /* Move the current window to the received message */
      if(step + 1 < steps) {
        rindex[step + 1] = rindex[step];
        sindex[step + 1] = rindex[step];
        wsize = rcount[step];
        step++;
      }
    }
    /*
     * Assertion: each process has 1 / p' of the total reduction result:
     * rcount[nsteps - 1] elements in the rbuf[rindex[nsteps - 1], ...].
     */

    /*
     * Step 3. Allgather by the recursive doubling algorithm.
     * Each process has 1 / p' of the total reduction result:
     * rcount[nsteps - 1] elements in the rbuf[rindex[nsteps - 1], ...].
     * All exchanges are executed in reverse order relative
     * to recursive doubling (previous step).
     */

    step = steps - 1;

    for(int mask = adjsize >> 1; mask > 0; mask >>= 1) {
      int vdest = vrank ^ mask;
      /* Translate vdest virtual rank to real rank */
      int dest = (vdest < nprocs_rem) ? vdest * 2 : vdest + nprocs_rem;

      /*
       * Send rcount[step] elements from rbuf[rindex[step]...]
       * Recv scount[step] elements to rbuf[sindex[step]...]
       */
      err = MPI_Sendrecv((char *)rbuf + (ptrdiff_t)rindex[step] * extent,
                      rcount[step], dtype, dest, 0,
                      (char *)rbuf + (ptrdiff_t)sindex[step] * extent,
                      scount[step], dtype, dest, 0, comm, MPI_STATUS_IGNORE);
      if(MPI_SUCCESS != err) { goto cleanup_and_return; }
      step--;
    }
  }

  /*
   * Step 4. Send total result to excluded odd ranks.
   */
  if(rank < 2 * nprocs_rem) {
    if(rank % 2 != 0) {
      /* Odd process -- recv result from rank - 1 */
      err = MPI_Recv(rbuf, count, dtype, rank - 1,
                  0, comm, MPI_STATUS_IGNORE);
      if(MPI_SUCCESS != err) { goto cleanup_and_return; }

    } else {
      /* Even process -- send result to rank + 1 */
      err = MPI_Send(rbuf, count, dtype, rank + 1,
                  0, comm);
      if(MPI_SUCCESS != err) { goto cleanup_and_return; }
    }
  }

  cleanup_and_return:
  if(NULL != tmp_buf_raw)
    free(tmp_buf_raw);
  if(NULL != rindex)
    free(rindex);
  if(NULL != sindex)
    free(sindex);
  if(NULL != rcount)
    free(rcount);
  if(NULL != scount)
    free(scount);
  return err;
}

int allreduce_bine_bdw_static(const void *send_buf, void *recv_buf, size_t count,
                               MPI_Datatype dtype, MPI_Op op, MPI_Comm comm){
  int size, rank, dest, err = MPI_SUCCESS; 
  int steps, step, split_rank;
  size_t small_blocks, big_blocks;
  ptrdiff_t lb, extent, true_extent, gap = 0, buf_size;
  char *tmp_send = NULL, *tmp_recv = NULL;
  char *tmp_buf_raw = NULL, *tmp_buf;

  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  // Still not implemented for non power of two number of processes
  steps = log_2(size);
  if(!is_power_of_two(size) || steps < 1){
    return MPI_ERR_ARG;
  }
  
  // Find the dimension in number of elements of the blocks.
  // Also find the number of big and small blocks.
  // The big blocks are the firsts count % size blocks
  // that have an extra element.
  COLL_BASE_COMPUTE_BLOCKCOUNT(count, size, split_rank,
                               big_blocks, small_blocks);

  MPI_Type_get_extent(dtype, &lb, &extent);
  MPI_Type_get_true_extent(dtype, &gap, &true_extent);

  buf_size = true_extent + extent * (count >> 1);
  tmp_buf_raw = (char *)malloc(buf_size);
  tmp_buf = tmp_buf_raw - gap;

  // Copy into receive_buffer content of send_buffer to not produce
  // side effects on send_buffer
  if(send_buf != MPI_IN_PLACE) {
    err = copy_buffer((char *)send_buf, (char *)recv_buf, count, dtype);
    if(MPI_SUCCESS != err) { goto cleanup_and_return; }
  }
  
  const int *s_bitmap = NULL, *r_bitmap = NULL;
  if(get_static_bitmap(&s_bitmap, &r_bitmap, steps, size, rank) == -1){
    err = MPI_ERR_OTHER;
    goto cleanup_and_return;
  }
  
  int w_size = size;
  size_t s_count, r_count;
  ptrdiff_t s_offset, r_offset;
  // Reduce-Scatter phase
  for(step = 0; step < steps; step++) {
    w_size >>= 1;
    dest = pi(rank, step, size);

    s_count = (s_bitmap[step] + w_size <= split_rank) ?
                (size_t)w_size * big_blocks :
                  (s_bitmap[step] >= split_rank) ?
                    (size_t)w_size * small_blocks :
                    (size_t)w_size * small_blocks + (size_t)(split_rank - s_bitmap[step]);
    s_offset = (s_bitmap[step] <= split_rank) ?
                (ptrdiff_t) s_bitmap[step] * (ptrdiff_t)(big_blocks * extent) :
                (ptrdiff_t)(s_bitmap[step] * (int) small_blocks + split_rank) * (ptrdiff_t) extent;

    r_count = (r_bitmap[step] + w_size <= split_rank) ?
                (size_t)w_size * big_blocks :
                  (r_bitmap[step] >= split_rank) ?
                    (size_t)w_size * small_blocks :
                    (size_t)w_size * small_blocks + (size_t)(split_rank - r_bitmap[step]);
    r_offset = (r_bitmap[step] <= split_rank) ?
                (ptrdiff_t) r_bitmap[step] * (ptrdiff_t)(big_blocks * extent) :
                (ptrdiff_t)(r_bitmap[step] * (int) small_blocks + split_rank) * (ptrdiff_t) extent;
    
    tmp_send = (char *)recv_buf + s_offset;
    err = MPI_Sendrecv(tmp_send, s_count, dtype, dest, 0,
                       tmp_buf, r_count, dtype, dest, 0,
                       comm, MPI_STATUS_IGNORE);
    if(MPI_SUCCESS != err) { goto cleanup_and_return; }
    
    tmp_recv = (char *) recv_buf + r_offset;
    err = MPI_Reduce_local(tmp_buf, tmp_recv, r_count, dtype, op);
    if(MPI_SUCCESS != err) { goto cleanup_and_return; }
  }
  
  // Allgather phase
  for(step = steps - 1; step >= 0; step--) {
    dest = pi(rank, step, size);
    
    s_count = (s_bitmap[step] + w_size <= split_rank) ?
                (size_t)w_size * big_blocks :
                  (s_bitmap[step] >= split_rank) ?
                    (size_t)w_size * small_blocks :
                    (size_t)w_size * small_blocks + (size_t)(split_rank - s_bitmap[step]);
    s_offset = (s_bitmap[step] <= split_rank) ?
                (ptrdiff_t) s_bitmap[step] * (ptrdiff_t)(big_blocks * extent) :
                (ptrdiff_t)(s_bitmap[step] * (int) small_blocks + split_rank) * (ptrdiff_t) extent;

    r_count = (r_bitmap[step] + w_size <= split_rank) ?
                (size_t)w_size * big_blocks :
                  (r_bitmap[step] >= split_rank) ?
                    (size_t)w_size * small_blocks :
                    (size_t)w_size * small_blocks + (size_t)(split_rank - r_bitmap[step]);
    r_offset = (r_bitmap[step] <= split_rank) ?
                (ptrdiff_t) r_bitmap[step] * (ptrdiff_t)(big_blocks * extent) :
                (ptrdiff_t)(r_bitmap[step] * (int)small_blocks + split_rank) * (ptrdiff_t) extent;
    
    tmp_send = (char *)recv_buf + s_offset;
    tmp_recv = (char *)recv_buf + r_offset;

    err = MPI_Sendrecv(tmp_recv, r_count, dtype, dest, 0,
                       tmp_send, s_count, dtype, dest, 0,
                       comm, MPI_STATUS_IGNORE);
    if(MPI_SUCCESS != err) { goto cleanup_and_return; }

    w_size <<= 1;
  }

  free(tmp_buf_raw);

  return MPI_SUCCESS;
cleanup_and_return:
  if(NULL != tmp_buf_raw)  free(tmp_buf_raw);
  return err;
}


int allreduce_bine_bdw_remap(const void *send_buf, void *recv_buf, size_t count,
                              MPI_Datatype dtype, MPI_Op op, MPI_Comm comm){
  int size, rank, dest, steps, step, err = MPI_SUCCESS;
  int *r_count = NULL, *s_count = NULL, *r_index = NULL, *s_index = NULL;
  size_t w_size;
  uint32_t vrank, vdest;

  char *tmp_send = NULL, *tmp_recv = NULL;
  char *tmp_buf_raw = NULL, *tmp_buf;
  ptrdiff_t lb, extent, true_extent, gap = 0, buf_size;

  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  // Does not support non-power-of-two or negative sizes
  steps = log_2(size);
  if( !is_power_of_two(size) || steps == -1 ) {
    return MPI_ERR_ARG;
  }

  // Allocate temporary buffer for send/recv and reduce operations
  MPI_Type_get_extent(dtype, &lb, &extent);
  MPI_Type_get_true_extent(dtype, &gap, &true_extent);
  buf_size = true_extent + extent * (count >> 1);
  tmp_buf_raw = (char *)malloc(buf_size);
  tmp_buf = tmp_buf_raw - gap;

  // Copy into receive_buffer content of send_buffer to not produce
  // side effects on send_buffer
  if(send_buf != MPI_IN_PLACE) {
    err = copy_buffer((char *)send_buf, (char *)recv_buf, count, dtype);
    if(MPI_SUCCESS != err) { goto cleanup_and_return; }
  }

  r_index = malloc(sizeof(*r_index) * steps);
  s_index = malloc(sizeof(*s_index) * steps);
  r_count = malloc(sizeof(*r_count) * steps);
  s_count = malloc(sizeof(*s_count) * steps);
  if(NULL == r_index || NULL == s_index || NULL == r_count || NULL == s_count) {
    err = MPI_ERR_NO_MEM;
    goto cleanup_and_return;
  }

  w_size = count;
  s_index[0] = r_index[0] = 0;
  vrank = remap_rank((uint32_t) size, (uint32_t) rank);

  // Reduce-Scatter phase
  for(step = 0; step < steps; step++) {
    dest = pi(rank, step, size);
    vdest = remap_rank((uint32_t) size, (uint32_t) dest);

    if(vrank < vdest) {
      r_count[step] = w_size / 2;
      s_count[step] = w_size - r_count[step];
      s_index[step] = r_index[step] + r_count[step];
    } else {
      s_count[step] = w_size / 2;
      r_count[step] = w_size - s_count[step];
      r_index[step] = s_index[step] + s_count[step];
    }
    tmp_send = (char *)recv_buf + s_index[step] * extent;
    err = MPI_Sendrecv(tmp_send, s_count[step], dtype, dest, 0,
                       tmp_buf, r_count[step], dtype, dest, 0,
                       comm, MPI_STATUS_IGNORE);
    if(MPI_SUCCESS != err) { goto cleanup_and_return; }

    tmp_recv = (char *) recv_buf + r_index[step] * extent;
    MPI_Reduce_local(tmp_buf, tmp_recv, r_count[step], dtype, op);

    if(step + 1 < steps) {
      r_index[step + 1] = r_index[step];
      s_index[step + 1] = r_index[step];
      w_size = r_count[step];
    }
  }

  // Allgather phase
  for(step = steps - 1; step >= 0; step--) {
    dest = pi(rank, step, size);

    tmp_send = (char *)recv_buf + r_index[step] * extent;
    tmp_recv = (char *)recv_buf + s_index[step] * extent;
    err = MPI_Sendrecv(tmp_send, r_count[step], dtype, dest, 0,
                       tmp_recv, s_count[step], dtype, dest, 0,
                       comm, MPI_STATUS_IGNORE);
    if(MPI_SUCCESS != err) { goto cleanup_and_return; }
  }

  free(tmp_buf_raw);
  free(r_index);
  free(s_index);
  free(r_count);
  free(s_count);
  return MPI_SUCCESS;

cleanup_and_return:
  if(NULL != tmp_buf_raw)  free(tmp_buf_raw);
  if(NULL != r_index)      free(r_index);
  if(NULL != s_index)      free(s_index);
  if(NULL != r_count)      free(r_count);
  if(NULL != s_count)      free(s_count);
  return err;
}

int allreduce_bine_block_by_block_any_even(const void *sbuf, void *rbuf, size_t count, 
                                            MPI_Datatype dt, MPI_Op op, MPI_Comm comm) {
  int size, rank, dtsize, err = MPI_SUCCESS;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  MPI_Type_size(dt, &dtsize);
  assert(size % 2 == 0); // This algorithm only works for even number of processes
  int count_so_far = 0;
  // TODO: We can avoid the extra array and do it as we do for the other algos
  int count_per_block = count / size;
  int* displs = (int*) malloc(size*sizeof(int));
  int* recvcounts = (int*) malloc(size*sizeof(int));
  for(int i = 0; i < size; i++){
    displs[i] = count_so_far;
    if(i < count % size){
      recvcounts[i] = count_per_block + 1; // Big block
    }else{
      recvcounts[i] = count_per_block; // Small block
    }
    count_so_far += recvcounts[i];
  }

  void* tmpbuf = malloc(count*dtsize);
  memcpy(rbuf, sbuf, count*dtsize);

  /**** Reduce-scatter phase ****/
  int mask = 0x1;
  MPI_Request* reqs_s = (MPI_Request*) malloc(size*sizeof(MPI_Request));  
  MPI_Request* reqs_r = (MPI_Request*) malloc(size*sizeof(MPI_Request));  
  int* blocks_to_recv = (int*) malloc(size*sizeof(int));
  int next_req_s = 0, next_req_r = 0;
  int reverse_step = log_2(size) - 1;
  while(mask < size){
    int partner;
    if(rank % 2 == 0){
        partner = mod(rank + negabinary_to_binary((mask << 1) - 1), size); 
    }else{
        partner = mod(rank - negabinary_to_binary((mask << 1) - 1), size); 
    }

    next_req_r = 0;
    next_req_s = 0;

    // We start from 1 because 0 never sends block 0
    for(size_t block = 1; block < size; block++){
      // Get the position of the highest set bit using clz
      // That gives us the first at which block departs from 0
      int k = 31 - __builtin_clz(get_nu(block, size));
      // Check if this must be sent
      if(k == reverse_step){
          // 0 would send this block
          size_t block_to_send, block_to_recv;
          if(rank % 2 == 0){
              // I am even, thus I need to shift by rank position to the right
              block_to_send = mod(block + rank, size);
              // What to receive? What my partner is sending
              // Since I am even, my partner is odd, thus I need to mirror it and then shift
              block_to_recv = mod(partner - block, size);
          }else{
              // I am odd, thus I need to mirror it
              block_to_send = mod(rank - block, size);
              // What to receive? What my partner is sending
              // Since I am odd, my partner is even, thus I need to mirror it and then shift   
              block_to_recv = mod(block + partner, size);
          }

          if(block_to_send != rank){
              err = MPI_Isend((char*) rbuf + displs[block_to_send]*dtsize, recvcounts[block_to_send], dt, partner, 0,
                          comm, &reqs_s[next_req_s]);
              if(MPI_SUCCESS != err) { goto err_hndl; }
              ++next_req_s;
          }

          if(block_to_recv != partner){
              blocks_to_recv[next_req_r] = block_to_recv;
              err = MPI_Irecv((char*) tmpbuf + displs[block_to_recv]*dtsize, recvcounts[block_to_recv], dt, partner, 0,
                      comm, &reqs_r[next_req_r]);
              if(MPI_SUCCESS != err) { goto err_hndl; }
              ++next_req_r;
          }
      }
    }

    for(size_t block = 0; block < next_req_r; block++){
        err = MPI_Wait(&reqs_r[block], MPI_STATUS_IGNORE);
        if(MPI_SUCCESS != err) { goto err_hndl; }
        err = MPI_Reduce_local((char*) tmpbuf + displs[blocks_to_recv[block]]*dtsize, (char*) rbuf + displs[blocks_to_recv[block]]*dtsize, recvcounts[blocks_to_recv[block]], dt, op);
        if(MPI_SUCCESS != err) { goto err_hndl; }
    }
    err = MPI_Waitall(next_req_s, reqs_s, MPI_STATUSES_IGNORE);
    if(MPI_SUCCESS != err) { goto err_hndl; }

    mask <<= 1;
    reverse_step--;
  }

  /**** Allgather phase ****/
  int step = 0;
  mask >>= 1; // We need to start from the last step
  while(mask > 0){
    int partner, req_count = 0;
    if(rank % 2 == 0){
      partner = mod(rank + negabinary_to_binary((mask << 1) - 1), size); 
    }else{
      partner = mod(rank - negabinary_to_binary((mask << 1) - 1), size); 
    }
    // We start from 1 because 0 never sends block 0
    for(size_t block = 1; block < size; block++){
      // Get the position of the highest set bit using clz
      // That gives us the first at which block departs from 0
      int k = 31 - __builtin_clz(get_nu(block, size));
      //int k = __builtin_ctz(get_nu(block, size));
      // Check if this must be sent (recvd in allgather)
      if(k == step || block == 0){
        // 0 would send this block
        size_t block_to_send, block_to_recv;
        // I invert what to send and what to receive wrt reduce-scatter
        if(rank % 2 == 0){
          // I am even, thus I need to shift by rank position to the right
          block_to_recv = mod(block + rank, size);
          // What to receive? What my partner is sending
          // Since I am even, my partner is odd, thus I need to mirror it and then shift
          block_to_send = mod(partner - block, size);
        }else{
          // I am odd, thus I need to mirror it
          block_to_recv = mod(rank - block, size);
          // What to receive? What my partner is sending
          // Since I am odd, my partner is even, thus I need to mirror it and then shift   
          block_to_send = mod(block + partner, size);
        }

        int partner_send = (block_to_send != partner) ? partner : MPI_PROC_NULL;
        int partner_recv = (block_to_recv != rank)  ? partner : MPI_PROC_NULL;

        err = MPI_Isend((char*) rbuf + displs[block_to_send]*dtsize, recvcounts[block_to_send], dt, partner_send, 0, comm, &reqs_s[req_count]);
        if(MPI_SUCCESS != err) { goto err_hndl; }

        err = MPI_Irecv((char*) rbuf + displs[block_to_recv]*dtsize, recvcounts[block_to_recv], dt, partner_recv, 0, comm, &reqs_r[req_count]);
        if(MPI_SUCCESS != err) { goto err_hndl; }
        
        ++req_count;
      }
    }
    err = MPI_Waitall(req_count, reqs_s, MPI_STATUSES_IGNORE);
    err = MPI_Waitall(req_count, reqs_r, MPI_STATUSES_IGNORE);
    mask >>= 1;
    step++;
  }

  free(blocks_to_recv);
  free(reqs_s);
  free(reqs_r);
  free(displs);
  free(tmpbuf);
  free(recvcounts);
  return MPI_SUCCESS;

err_hndl:
  if (NULL != blocks_to_recv) free(blocks_to_recv);
  if (NULL != reqs_s) free(reqs_s);
  if (NULL != reqs_r) free(reqs_r);
  if (NULL != displs) free(displs);
  if (NULL != tmpbuf) free(tmpbuf);
  if (NULL != recvcounts) free(recvcounts);
  return err;

}

int allreduce_bine_bdw_remap_segmented(const void *sbuf, void *rbuf, size_t count, 
                                        MPI_Datatype dtype, MPI_Op op, MPI_Comm comm) {
  int size, rank, dest, steps, step, err = MPI_SUCCESS;
  int *r_count = NULL, *s_count = NULL, *r_index = NULL, *s_index = NULL;
  int phase_scount, phase_rcount, num_phases, inbi, vdest;
  size_t w_size, segsize, segcount;
  uint32_t vrank;
  char *tmp_send = NULL, *tmp_recv = NULL;
  char *inbuf[2] = {NULL, NULL}, *inbuf_free[2] = {NULL, NULL};
  ptrdiff_t lb, extent, true_extent, gap = 0, inbuf_size;
  MPI_Request reqs[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};

  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  steps = hibit(size, (int)(sizeof(size) * CHAR_BIT) - 1);
  if(steps == -1) {
      return MPI_ERR_ARG;
  }
  int adjsize = 1 << steps;  // Largest power of two <= size

  // Number of nodes that exceed the largest power of two less than or equal to size
  int extra_ranks = size - adjsize;
  int is_power_of_two = (size & (size - 1)) == 0;


  MPI_Type_get_extent(dtype, &lb, &extent);
  MPI_Type_get_true_extent(dtype, &gap, &true_extent);

  segsize = bine_allreduce_segsize;
  segcount = segsize / extent;      // Number of elements in a segment
  if (segsize == 0) {
    segcount = count;
    segsize = segcount * extent;
  }

  // Allocate temporary buffer for send/recv and reduce operations
  inbuf_size = (segcount < (count >> 1)) ?
                  true_extent + extent * segcount : true_extent + extent * (count >> 1);
  inbuf_free[0] = (char *)malloc(inbuf_size);
  inbuf_free[1] = (char *)malloc(inbuf_size);
  if(NULL == inbuf_free[0] || NULL == inbuf_free[1]) {
    err = MPI_ERR_NO_MEM;
    goto cleanup_and_return;
  }
  inbuf[0] = inbuf_free[0] - gap;
  inbuf[1] = inbuf_free[1] - gap;

  // First part of computation to get a 2^n number of nodes.
  // What happens is that first #extra_rank even nodes sends their
  // data to the successive node and do not partecipate in the general
  // collective call operation.
  // All the nodes that do not stop their computation will receive an alias
  // called new_node, used to calculate their correct destination wrt this
  // new "cut" topology.
  int new_rank = rank, loop_flag = 0;
  if(rank < (2 * extra_ranks)) {
    if(0 == (rank % 2)) {
      err = MPI_Send(sbuf, count, dtype, (rank + 1), 0, comm);
      if(MPI_SUCCESS != err) { goto cleanup_and_return; }
      loop_flag = 1;
    } else {
      // TODO: Pay attention to commuitativity of the operation
      err = MPI_Recv(rbuf, count, dtype, (rank - 1), 0, comm, MPI_STATUS_IGNORE);
      if(MPI_SUCCESS != err) { goto cleanup_and_return; }
      MPI_Reduce_local((char *) sbuf, (char *) rbuf, count, dtype, op);
      new_rank = rank >> 1;
    }
  } else {
    new_rank = rank - extra_ranks;
    // Copy into receive_buffer content of send_buffer to not produce
    // side effects on send_buffer
    if(sbuf != MPI_IN_PLACE) {
      err = copy_buffer((char *)sbuf, (char *) rbuf, count, dtype);
      if(MPI_SUCCESS != err) { goto cleanup_and_return; }
    }
  }

  // Here the actual allreduce starts
  r_index = (int *) malloc(sizeof(*r_index) * steps);
  s_index = (int *) malloc(sizeof(*s_index) * steps);
  r_count = (int *) malloc(sizeof(*r_count) * steps);
  s_count = (int *) malloc(sizeof(*s_count) * steps);
  if(NULL == r_index || NULL == s_index || NULL == r_count || NULL == s_count) {
    err = MPI_ERR_NO_MEM;
    goto cleanup_and_return;
  }

  // Only the remaining ranks will do the following part
  if(!loop_flag){
    // Reduce-Scatter phase
    w_size = count;
    s_index[0] = r_index[0] = 0;
    vrank = remap_rank((uint32_t) adjsize, (uint32_t) new_rank);

    for(step = 0; step < steps; step++) {
      vdest = pi(new_rank, step, adjsize);

      dest = is_power_of_two ?
                vdest :
                (vdest < extra_ranks) ?
                (vdest << 1) + 1 : vdest + extra_ranks;

      //printf("Rank %d step %d dest %d vdest %d vrank %d\n",
      //       rank, step, dest, vdest, vrank);
      // TODO: dest or vdest as param?
      vdest = remap_rank((uint32_t) adjsize, (uint32_t) vdest);

      if(vrank < vdest) {
        r_count[step] = w_size / 2;
        s_count[step] = w_size - r_count[step];
        s_index[step] = r_index[step] + r_count[step];
      } else {
        s_count[step] = w_size / 2;
        r_count[step] = w_size - s_count[step];
        r_index[step] = s_index[step] + s_count[step];
      }

      num_phases = (r_count[step] > s_count[step]) ?
                      (int) (r_count[step] / segcount) :
                      (int) (s_count[step] / segcount);

      phase_scount = (s_count[step] > segcount) ? segcount : s_count[step];
      phase_rcount = (r_count[step] > segcount) ? segcount : r_count[step];

      inbi = 0;
      err = MPI_Irecv(inbuf[inbi], phase_rcount, dtype, dest, 0, comm, &reqs[inbi]);
      if(MPI_SUCCESS != err) { goto cleanup_and_return; }

      tmp_send = (char *)rbuf + s_index[step] * extent;
      err = MPI_Send(tmp_send, phase_scount, dtype, dest, 0, comm);
      if(MPI_SUCCESS != err) { goto cleanup_and_return; }

      tmp_recv = (char *)rbuf + r_index[step] * extent;

      for(int phase = 0; phase < num_phases - 1; phase++){
        char *tmp_recv_phase = tmp_recv + (ptrdiff_t)(phase * phase_rcount * extent);
        char *tmp_send_phase = tmp_send + (ptrdiff_t)((phase + 1) * phase_scount * extent);
        inbi = inbi ^ 0x1;

        err = MPI_Irecv(inbuf[inbi], phase_rcount, dtype, dest, 0, comm, &reqs[inbi]);
        if(MPI_SUCCESS != err) { goto cleanup_and_return; }

        err = MPI_Wait(&reqs[inbi ^ 0x1], MPI_STATUS_IGNORE);
        if(MPI_SUCCESS != err) { goto cleanup_and_return; }

        err = MPI_Reduce_local(inbuf[inbi ^ 0x1], tmp_recv_phase, phase_rcount, dtype, op);
        if(MPI_SUCCESS != err) { goto cleanup_and_return; }

        err = MPI_Send(tmp_send_phase, phase_scount, dtype, dest, 0, comm);
        if(MPI_SUCCESS != err) { goto cleanup_and_return; }
      }

      err = MPI_Wait(&reqs[inbi], MPI_STATUS_IGNORE);
      if(MPI_SUCCESS != err) { goto cleanup_and_return; }

      if(num_phases != 0){
        tmp_recv += (ptrdiff_t)((num_phases - 1) * phase_rcount * extent);
      }
      err = MPI_Reduce_local(inbuf[inbi], tmp_recv, phase_rcount, dtype, op);
      if(MPI_SUCCESS != err) { goto cleanup_and_return; }

      if(step + 1 < steps) {
        r_index[step + 1] = r_index[step];
        s_index[step + 1] = r_index[step];
        w_size = r_count[step];
      }
    }

    // Allgather phase
    for(step = steps - 1; step >= 0; step--) {
      vdest = pi(new_rank, step, adjsize);

      dest = is_power_of_two ?
                vdest :
                (vdest < extra_ranks) ?
                (vdest << 1) + 1 : vdest + extra_ranks;

      tmp_send = (char *)rbuf + r_index[step] * extent;
      tmp_recv = (char *)rbuf + s_index[step] * extent;
      err = MPI_Sendrecv(tmp_send, r_count[step], dtype, dest, 0,
                        tmp_recv, s_count[step], dtype, dest, 0,
                        comm, MPI_STATUS_IGNORE);
      if(MPI_SUCCESS != err) { goto cleanup_and_return; }
    }
  }

  // Final results is sent to nodes that are not included in general computation
  // (general computation loop requires 2^n nodes).
  if(rank < (2 * extra_ranks)){
    if(!loop_flag){
      err = MPI_Send(rbuf, count, dtype, (rank - 1), 0, comm);
      if(MPI_SUCCESS != err) { goto cleanup_and_return; }
    } else {
      err = MPI_Recv(rbuf, count, dtype, (rank + 1), 0, comm, MPI_STATUS_IGNORE);
      if(MPI_SUCCESS != err) { goto cleanup_and_return; }
    }
  }

  free(inbuf_free[0]);
  free(inbuf_free[1]);
  free(r_index);
  free(s_index);
  free(r_count);
  free(s_count);
  return MPI_SUCCESS;

cleanup_and_return:
  if(NULL != inbuf_free[0]) free(inbuf_free[0]);
  if(NULL != inbuf_free[1]) free(inbuf_free[1]);
  if(NULL != r_index)       free(r_index);
  if(NULL != s_index)       free(s_index);
  if(NULL != r_count)       free(r_count);
  if(NULL != s_count)       free(s_count);
  return err;
}

// WARNING: Old version, only working for powers of two number of processes
//int allreduce_bine_bdw_remap_segmented(const void *send_buf, void *recv_buf, size_t count,
//                                        MPI_Datatype dtype, MPI_Op op, MPI_Comm comm){
//  int size, rank, dest, steps, step, err = MPI_SUCCESS;
//  int *r_count = NULL, *s_count = NULL, *r_index = NULL, *s_index = NULL;
//  int phase_scount, phase_rcount, num_phases, inbi;
//  size_t w_size, segsize, segcount;
//  uint32_t vrank, vdest;
//  char *tmp_send = NULL, *tmp_recv = NULL;
//  char *inbuf[2] = {NULL, NULL}, *inbuf_free[2] = {NULL, NULL};
//  ptrdiff_t lb, extent, true_extent, gap = 0, inbuf_size;
//  MPI_Request reqs[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
//
//  MPI_Comm_size(comm, &size);
//  MPI_Comm_rank(comm, &rank);
//
//  // Does not support non-power-of-two or negative sizes
//  steps = log_2(size);
//  if(!is_power_of_two(size) || steps == -1) {
//    return MPI_ERR_ARG;
//  }
//
//  MPI_Type_get_extent(dtype, &lb, &extent);
//  MPI_Type_get_true_extent(dtype, &gap, &true_extent);
//
//  segsize = bine_allreduce_segsize;
//  segcount = segsize / extent;      // Number of elements in a segment
//  if (segsize == 0) {
//    segcount = count / size;
//    segsize = segcount * extent;
//  }
//
//  // Allocate temporary buffer for send/recv and reduce operations
//  inbuf_size = (segcount < (count >> 1)) ?
//                  true_extent + extent * segcount : true_extent + extent * (count >> 1);
//  inbuf_free[0] = (char *)malloc(inbuf_size);
//  inbuf_free[1] = (char *)malloc(inbuf_size);
//  if(NULL == inbuf_free[0] || NULL == inbuf_free[1]) {
//    err = MPI_ERR_NO_MEM;
//    goto cleanup_and_return;
//  }
//  inbuf[0] = inbuf_free[0] - gap;
//  inbuf[1] = inbuf_free[1] - gap;
//
//  // Copy into receive_buffer content of send_buffer to not produce
//  // side effects on send_buffer
//  if(send_buf != MPI_IN_PLACE) {
//    err = copy_buffer((char *)send_buf, (char *)recv_buf, count, dtype);
//    if(MPI_SUCCESS != err) { goto cleanup_and_return; }
//  }
//
//  r_index = (int *) malloc(sizeof(*r_index) * steps);
//  s_index = (int *) malloc(sizeof(*s_index) * steps);
//  r_count = (int *) malloc(sizeof(*r_count) * steps);
//  s_count = (int *) malloc(sizeof(*s_count) * steps);
//  if(NULL == r_index || NULL == s_index || NULL == r_count || NULL == s_count) {
//    err = MPI_ERR_NO_MEM;
//    goto cleanup_and_return;
//  }
//
//  // Reduce-Scatter phase
//  w_size = count;
//  s_index[0] = r_index[0] = 0;
//  vrank = remap_rank((uint32_t) size, (uint32_t) rank);
//  for(step = 0; step < steps; step++) {
//    dest = pi(rank, step, size);
//    vdest = remap_rank((uint32_t) size, (uint32_t) dest);
//
//    if(vrank < vdest) {
//      r_count[step] = w_size / 2;
//      s_count[step] = w_size - r_count[step];
//      s_index[step] = r_index[step] + r_count[step];
//    } else {
//      s_count[step] = w_size / 2;
//      r_count[step] = w_size - s_count[step];
//      r_index[step] = s_index[step] + s_count[step];
//    }
//
//    num_phases = (r_count[step] > s_count[step]) ?
//                    (int) (r_count[step] / segcount) :
//                    (int) (s_count[step] / segcount);
//
//    phase_scount = (s_count[step] > segcount) ? segcount : s_count[step];
//    phase_rcount = (r_count[step] > segcount) ? segcount : r_count[step];
//
//    inbi = 0;
//    err = MPI_Irecv(inbuf[inbi], phase_rcount, dtype, dest, 0, comm, &reqs[inbi]);
//    if(MPI_SUCCESS != err) { goto cleanup_and_return; }
//
//    tmp_send = (char *)recv_buf + s_index[step] * extent;
//    err = MPI_Send(tmp_send, phase_scount, dtype, dest, 0, comm);
//    if(MPI_SUCCESS != err) { goto cleanup_and_return; }
//
//    tmp_recv = (char *)recv_buf + r_index[step] * extent;
//
//    for(int phase = 0; phase < num_phases - 1; phase++){
//      char *tmp_recv_phase = tmp_recv + (ptrdiff_t)(phase * phase_rcount * extent);
//      char *tmp_send_phase = tmp_send + (ptrdiff_t)((phase + 1) * phase_scount * extent);
//      inbi = inbi ^ 0x1;
//
//      err = MPI_Irecv(inbuf[inbi], phase_rcount, dtype, dest, 0, comm, &reqs[inbi]);
//      if(MPI_SUCCESS != err) { goto cleanup_and_return; }
//
//      err = MPI_Wait(&reqs[inbi ^ 0x1], MPI_STATUS_IGNORE);
//      if(MPI_SUCCESS != err) { goto cleanup_and_return; }
//
//      err = MPI_Reduce_local(inbuf[inbi ^ 0x1], tmp_recv_phase, phase_rcount, dtype, op);
//      if(MPI_SUCCESS != err) { goto cleanup_and_return; }
//
//      err = MPI_Send(tmp_send_phase, phase_scount, dtype, dest, 0, comm);
//      if(MPI_SUCCESS != err) { goto cleanup_and_return; }
//    }
//
//    err = MPI_Wait(&reqs[inbi], MPI_STATUS_IGNORE);
//    if(MPI_SUCCESS != err) { goto cleanup_and_return; }
//
//    if(num_phases != 0){
//      tmp_recv += (ptrdiff_t)((num_phases - 1) * phase_rcount * extent);
//    }
//    err = MPI_Reduce_local(inbuf[inbi], tmp_recv, phase_rcount, dtype, op);
//    if(MPI_SUCCESS != err) { goto cleanup_and_return; }
//
//    if(step + 1 < steps) {
//      r_index[step + 1] = r_index[step];
//      s_index[step + 1] = r_index[step];
//      w_size = r_count[step];
//    }
//  }
//
//  // Allgather phase
//  for(step = steps - 1; step >= 0; step--) {
//    dest = pi(rank, step, size);
//
//    tmp_send = (char *)recv_buf + r_index[step] * extent;
//    tmp_recv = (char *)recv_buf + s_index[step] * extent;
//    err = MPI_Sendrecv(tmp_send, r_count[step], dtype, dest, 0,
//                       tmp_recv, s_count[step], dtype, dest, 0,
//                       comm, MPI_STATUS_IGNORE);
//    if(MPI_SUCCESS != err) { goto cleanup_and_return; }
//  }
//
//  free(inbuf_free[0]);
//  free(inbuf_free[1]);
//  free(r_index);
//  free(s_index);
//  free(r_count);
//  free(s_count);
//  return MPI_SUCCESS;
//
//cleanup_and_return:
//  if(NULL != inbuf_free[0]) free(inbuf_free[0]);
//  if(NULL != inbuf_free[1]) free(inbuf_free[1]);
//  if(NULL != r_index)       free(r_index);
//  if(NULL != s_index)       free(s_index);
//  if(NULL != r_count)       free(r_count);
//  if(NULL != s_count)       free(s_count);
//  return err;
//}

#ifdef CUDA_AWARE
// TODO: add allreduce_bine_bdw_hier_gpu
#endif

// NOTE: Not fully implemented
//
// int allreduce_bine_bdw_segmented(const void *send_buf, void *recv_buf, size_t count, MPI_Datatype dtype,
//                                   MPI_Op op, MPI_Comm comm, uint32_t segsize)
// { 
//   int size, rank, dest, steps;
//   int k, inbi, split_rank;
//   size_t small_block_count, large_block_count, max_seg_count;
//   ptrdiff_t lb, extent, gap;
//   char *tmp_send = NULL, *tmp_recv = NULL, *tmp_buf[2] = {NULL, NULL};
//   MPI_Request reqs[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
//   MPI_Comm_size(comm, &size);
//   MPI_Comm_rank(comm, &rank);
//
//   steps = log_2(size);
//   if(!is_power_of_two(size) || steps == -1) {
//     return MPI_ERR_ARG;
//   }
//
//   COLL_BASE_COMPUTE_BLOCKCOUNT(count, size, split_rank, large_block_count, small_block_count);
//
//   int typelng;
//   size_t seg_count = large_block_count;
//   MPI_Type_size(dtype, &typelng);
//   COLL_BASE_COMPUTED_SEGCOUNT(segsize, typelng, seg_count);
//
//   int num_phases = (int) (large_block_count / seg_count);
//   if ((large_block_count % seg_count) != 0) num_phases++;
//
//   COLL_BASE_COMPUTE_BLOCKCOUNT(large_block_count, num_phases, inbi, max_seg_count, k);
//   MPI_Type_get_extent(dtype, &lb, &extent);
//   ptrdiff_t max_real_segsize = datatype_span(dtype, max_seg_count, &gap);
//
//   /* Allocate and initialize temporary buffers */
//   tmp_buf[0] = (char *) malloc(max_real_segsize);
//   tmp_buf[1] = (char *) malloc(max_real_segsize);
//
//   // Copy into receive_buffer content of send_buffer to not produce side effects on send_buffer
//   if (send_buf != MPI_IN_PLACE) {
//     copy_buffer((char *)send_buf, (char *)recv_buf, count, dtype);
//   }
//   
//   int *s_bitmap = NULL, *r_bitmap = NULL;
//   int bitmap_offset = 0;
//   s_bitmap = (int *) calloc(size * steps, sizeof(int));
//   r_bitmap = (int *) calloc(size * steps, sizeof(int));
//
//   int step, s_ind, r_ind, s_split_seg, r_split_seg, seg_ind;
//   size_t s_block_count, r_block_count;
//   size_t s_large_seg_count, s_small_seg_count, r_large_seg_count, r_small_seg_count;
//   size_t r_count, s_count, prev_r_count;
//   ptrdiff_t s_block_offset, r_block_offset, r_seg_offset, s_seg_offset;
//   // Reduce-Scatter phase
//   for (step = 0; step < steps; step++) {
//     dest = pi(rank, step, size);
//     
//     get_indexes(rank, step, steps, size, s_bitmap + bitmap_offset);
//     get_indexes(dest, step, steps, size, r_bitmap + bitmap_offset);
//
//     s_ind = 0;
//     r_ind = 0;
//     while (s_ind < size && r_ind < size) {
//       // Navigate send and recv bitmap to find first block to send and recv
//       while (s_ind < size && s_bitmap[s_ind + bitmap_offset] != 1) { s_ind++;}
//       while (r_ind < size && r_bitmap[r_ind + bitmap_offset] != 1) { r_ind++;}
//       
//       // Scatter reduce the block
//       if (r_ind < size && s_ind < size) {
//         inbi = 0;
//         
//         // For each one of send block and recv block calculate:
//         // - block_count: number of elements in the block
//         // - large_seg_count, small_seg_count: number of elements in big and small segments
//         // - split_seg: indicates the first of the small segments
//         s_block_count = (s_ind < split_rank) ? large_block_count : small_block_count;
//         r_block_count = (r_ind < split_rank) ? large_block_count : small_block_count;
//         COLL_BASE_COMPUTE_BLOCKCOUNT(s_block_count, num_phases, s_split_seg, s_large_seg_count, s_small_seg_count);
//         COLL_BASE_COMPUTE_BLOCKCOUNT(r_block_count, num_phases, r_split_seg, r_large_seg_count, r_small_seg_count);
//
//         // Calculate the offset of the send and recv block wrt buffer (in bytes)
//         s_block_offset = (s_ind < split_rank) ? ((ptrdiff_t) s_ind * (ptrdiff_t) large_block_count) * extent :
//                           ((ptrdiff_t) s_ind * (ptrdiff_t) small_block_count + split_rank) * extent;
//         r_block_offset = (r_ind < split_rank) ? ((ptrdiff_t) r_ind * (ptrdiff_t) large_block_count) * extent : 
//                           ((ptrdiff_t) r_ind * (ptrdiff_t) small_block_count + split_rank) * extent;
//
//         // Post an irecv for the first segment
//         MPI_Irecv(tmp_buf[inbi], r_large_seg_count, dtype, dest, 0, comm, &reqs[inbi]);
//         
//         // Send the first segment
//         tmp_send = (char *)recv_buf + s_block_offset;
//         MPI_Send(tmp_send, s_large_seg_count, dtype, dest, 0, comm);
//         
//         for(seg_ind = 1; seg_ind < num_phases; seg_ind++){
//           inbi = inbi ^ 0x1;
//           
//           // Post an irecv for the current segment (i.e. seg[seg_ind])
//           r_count = (seg_ind < r_split_seg) ? r_large_seg_count: r_small_seg_count;
//           MPI_Irecv(tmp_buf[inbi], r_count, dtype, dest, 0, comm, &reqs[inbi]);
//
//           // Wait for the arrival of the previous block
//           MPI_Wait(&reqs[inbi ^ 0x1], MPI_STATUS_IGNORE);
//           
//           // Calculate the offset of the recv segment wrt the start of the block (in bytes) and the number of elements of the previous recv
//           r_seg_offset = (seg_ind - 1 < r_split_seg) ? ((ptrdiff_t) (seg_ind - 1) * (ptrdiff_t) r_large_seg_count) * extent :
//                                                        (((ptrdiff_t) (seg_ind - 1) * (ptrdiff_t) r_small_seg_count) + (ptrdiff_t) r_split_seg) * extent;
//           tmp_recv = (char *) recv_buf + (r_block_offset + r_seg_offset);
//           prev_r_count = ((seg_ind - 1) < r_split_seg) ? r_large_seg_count : r_small_seg_count;
//
//           // Reduce the previous block
//           MPI_Reduce_local(tmp_buf[inbi ^ 0x1], tmp_recv, prev_r_count, dtype, op);
//
//           // Calculate offset and count of the current block and send it
//           s_seg_offset = (seg_ind < s_split_seg) ? ((ptrdiff_t) seg_ind * (ptrdiff_t) s_large_seg_count) * extent :
//                                                    (((ptrdiff_t) seg_ind * (ptrdiff_t) s_small_seg_count) + (ptrdiff_t) s_split_seg) * extent;
//           tmp_send += s_seg_offset;
//           s_count = (seg_ind < s_split_seg) ? s_large_seg_count: s_small_seg_count;
//           MPI_Send(tmp_send, s_count, dtype, dest, 0, comm);
//         }
//         // Wait for the last segment to arrive
//         MPI_Wait(&reqs[inbi], MPI_STATUS_IGNORE);
//
//         // Reduce the last segment
//         r_seg_offset = (((ptrdiff_t) (num_phases - 1) * (ptrdiff_t) r_small_seg_count) + (ptrdiff_t) r_split_seg) * extent;
//         tmp_recv = (char*) recv_buf + (r_block_offset + r_seg_offset);
//         MPI_Reduce_local(tmp_buf[inbi], tmp_recv, r_small_seg_count, dtype, op);
//       }
//       s_ind++;
//       r_ind++;
//     }
//     bitmap_offset += size;
//   }
//   
//   
//   // Allgather phase
//   MPI_Datatype s_ind_dtype = MPI_DATATYPE_NULL, r_ind_dtype = MPI_DATATYPE_NULL;
//   int *block_len, *disp;
//   block_len = (int *)malloc(size * sizeof(int));
//   disp = (int *)malloc(size * sizeof(int));
//   bitmap_offset -= size;
//   size_t w_size = 1;
//   for(step = steps - 1; step >= 0; step--) {
//     dest = pi(rank, step, size);
//
//     libbine_indexed_datatype(&s_ind_dtype, s_bitmap + bitmap_offset, size, w_size,
//                               small_block_count, split_rank, dtype, block_len, disp);
//     libbine_indexed_datatype(&r_ind_dtype, r_bitmap + bitmap_offset, size, w_size,
//                               small_block_count, split_rank, dtype, block_len, disp);
//
//     MPI_Sendrecv(recv_buf, 1, r_ind_dtype, dest, 0, recv_buf, 1, s_ind_dtype, dest, 0, comm, MPI_STATUS_IGNORE);
//
//     MPI_Type_free(&s_ind_dtype);
//     s_ind_dtype = MPI_DATATYPE_NULL;
//     MPI_Type_free(&r_ind_dtype);
//     r_ind_dtype = MPI_DATATYPE_NULL;
//
//     w_size <<= 1;
//     bitmap_offset -= size;
//   }
//  
//   free(s_bitmap);
//   free(r_bitmap);
//
//   free(block_len);
//   free(disp);
//
//   free(tmp_buf[0]);
//   free(tmp_buf[1]);
//
//   return MPI_SUCCESS;
// }

