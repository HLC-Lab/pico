#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>

#include "libswing.h"
#include "libswing_utils.h"
#include "libswing_utils_bitmaps.h"



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
  if (1 == size) {
    if (MPI_IN_PLACE != sbuf) {
      ret = copy_buffer((char *) sbuf, (char *) rbuf, count, dtype);
      if (ret < 0) { line = __LINE__; goto error_hndl; }
    }
    return MPI_SUCCESS;
  }

  /* Allocate and initialize temporary send buffer */
  span = datatype_span(dtype, count, &gap);

  inplacebuf_free = (char*) malloc(span);
  if (NULL == inplacebuf_free) { ret = -1; line = __LINE__; goto error_hndl; }
  inplacebuf = inplacebuf_free - gap;

  if (MPI_IN_PLACE == sbuf) {
      ret = copy_buffer((char*)rbuf, inplacebuf, count, dtype);
      if (ret < 0) { line = __LINE__; goto error_hndl; }
  } else {
      ret = copy_buffer((char*)sbuf, inplacebuf, count, dtype);
      if (ret < 0) { line = __LINE__; goto error_hndl; }
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
  if (rank <  (2 * extra_ranks)) {
    if (0 == (rank % 2)) {
      ret = MPI_Send(tmpsend, count, dtype, (rank + 1), 0, comm);
      if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
      newrank = -1;
    } else {
      ret = MPI_Recv(tmprecv, count, dtype, (rank - 1), 0, comm,
                  MPI_STATUS_IGNORE);
      if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
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
  for (distance = 0x1; distance < adjsize; distance <<=1) {
    if (newrank < 0) break;
    /* Determine remote node */
    newremote = newrank ^ distance;
    remote = (newremote < extra_ranks) ? (newremote * 2 + 1) : (newremote + extra_ranks);
   
    /* Exchange the data */
    ret = MPI_Sendrecv(tmpsend, count, dtype, remote, 0,
                       tmprecv, count, dtype, remote, 0,
                       comm, MPI_STATUS_IGNORE);
    if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }

    // reduction((int64_t *) tmprecv, (int64_t *) tmpsend, count);
    MPI_Reduce_local((char *) tmprecv, (char *) tmpsend, count, dtype, op);
  }

  /* Handle non-power-of-two case:
     - Odd ranks less than 2 * extra_ranks send result from tmpsend to
     (rank - 1)
     - Even ranks less than 2 * extra_ranks receive result from (rank + 1)
  */
  if (rank < (2 * extra_ranks)) {
    if (0 == (rank % 2)) {
      ret = MPI_Recv(rbuf, count, dtype, (rank + 1), 0, comm, MPI_STATUS_IGNORE);
      if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
      tmpsend = (char*)rbuf;
    } else {
      ret = MPI_Send(tmpsend, count, dtype, (rank - 1), 0, comm);
      if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
    }
  }

  /* Ensure that the final result is in rbuf */
  if (tmpsend != rbuf) {
    ret = copy_buffer(tmpsend, (char*)rbuf, count, dtype);
    if (ret < 0) { line = __LINE__; goto error_hndl; }
  }

  if (NULL != inplacebuf_free) free(inplacebuf_free);
  return MPI_SUCCESS;

  error_hndl:
    SWING_DEBUG_PRINT("\n%s:%4d\tRank %d Error occurred %d\n", __FILE__, line, rank, ret);
    (void)line;  // silence compiler warning
    if (NULL != inplacebuf_free) free(inplacebuf_free);
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
  if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
  ret = MPI_Comm_size(comm, &size);
  if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }

  // if (rank == 0) {
  //   printf("4: RING\n");
  //   fflush(stdout);
  // }

  /* Special case for size == 1 */
  if (1 == size) {
    if (MPI_IN_PLACE != sbuf) {
      ret = copy_buffer((char *) sbuf, (char *) rbuf, count, dtype);
      if (ret < 0) { line = __LINE__; goto error_hndl; }
    }
    return MPI_SUCCESS;
  }

  /* Special case for count less than size - use recursive doubling */
  if (count < (size_t) size) {
    return (allreduce_recursivedoubling(sbuf, rbuf, count, dtype, op, comm));
  }

  /* Allocate and initialize temporary buffers */
  ret = MPI_Type_get_extent(dtype, &lb, &extent);
  if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
  ret = MPI_Type_get_true_extent(dtype, &true_lb, &true_extent);
  if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }

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
  if (NULL == inbuf[0]) { ret = -1; line = __LINE__; goto error_hndl; }
  if (size > 2) {
    inbuf[1] = (char*)malloc(max_real_segsize);
    if (NULL == inbuf[1]) { ret = -1; line = __LINE__; goto error_hndl; }
  }

  /* Handle MPI_IN_PLACE */
  if (MPI_IN_PLACE != sbuf) {
    ret = copy_buffer((char *)sbuf, (char *) rbuf, count, dtype);
    if (ret < 0) { line = __LINE__; goto error_hndl; }
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
  if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
  /* Send first block (my block) to the neighbor on the right */
  block_offset = ((rank < split_rank)?
          ((ptrdiff_t)rank * (ptrdiff_t)early_segcount) :
          ((ptrdiff_t)rank * (ptrdiff_t)late_segcount + split_rank));
  block_count = ((rank < split_rank)? early_segcount : late_segcount);
  tmpsend = ((char*)rbuf) + block_offset * extent;
  ret = MPI_Send(tmpsend, block_count, dtype, send_to, 0, comm);
  if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }

  for (k = 2; k < size; k++) {
    const int prevblock = (rank + size - k + 1) % size;

    inbi = inbi ^ 0x1;

    /* Post irecv for the current block */
    ret = MPI_Irecv(inbuf[inbi], max_segcount, dtype, recv_from, 0, comm, &reqs[inbi]);
    if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }

    /* Wait on previous block to arrive */
    ret = MPI_Wait(&reqs[inbi ^ 0x1], MPI_STATUS_IGNORE);
    if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }

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
    if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
  }

  /* Wait on the last block to arrive */
  ret = MPI_Wait(&reqs[inbi], MPI_STATUS_IGNORE);
  if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }

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
  for (k = 0; k < size - 1; k++) {
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
    if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl;}

  }

  if (NULL != inbuf[0]) free(inbuf[0]);
  if (NULL != inbuf[1]) free(inbuf[1]);

  return MPI_SUCCESS;

 error_hndl:
  SWING_DEBUG_PRINT("\n%s:%4d\tRank %d Error occurred %d\n\n", __FILE__, line, rank, ret);
  MPI_Request_free(&reqs[0]);
  MPI_Request_free(&reqs[1]);
  (void)line;  // silence compiler warning
  if (NULL != inbuf[0]) free(inbuf[0]);
  if (NULL != inbuf[1]) free(inbuf[1]);
  return ret;
}

int allreduce_swing_lat(const void *sbuf, void *rbuf, size_t count, MPI_Datatype dtype, MPI_Op op, MPI_Comm comm) {
  int rank, size;
  int ret, line; // for error handling
  char *tmpsend, *tmprecv, *inplacebuf_free = NULL;
  ptrdiff_t span, gap = 0;
  

  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  // Special case for size == 1
  if (1 == size) {
    if (MPI_IN_PLACE != sbuf) {
      ret = copy_buffer((char *) sbuf, (char *) rbuf, count, dtype);
      if (ret < 0) { line = __LINE__; goto error_hndl; }
    }
    return MPI_SUCCESS;
  }
  
  // Allocate and initialize temporary send buffer
  span = datatype_span(dtype, count, &gap);
  inplacebuf_free = (char*) malloc(span + gap);
  char *inplacebuf = inplacebuf_free + gap;

  // Copy content from sbuffer to inplacebuf
  if (MPI_IN_PLACE == sbuf) {
      ret = copy_buffer((char*)rbuf, inplacebuf, count, dtype);
      if (ret < 0) { line = __LINE__; goto error_hndl; }
  } else {
      ret = copy_buffer((char*)sbuf, inplacebuf, count, dtype);
      if (ret < 0) { line = __LINE__; goto error_hndl; }
  }

  tmpsend = inplacebuf;
  tmprecv = (char*) rbuf;
  
  // Determine nearest power of two less than or equal to size
  // and return an error if size is 0
  int steps = hibit(size, (int)(sizeof(size) * CHAR_BIT) - 1);
  if (steps == -1) {
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
  if (rank <  (2 * extra_ranks)) {
    if (0 == (rank % 2)) {
      ret = MPI_Send(tmpsend, count, dtype, (rank + 1), 0, comm);
      if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
      loop_flag = 1;
    } else {
      ret = MPI_Recv(tmprecv, count, dtype, (rank - 1), 0, comm, MPI_STATUS_IGNORE);
      if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
      MPI_Reduce_local((char *) tmprecv, (char *) tmpsend, count, dtype, op);
      new_rank = rank >> 1;
    }
  } else new_rank = rank - extra_ranks;
  
  
  // Actual allreduce computation for general cases
  int s, vdest, dest;
  for (s = 0; s < steps; s++){
    if (loop_flag) break;
    vdest = pi(new_rank, s, adjsize);

    dest = is_power_of_two ?
              vdest :
              (vdest < extra_ranks) ?
              (vdest << 1) + 1 : vdest + extra_ranks;

    ret = MPI_Sendrecv(tmpsend, count, dtype, dest, 0,
                       tmprecv, count, dtype, dest, 0,
                       comm, MPI_STATUS_IGNORE);
    if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
    
    MPI_Reduce_local((char *) tmprecv, (char *) tmpsend, count, dtype, op);
  }
  
  // Final results is sent to nodes that are not included in general computation
  // (general computation loop requires 2^n nodes).
  if (rank < (2 * extra_ranks)){
    if (!loop_flag){
      ret = MPI_Send(tmpsend, count, dtype, (rank - 1), 0, comm);
      if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
    } else {
      ret = MPI_Recv(rbuf, count, dtype, (rank + 1), 0, comm, MPI_STATUS_IGNORE);
      if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
      tmpsend = (char*)rbuf;
    }
  }

  if (tmpsend != rbuf) {
    ret = copy_buffer(tmpsend, (char*) rbuf, count, dtype);
    if (ret < 0) { line = __LINE__; goto error_hndl; }
  }

  free(inplacebuf_free);
  return MPI_SUCCESS;

  error_hndl:
    SWING_DEBUG_PRINT("\n%s:%4d\tRank %d Error occurred %d\n\n", __FILE__, line, rank, ret);
    (void)line;  // silence compiler warning
    if (NULL != inplacebuf_free) free(inplacebuf_free);
    return ret;
}

int allreduce_swing_bdw_static(const void *send_buf, void *recv_buf, size_t count,
                               MPI_Datatype dtype, MPI_Op op, MPI_Comm comm){
  int size, rank; 
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  // Find number of steps of scatter-reduce and allgather,
  // biggest power of two smaller or equal to size,
  // size of send_window (number of chunks to send/recv at each step)
  // and alias of the rank to be used if size != adj_size
  //Determine nearest power of two less than or equal to size
  int steps = hibit(size, (int) (sizeof(size) * CHAR_BIT) - 1);
  if (-1 == steps){
    return MPI_ERR_ARG;
  }
  int adjsize = 1 << steps;
  
  //WARNING: Assuming size is a pow of 2
  int vrank, vdest;
  vrank = rank;
  
  ptrdiff_t lb, extent, gap = 0;
  MPI_Type_get_extent(dtype, &lb, &extent);
  
  int split_rank;
  size_t small_blocks, big_blocks;
  COLL_BASE_COMPUTE_BLOCKCOUNT(count, adjsize, split_rank,
                               big_blocks, small_blocks);
  
  // Find the biggest power-of-two smaller than count to allocate
  // as few memory as necessary for buffers
  int n_pow = hibit((int) count, (int) (sizeof(count) * CHAR_BIT) -1); 
  size_t buf_count = 1 << n_pow;
  ptrdiff_t buf_size = datatype_span(dtype, buf_count, &gap);

  // Temporary target buffer for send operations and source buffer
  // for reduce and overwrite operations
  char *tmp_send = NULL, *tmp_recv = NULL;
  char *tmp_buf_raw = NULL, *tmp_buf;
  tmp_buf_raw = (char *)malloc(buf_size);
  tmp_buf = tmp_buf_raw - gap;

  // Copy into receive_buffer content of send_buffer to not produce
  // side effects on send_buffer
  if (send_buf != MPI_IN_PLACE) {
    copy_buffer((char *)recv_buf, (char *)send_buf, count, dtype);
  }
  
  const int *s_bitmap = NULL, *r_bitmap = NULL;
  if(get_static_bitmap(&s_bitmap, &r_bitmap, steps, size, rank) == -1){
    free(tmp_buf);
    return MPI_ERR_UNKNOWN;
  }
  
  int step, w_size = adjsize;
  size_t s_count, r_count;
  ptrdiff_t s_offset, r_offset;
  // Reduce-Scatter phase
  for (step = 0; step < steps; step++) {
    w_size >>= 1;
    vdest = pi(vrank, step, adjsize);

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
    MPI_Sendrecv(tmp_send, s_count, dtype, vdest, 0,
                 tmp_buf, r_count, dtype, vdest, 0,
                 comm, MPI_STATUS_IGNORE);
    
    tmp_recv = (char *) recv_buf + r_offset;
    MPI_Reduce_local(tmp_buf, tmp_recv, r_count, dtype, op);
  }
  
  // Allgather phase
  for(step = steps - 1; step >= 0; step--) {
    vdest = pi(vrank, step, adjsize);
    
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

    MPI_Sendrecv(tmp_recv, r_count, dtype, vdest, 0,
                 tmp_send, s_count, dtype, vdest, 0,
                 comm, MPI_STATUS_IGNORE);
    
    w_size <<= 1;
  }

  free(tmp_buf_raw);

  return MPI_SUCCESS;
}



int allreduce_rabenseifner( const void *sbuf, void *rbuf, size_t count,
                           MPI_Datatype dtype, MPI_Op op, MPI_Comm comm)
{
  int *rindex = NULL, *rcount = NULL, *sindex = NULL, *scount = NULL;
  int size, rank;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  // if (rank == 0) {
  //   printf("6: RABENSEIFNER\n");
  //   fflush(stdout);
  // }

  // Find number of steps of scatter-reduce and allgather,
  // biggest power of two smaller or equal to size,
  // size of send_window (number of chunks to send/recv at each step)
  // and alias of the rank to be used if size != adj_size
  //Determine nearest power of two less than or equal to size
  int steps = hibit(size, (int) (sizeof(size) * CHAR_BIT) - 1);
  if (-1 == steps){
    return MPI_ERR_ARG;
  }
  int adjsize = 1 << steps;

  int err = MPI_SUCCESS;
  ptrdiff_t lb, extent, gap = 0;
  MPI_Type_get_extent(dtype, &lb, &extent);
  // Find the biggest power-of-two smaller than count to allocate
  // as few memory as necessary for buffers
  int n_pow = hibit((int) count, (int) (sizeof(count) * CHAR_BIT) -1); 
  size_t buf_count = 1 << n_pow;
  ptrdiff_t buf_size = datatype_span(dtype, buf_count, &gap);

  /* Temporary buffer for receiving messages */
  char *tmp_buf = NULL;
  char *tmp_buf_raw = (char *)malloc(buf_size);
  if (NULL == tmp_buf_raw)
    return MPI_ERR_UNKNOWN;
  tmp_buf = tmp_buf_raw - gap;

  if (sbuf != MPI_IN_PLACE) {
    err = copy_buffer((char *)rbuf, (char *)sbuf, count, dtype);
    if (MPI_SUCCESS != err) { goto cleanup_and_return; }
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

  if (rank < 2 * nprocs_rem) {
    int count_lhalf = count / 2;
    int count_rhalf = count - count_lhalf;

    if (rank % 2 != 0) {
      /*
       * Odd process -- exchange with rank - 1
       * Send the left half of the input vector to the left neighbor,
       * Recv the right half of the input vector from the left neighbor
       */
      err = MPI_Sendrecv(rbuf, count_lhalf, dtype, rank - 1, 0,
                      (char *)tmp_buf + (ptrdiff_t)count_lhalf * extent,
                      count_rhalf, dtype, rank - 1, 0, comm, MPI_STATUS_IGNORE);
      if (MPI_SUCCESS != err) { goto cleanup_and_return; }

      /* Reduce on the right half of the buffers (result in rbuf) */
      MPI_Reduce_local((char *)tmp_buf + (ptrdiff_t)count_lhalf * extent,
                       (char *)rbuf + count_lhalf * extent, count_rhalf, dtype, op);

      /* Send the right half to the left neighbor */
      err = MPI_Send((char *)rbuf + (ptrdiff_t)count_lhalf * extent,
                     count_rhalf, dtype, rank - 1, 0, comm);
      if (MPI_SUCCESS != err) { goto cleanup_and_return; }

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
      if (MPI_SUCCESS != err) { goto cleanup_and_return; }

      /* Reduce on the right half of the buffers (result in rbuf) */
      MPI_Reduce_local(tmp_buf, rbuf, count_lhalf, dtype, op);

      /* Recv the right half from the right neighbor */
      err = MPI_Recv((char *)rbuf + (ptrdiff_t)count_lhalf * extent,
                  count_rhalf, dtype, rank + 1, 0, comm, MPI_STATUS_IGNORE);
      if (MPI_SUCCESS != err) { goto cleanup_and_return; }

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
  if (NULL == rindex || NULL == sindex || NULL == rcount || NULL == scount) {
    err = MPI_ERR_UNKNOWN;
    goto cleanup_and_return;
  }

  if (vrank != -1) {
    step = 0;
    wsize = count;
    sindex[0] = rindex[0] = 0;

    for (int mask = 1; mask < adjsize; mask <<= 1) {
      /*
       * On each iteration: rindex[step] = sindex[step] -- beginning of the
       * current window. Length of the current window is storded in wsize.
       */
      int vdest = vrank ^ mask;
      /* Translate vdest virtual rank to real rank */
      int dest = (vdest < nprocs_rem) ? vdest * 2 : vdest + nprocs_rem;

      if (rank < dest) {
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
      if (MPI_SUCCESS != err) { goto cleanup_and_return; }

      /* Local reduce: rbuf[] = tmp_buf[] <op> rbuf[] */
      MPI_Reduce_local((char *)tmp_buf + (ptrdiff_t)rindex[step] * extent,
               (char *)rbuf + (ptrdiff_t)rindex[step] * extent,
               rcount[step], dtype, op);

      /* Move the current window to the received message */
      if (step + 1 < steps) {
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

    for (int mask = adjsize >> 1; mask > 0; mask >>= 1) {
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
      if (MPI_SUCCESS != err) { goto cleanup_and_return; }
      step--;
    }
  }

  /*
   * Step 4. Send total result to excluded odd ranks.
   */
  if (rank < 2 * nprocs_rem) {
    if (rank % 2 != 0) {
      /* Odd process -- recv result from rank - 1 */
      err = MPI_Recv(rbuf, count, dtype, rank - 1,
                  0, comm, MPI_STATUS_IGNORE);
      if (MPI_SUCCESS != err) { goto cleanup_and_return; }

    } else {
      /* Even process -- send result to rank + 1 */
      err = MPI_Send(rbuf, count, dtype, rank + 1,
                  0, comm);
      if (MPI_SUCCESS != err) { goto cleanup_and_return; }
    }
  }

  cleanup_and_return:
  if (NULL != tmp_buf_raw)
    free(tmp_buf_raw);
  if (NULL != rindex)
    free(rindex);
  if (NULL != sindex)
    free(sindex);
  if (NULL != rcount)
    free(rcount);
  if (NULL != scount)
    free(scount);
  return err;
}
