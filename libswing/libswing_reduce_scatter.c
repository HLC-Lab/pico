#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>

#include "libswing.h"
#include "libswing_utils.h"
#include "libswing_utils_bitmaps.h"

int reduce_scatter_recursivehalving(const void *sbuf, void *rbuf, const int rcounts[],
                                    MPI_Datatype dtype, MPI_Op op, MPI_Comm comm)
{
  int i, rank, size, err = MPI_SUCCESS;
  int tmp_size, remain = 0, tmp_rank;
  size_t count;
  ptrdiff_t *disps = NULL;
  ptrdiff_t extent, lb, buf_size, gap = 0;
  char *recv_buf = NULL, *recv_buf_free = NULL;
  char *result_buf = NULL, *result_buf_free = NULL;

  err = MPI_Comm_size(comm, &size);
  err = MPI_Comm_rank(comm, &rank);


  /* Find displacements and the like */
  disps = (ptrdiff_t*) malloc(sizeof(ptrdiff_t) * size);
  if (NULL == disps) return MPI_ERR_NO_MEM;

  disps[0] = 0;
  for (i = 0; i < (size - 1); ++i) {
    disps[i + 1] = disps[i] + rcounts[i];
  }
  count = disps[size - 1] + rcounts[size - 1];

  /* short cut the trivial case */
  if (0 == count) {
    free(disps);
    return MPI_SUCCESS;
  }

  /* get datatype information */
  MPI_Type_get_extent(dtype, &lb, &extent);
  buf_size = datatype_span(dtype, count, &gap);

  /* Handle MPI_IN_PLACE */
  if (MPI_IN_PLACE == sbuf) {
    sbuf = rbuf;
  }

  /* Allocate temporary receive buffer. */
  recv_buf_free = (char*) malloc(buf_size);
  recv_buf = recv_buf_free - gap;
  if (NULL == recv_buf_free) {
    err = MPI_ERR_NO_MEM;
    goto cleanup;
  }

  /* allocate temporary buffer for results */
  result_buf_free = (char*) malloc(buf_size);
  result_buf = result_buf_free - gap;

  /* copy local buffer into the temporary results */
  err = copy_buffer_different_dt(sbuf, count, dtype, result_buf, count, dtype);
  if (MPI_SUCCESS != err) goto cleanup;

  /* figure out power of two mapping: grow until larger than
     comm size, then go back one, to get the largest power of
     two less than comm size */
  tmp_size = next_poweroftwo (size);
  tmp_size >>= 1;
  remain = size - tmp_size;

  /* If comm size is not a power of two, have the first "remain"
     procs with an even rank send to rank + 1, leaving a power of
     two procs to do the rest of the algorithm */
  if (rank < 2 * remain) {
    if ((rank & 1) == 0) {
      err = MPI_Send(result_buf, count, dtype, rank + 1, 0, comm);
      if (MPI_SUCCESS != err) goto cleanup;

      /* we don't participate from here on out */
      tmp_rank = -1;
    } else {
      err = MPI_Recv(recv_buf, count, dtype, rank - 1, 0, comm, MPI_STATUS_IGNORE);

      /* integrate their results into our temp results */
      MPI_Reduce_local(recv_buf, result_buf, count, dtype, op);

      /* adjust rank to be the bottom "remain" ranks */
      tmp_rank = rank / 2;
    }
  } else {
    /* just need to adjust rank to show that the bottom "even
       remain" ranks dropped out */
    tmp_rank = rank - remain;
  }

  /* For ranks not kicked out by the above code, perform the
     recursive halving */
  if (tmp_rank >= 0) {
    size_t *tmp_rcounts = NULL;
    ptrdiff_t *tmp_disps = NULL;
    int mask, send_index, recv_index, last_index;

    /* recalculate disps and rcounts to account for the
       special "remainder" processes that are no longer doing
       anything */
    tmp_rcounts = (size_t*) malloc(tmp_size * sizeof(size_t));
    if (NULL == tmp_rcounts) {
      err = MPI_ERR_NO_MEM;
      goto cleanup;
    }
    tmp_disps = (ptrdiff_t*) malloc(tmp_size * sizeof(ptrdiff_t));
    if (NULL == tmp_disps) {
      free(tmp_rcounts);
      err = MPI_ERR_NO_MEM;
      goto cleanup;
    }

    for (i = 0 ; i < tmp_size ; ++i) {
      if (i < remain) {
        /* need to include old neighbor as well */
        tmp_rcounts[i] = rcounts[i * 2 + 1] + rcounts[i * 2];
      } else {
        tmp_rcounts[i] = rcounts[i + remain];
      }
    }

    tmp_disps[0] = 0;
    for (i = 0; i < tmp_size - 1; ++i) {
      tmp_disps[i + 1] = tmp_disps[i] + tmp_rcounts[i];
    }

    /* do the recursive halving communication.  Don't use the
       dimension information on the communicator because I
       think the information is invalidated by our "shrinking"
       of the communicator */
    mask = tmp_size >> 1;
    send_index = recv_index = 0;
    last_index = tmp_size;
    while (mask > 0) {
      int tmp_peer, peer;
      size_t send_count, recv_count;
      MPI_Request request;

      tmp_peer = tmp_rank ^ mask;
      peer = (tmp_peer < remain) ? tmp_peer * 2 + 1 : tmp_peer + remain;

      /* figure out if we're sending, receiving, or both */
      send_count = recv_count = 0;
      if (tmp_rank < tmp_peer) {
        send_index = recv_index + mask;
        for (i = send_index ; i < last_index ; ++i) {
          send_count += tmp_rcounts[i];
        }
        for (i = recv_index ; i < send_index ; ++i) {
          recv_count += tmp_rcounts[i];
        }
      } else {
        recv_index = send_index + mask;
        for (i = send_index ; i < recv_index ; ++i) {
          send_count += tmp_rcounts[i];
        }
        for (i = recv_index ; i < last_index ; ++i) {
          recv_count += tmp_rcounts[i];
        }
      }

      /* actual data transfer.  Send from result_buf,
         receive into recv_buf */
      if (recv_count > 0) {
        err = MPI_Irecv(recv_buf + tmp_disps[recv_index] * extent,
                        recv_count, dtype, peer, 0, comm, &request);
        if (MPI_SUCCESS != err) {
          free(tmp_rcounts);
          free(tmp_disps);
          goto cleanup;
        }
      }
      if (send_count > 0) {
        err = MPI_Send(result_buf + tmp_disps[send_index] * extent,
                       send_count, dtype, peer, 0, comm);
        if (MPI_SUCCESS != err) {
          free(tmp_rcounts);
          free(tmp_disps);
          goto cleanup;
        }
      }

      /* if we received something on this step, push it into
         the results buffer */
      if (recv_count > 0) {
        err = MPI_Wait(&request, MPI_STATUS_IGNORE);
        if (MPI_SUCCESS != err) {
          free(tmp_rcounts);
          free(tmp_disps);
          goto cleanup;
        }

        MPI_Reduce_local(recv_buf + tmp_disps[recv_index] * extent,
                         result_buf + tmp_disps[recv_index] * extent,
                         recv_count, dtype, op);
      }

      /* update for next iteration */
      send_index = recv_index;
      last_index = recv_index + mask;
      mask >>= 1;
    }

    /* copy local results from results buffer into real receive buffer */
    if (0 != rcounts[rank]) {
      err = copy_buffer_different_dt(result_buf + disps[rank] * extent, rcounts[rank],
                                     dtype, rbuf, rcounts[rank], dtype);
      if (MPI_SUCCESS != err) {
        free(tmp_rcounts);
        free(tmp_disps);
        goto cleanup;
      }
    }

    free(tmp_rcounts);
    free(tmp_disps);
  }

  /* Now fix up the non-power of two case, by having the odd
     procs send the even procs the proper results */
  if (rank < (2 * remain)) {
    if ((rank & 1) == 0) {
      if (rcounts[rank]) {
        err = MPI_Recv(rbuf, rcounts[rank], dtype, rank + 1, 0, comm, MPI_STATUS_IGNORE);
        if (MPI_SUCCESS != err) goto cleanup;
      }
    } else {
      if (rcounts[rank - 1]) {
        err = MPI_Send(result_buf + disps[rank - 1] * extent,
                       rcounts[rank - 1], dtype, rank - 1, 0, comm);
        if (MPI_SUCCESS != err) goto cleanup;
      }
    }
  }

 cleanup:
  if (NULL != disps) free(disps);
  if (NULL != recv_buf_free) free(recv_buf_free);
  if (NULL != result_buf_free) free(result_buf_free);

  return err;
}


int reduce_scatter_ring( const void *sbuf, void *rbuf, const int rcounts[],
                        MPI_Datatype dtype, MPI_Op op, MPI_Comm comm)
{
  int ret, line, rank, size, i, k, recv_from, send_to;
  int inbi;
  size_t total_count, max_block_count;
  ptrdiff_t *displs = NULL;
  char *tmpsend = NULL, *tmprecv = NULL, *accumbuf = NULL, *accumbuf_free = NULL;
  char *inbuf_free[2] = {NULL, NULL}, *inbuf[2] = {NULL, NULL};
  ptrdiff_t extent, lb, max_real_segsize, dsize, gap = 0;
  MPI_Request reqs[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};

  ret = MPI_Comm_size(comm, &size);
  ret = MPI_Comm_rank(comm, &rank);

  /* Determine the maximum number of elements per node,
     corresponding block size, and displacements array.
  */
  displs = (ptrdiff_t*) malloc(size * sizeof(ptrdiff_t));
  if (NULL == displs) { ret = -1; line = __LINE__; goto error_hndl; }
  displs[0] = 0;
  total_count = rcounts[0];
  max_block_count = rcounts[0];
  for (i = 1; i < size; i++) {
    displs[i] = total_count;
    total_count += rcounts[i];
    if (max_block_count < rcounts[i]) max_block_count = rcounts[i];
  }

  /* Special case for size == 1 */
  if (1 == size) {
    if (MPI_IN_PLACE != sbuf) {
      ret = copy_buffer((char*) sbuf, (char*) rbuf,total_count, dtype);
      if (ret < 0) { line = __LINE__; goto error_hndl; }
    }
    free(displs);
    return MPI_SUCCESS;
  }

  /* Allocate and initialize temporary buffers, we need:
     - a temporary buffer to perform reduction (size total_count) since
     rbuf can be of rcounts[rank] size.
     - up to two temporary buffers used for communication/computation overlap.
  */
  ret = MPI_Type_get_extent(dtype, &lb, &extent);
  if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }

  max_real_segsize = datatype_span(dtype, max_block_count, &gap);
  dsize = datatype_span(dtype, total_count, &gap);

  accumbuf_free = (char*)malloc(dsize);
  if (NULL == accumbuf_free) { ret = -1; line = __LINE__; goto error_hndl; }
  accumbuf = accumbuf_free - gap;

  inbuf_free[0] = (char*)malloc(max_real_segsize);
  if (NULL == inbuf_free[0]) { ret = -1; line = __LINE__; goto error_hndl; }
  inbuf[0] = inbuf_free[0] - gap;
  if (size > 2) {
    inbuf_free[1] = (char*)malloc(max_real_segsize);
    if (NULL == inbuf_free[1]) { ret = -1; line = __LINE__; goto error_hndl; }
    inbuf[1] = inbuf_free[1] - gap;
  }

  /* Handle MPI_IN_PLACE for size > 1 */
  if (MPI_IN_PLACE == sbuf) {
    sbuf = rbuf;
  }

  ret = copy_buffer((char*) sbuf, accumbuf, total_count, dtype);
  if (ret < 0) { line = __LINE__; goto error_hndl; }

  /* Computation loop */

  /*
     For each of the remote nodes:
     - post irecv for block (r-2) from (r-1) with wrap around
     - send block (r-1) to (r+1)
     - in loop for every step k = 2 .. n
     - post irecv for block (r - 1 + n - k) % n
     - wait on block (r + n - k) % n to arrive
     - compute on block (r + n - k ) % n
     - send block (r + n - k) % n
     - wait on block (r)
     - compute on block (r)
     - copy block (r) to rbuf
     Note that we must be careful when computing the beginning of buffers and
     for send operations and computation we must compute the exact block size.
  */
  send_to = (rank + 1) % size;
  recv_from = (rank + size - 1) % size;

  inbi = 0;
  /* Initialize first receive from the neighbor on the left */
  ret = MPI_Irecv(inbuf[inbi], max_block_count, dtype, recv_from, 0, comm, &reqs[inbi]);
  if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
  tmpsend = accumbuf + displs[recv_from] * extent;
  ret = MPI_Send(tmpsend, rcounts[recv_from], dtype, send_to, 0, comm);
  if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }

  for (k = 2; k < size; k++) {
    const int prevblock = (rank + size - k) % size;

    inbi = inbi ^ 0x1;

    /* Post irecv for the current block */
    ret = MPI_Irecv(inbuf[inbi], max_block_count, dtype, recv_from, 0, comm, &reqs[inbi]);
    if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }

    /* Wait on previous block to arrive */
    ret = MPI_Wait(&reqs[inbi ^ 0x1], MPI_STATUS_IGNORE);
    if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }

    /* Apply operation on previous block: result goes to rbuf
       rbuf[prevblock] = inbuf[inbi ^ 0x1] (op) rbuf[prevblock]
    */
    tmprecv = accumbuf + displs[prevblock] * extent;
    MPI_Reduce_local(inbuf[inbi ^ 0x1], tmprecv, rcounts[prevblock], dtype, op);

    /* send previous block to send_to */
    ret = MPI_Send(tmprecv, rcounts[prevblock], dtype, send_to, 0, comm);
    if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
  }

  /* Wait on the last block to arrive */
  ret = MPI_Wait(&reqs[inbi], MPI_STATUS_IGNORE);
  if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }

  /* Apply operation on the last block (my block)
     rbuf[rank] = inbuf[inbi] (op) rbuf[rank] */
  tmprecv = accumbuf + displs[rank] * extent;
  MPI_Reduce_local(inbuf[inbi], tmprecv, rcounts[rank], dtype, op);

  /* Copy result from tmprecv to rbuf */
  ret = copy_buffer(tmprecv, (char *)rbuf, rcounts[rank], dtype);
  if (ret < 0) { line = __LINE__; goto error_hndl; }

  if (NULL != displs) free(displs);
  if (NULL != accumbuf_free) free(accumbuf_free);
  if (NULL != inbuf_free[0]) free(inbuf_free[0]);
  if (NULL != inbuf_free[1]) free(inbuf_free[1]);

  return MPI_SUCCESS;

 error_hndl:
  fprintf(stderr, "\n%s:%4d\tRank %d Error occurred %d\n\n", __FILE__, line, rank, ret);
  (void)line;  // silence compiler warning
  if (NULL != displs) free(displs);
  if (NULL != accumbuf_free) free(accumbuf_free);
  if (NULL != inbuf_free[0]) free(inbuf_free[0]);
  if (NULL != inbuf_free[1]) free(inbuf_free[1]);
  return ret;
}


int reduce_scatter_butterfly(const void *sbuf, void *rbuf, const int rcounts[],
                             MPI_Datatype dtype, MPI_Op op, MPI_Comm comm)
{
  char *tmpbuf[2] = {NULL, NULL}, *psend, *precv;
  ptrdiff_t *displs = NULL, index;
  ptrdiff_t span, gap, totalcount, extent, lb;
  int rank, comm_size, err = MPI_SUCCESS;
  err = MPI_Comm_size(comm, &comm_size);
  err = MPI_Comm_rank(comm, &rank);

  if (comm_size < 2)
    return MPI_SUCCESS;

  displs = malloc(sizeof(*displs) * comm_size);
  if (NULL == displs) {
    err = MPI_ERR_NO_MEM;
    goto cleanup_and_return;
  }
  displs[0] = 0;
  for (int i = 1; i < comm_size; i++) {
    displs[i] = displs[i - 1] + rcounts[i - 1];
  }
  totalcount = displs[comm_size - 1] + rcounts[comm_size - 1];

  MPI_Type_get_extent(dtype, &lb, &extent);
  span = datatype_span(dtype, totalcount, &gap);
  tmpbuf[0] = malloc(span);
  tmpbuf[1] = malloc(span);
  if (NULL == tmpbuf[0] || NULL == tmpbuf[1]) {
    err = MPI_ERR_NO_MEM;
    goto cleanup_and_return;
  }
  psend = tmpbuf[0] - gap;
  precv = tmpbuf[1] - gap;

  if (sbuf != MPI_IN_PLACE) {
    err = copy_buffer((char *) sbuf, psend, totalcount, dtype);
    if (MPI_SUCCESS != err) { goto cleanup_and_return; }
  } else {
    err = copy_buffer(rbuf, psend, totalcount, dtype);
    if (MPI_SUCCESS != err) { goto cleanup_and_return; }
  }

  /*
   * Step 1. Reduce the number of processes to the nearest lower power of two
   * p' = 2^{\floor{\log_2 p}} by removing r = p - p' processes.
   * In the first 2r processes (ranks 0 to 2r - 1), all the even ranks send
   * the input vector to their neighbor (rank + 1) and all the odd ranks recv
   * the input vector and perform local reduction.
   * The odd ranks (0 to 2r - 1) contain the reduction with the input
   * vector on their neighbors (the even ranks). The first r odd
   * processes and the p - 2r last processes are renumbered from
   * 0 to 2^{\floor{\log_2 p}} - 1. Even ranks do not participate in the
   * rest of the algorithm.
   */

  /* Find nearest power-of-two less than or equal to comm_size */
  int nprocs_pof2 = next_poweroftwo(comm_size);
  nprocs_pof2 >>= 1;
  int nprocs_rem = comm_size - nprocs_pof2;
  int log2_size = log_2(nprocs_pof2);

  int vrank = -1;
  if (rank < 2 * nprocs_rem) {
    if ((rank % 2) == 0) {
      /* Even process */
      err = MPI_Send(psend, totalcount, dtype, rank + 1, 0, comm);
      if (MPI_SUCCESS != err) { goto cleanup_and_return; }
      /* This process does not participate in the rest of the algorithm */
      vrank = -1;
    } else {
      /* Odd process */
      err = MPI_Recv(precv, totalcount, dtype, rank - 1, 0, comm, MPI_STATUS_IGNORE);
      if (MPI_SUCCESS != err) { goto cleanup_and_return; }
      MPI_Reduce_local(precv, psend, totalcount, dtype, op);
      /* Adjust rank to be the bottom "remain" ranks */
      vrank = rank / 2;
    }
  } else {
    /* Adjust rank to show that the bottom "even remain" ranks dropped out */
    vrank = rank - nprocs_rem;
  }

  if (vrank != -1) {
    /*
     * Now, psend vector of size totalcount is divided into nprocs_pof2 blocks:
     * block 0:   rcounts[0] and rcounts[1] -- for process 0 and 1
     * block 1:   rcounts[2] and rcounts[3] -- for process 2 and 3
     * ...
     * block r-1: rcounts[2*(r-1)] and rcounts[2*(r-1)+1]
     * block r:   rcounts[r+r]
     * block r+1: rcounts[r+r+1]
     * ...
     * block nprocs_pof2 - 1: rcounts[r+nprocs_pof2-1]
     */
    int nblocks = nprocs_pof2, send_index = 0, recv_index = 0;
    for (int mask = 1; mask < nprocs_pof2; mask <<= 1) {
      int vpeer = vrank ^ mask;
      int peer = (vpeer < nprocs_rem) ? vpeer * 2 + 1 : vpeer + nprocs_rem;

      nblocks /= 2;
      if ((vrank & mask) == 0) {
        /* Send the upper half of reduction buffer, recv the lower half */
        send_index += nblocks;
      } else {
        /* Send the upper half of reduction buffer, recv the lower half */
        recv_index += nblocks;
      }

      /* Send blocks: [send_index, send_index + nblocks - 1] */
      size_t send_count = sum_counts(rcounts, displs, nprocs_rem,
                                     send_index, send_index + nblocks - 1);
      index = (send_index < nprocs_rem) ? 2 * send_index : nprocs_rem + send_index;
      ptrdiff_t sdispl = displs[index];

      /* Recv blocks: [recv_index, recv_index + nblocks - 1] */
      size_t recv_count = sum_counts(rcounts, displs, nprocs_rem,
                                          recv_index, recv_index + nblocks - 1);
      index = (recv_index < nprocs_rem) ? 2 * recv_index : nprocs_rem + recv_index;
      ptrdiff_t rdispl = displs[index];

      err = MPI_Sendrecv(psend + sdispl * extent, send_count, dtype, peer, 0,
                         precv + rdispl * extent, recv_count, dtype, peer, 0,
                         comm, MPI_STATUS_IGNORE);
      if (MPI_SUCCESS != err) { goto cleanup_and_return; }

      if (vrank < vpeer) {
        /* precv = psend <op> precv */
        MPI_Reduce_local(psend + rdispl * extent, precv + rdispl * extent,
                         recv_count, dtype, op);
        char *p = psend;
        psend = precv;
        precv = p;
      } else {
        /* psend = precv <op> psend */
        MPI_Reduce_local(precv + rdispl * extent, psend + rdispl * extent,
                         recv_count, dtype, op);
      }
      send_index = recv_index;
    }
    /*
     * psend points to the result block [send_index]
     * Exchange results with remote process according to a mirror permutation.
     */
    int vpeer = mirror_perm(vrank, log2_size);
    int peer = (vpeer < nprocs_rem) ? vpeer * 2 + 1 : vpeer + nprocs_rem;
    index = (send_index < nprocs_rem) ? 2 * send_index : nprocs_rem + send_index;

    if (vpeer < nprocs_rem) {
      /*
       * Process has two blocks: for excluded process and own.
       * Send the first block to excluded process.
       */
      err = MPI_Send(psend + displs[index] * extent, rcounts[index], dtype,
                     peer - 1, 0, comm);
      if (MPI_SUCCESS != err) { goto cleanup_and_return; }
    }

    /* If process has two blocks, then send the second block (own block) */
    if (vpeer < nprocs_rem)
      index++;
    if (vpeer != vrank) {
      err = MPI_Sendrecv(psend + displs[index] * extent, rcounts[index], dtype, peer, 0,
                         rbuf, rcounts[rank], dtype, peer, 0, comm, MPI_STATUS_IGNORE);
      if (MPI_SUCCESS != err) { goto cleanup_and_return; }
    } else {
      err = copy_buffer(psend + displs[rank] * extent, rbuf, rcounts[rank], dtype);
      if (MPI_SUCCESS != err) { goto cleanup_and_return; }
    }

  } else {
    /* Excluded process: receive result */
    int vpeer = mirror_perm((rank + 1) / 2, log2_size);
    int peer = (vpeer < nprocs_rem) ? vpeer * 2 + 1 : vpeer + nprocs_rem;
    err = MPI_Recv(rbuf, rcounts[rank], dtype, peer, 0, comm, MPI_STATUS_IGNORE);
    if (MPI_SUCCESS != err) { goto cleanup_and_return; }
   }

cleanup_and_return:
  if (displs)
    free(displs);
  if (tmpbuf[0])
    free(tmpbuf[0]);
  if (tmpbuf[1])
    free(tmpbuf[1]);
  return err;
}

int reduce_scatter_swing_static(const void *sbuf, void *rbuf, const int rcounts[],
                                    MPI_Datatype dtype, MPI_Op op, MPI_Comm comm)
{
  int i, rank, size, steps, w_size, err = MPI_SUCCESS;
  size_t count;
  ptrdiff_t *disps = NULL;
  ptrdiff_t extent, lb, buf_size, gap = 0;
  char *recv_buf = NULL, *recv_buf_free = NULL;
  char *result_buf = NULL, *result_buf_free = NULL;
  const int *s_bitmap = NULL, *r_bitmap = NULL;
  int *permutation = NULL;

  err = MPI_Comm_size(comm, &size);
  err = MPI_Comm_rank(comm, &rank);


  /* Find displacements and total count*/
  disps = (ptrdiff_t*) malloc(sizeof(ptrdiff_t) * size);
  if (NULL == disps) return MPI_ERR_NO_MEM;

  disps[0] = 0;
  for (i = 0; i < (size - 1); ++i) {
    disps[i + 1] = disps[i] + rcounts[i];
  }
  count = disps[size - 1] + rcounts[size - 1];

  /* short cut the trivial case */
  if (0 == count) {
    free(disps);
    return MPI_SUCCESS;
  }

  /* get datatype information */
  MPI_Type_get_extent(dtype, &lb, &extent);
  buf_size = datatype_span(dtype, count, &gap);

  /* Handle MPI_IN_PLACE */
  if (MPI_IN_PLACE == sbuf) {
    sbuf = rbuf;
  }

  /* Allocate temporary receive buffer. */
  recv_buf_free = (char*) malloc(buf_size);
  recv_buf = recv_buf_free - gap;
  if (NULL == recv_buf_free) {
    err = MPI_ERR_NO_MEM;
    goto cleanup;
  }

  /* allocate temporary buffer for results */
  result_buf_free = (char*) malloc(buf_size);
  result_buf = result_buf_free - gap;

  /* copy local buffer into the temporary results */
  err = copy_buffer_different_dt(sbuf, count, dtype, result_buf, count, dtype);
  if (MPI_SUCCESS != err) goto cleanup;

  /* Current implementation only handles power-of-two number of processes.*/
  steps = log_2(size);
  if (size != (1 << steps)) {
    fprintf (stderr, "ERROR! Swing static reduce scatter works only with po2 ranks!");
    err = MPI_ERR_SIZE;
    goto cleanup;
  }
  if (get_static_bitmap(&s_bitmap, &r_bitmap, steps, size, rank) == -1 ||
      get_perm_bitmap(&permutation, steps, size) == -1) {
    err = MPI_ERR_UNKNOWN;
    goto cleanup;
  }

  /* Main communication and reduction loop */
  w_size = size >> 1;
  for (int step = 0; step < steps; step++){
    int peer, send_index, recv_index;
    size_t s_count, r_count;

    /* determine the peer we will communicate with */
    peer = pi(rank, step, size);

    /* use the bitmaps to figure out what we're sending and
      receiving */
    send_index = s_bitmap[step];
    recv_index = r_bitmap[step];
    s_count = r_count = 0;
    for (int i = 0; i < w_size; ++i) {
      s_count += rcounts[s_bitmap[step] + i];
      r_count += rcounts[r_bitmap[step] + i];
    }

    /* actual data transfer.  Send from result_buf,
        receive into recv_buf */
    err = MPI_Sendrecv(result_buf + disps[send_index] * extent, s_count, dtype, peer, 0,
               recv_buf + disps[recv_index] * extent, r_count, dtype, peer, 0,
               comm, MPI_STATUS_IGNORE);
    if (MPI_SUCCESS != err) {
      goto cleanup;
    }

    /* if we received something on this step, push it into
        the results buffer and perform the reduction*/
    if (r_count > 0) {
      err = MPI_Reduce_local(recv_buf + disps[recv_index] * extent,
                        result_buf + disps[recv_index] * extent,
                        r_count, dtype, op);
      if (MPI_SUCCESS != err) {
        goto cleanup;
      }
    }
    /* halve the window size for the next iteration */
    w_size >>= 1;
  }


  if (rank != permutation[rank]) {
    /* send result to correct rank's recv buffer */
    err = MPI_Sendrecv(result_buf + disps[r_bitmap[steps - 1]] * extent,
            rcounts[permutation[rank]], dtype, permutation[rank], 0,
            rbuf, rcounts[rank], dtype, get_sender(permutation, size, rank), 0,
            comm, MPI_STATUS_IGNORE);
    if (MPI_SUCCESS != err) {
      goto cleanup;
    }
  } else {
    /* copy local results from results buffer into real receive buffer */
    err = copy_buffer_different_dt(result_buf + disps[rank] * extent, rcounts[rank],
                                    dtype, rbuf, rcounts[rank], dtype);
    if (MPI_SUCCESS != err) {
      goto cleanup;
    }
  }

 cleanup:
  if (NULL != disps) free(disps);
  if (NULL != recv_buf_free) free(recv_buf_free);
  if (NULL != result_buf_free) free(result_buf_free);

  return err;
}
