#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>

#include "libswing.h"
#include "libswing_utils.h"
#include "libswing_utils_bitmaps.h"



int allgather_recursivedoubling(const void *sbuf, size_t scount, MPI_Datatype sdtype,
                                 void* rbuf, size_t rcount, MPI_Datatype rdtype, MPI_Comm comm)
{
  int line = -1, rank, size, pow2size, err;
  int remote, distance, sendblocklocation;
  ptrdiff_t rlb, rext;
  char *tmpsend = NULL, *tmprecv = NULL;

  err = MPI_Comm_size(comm, &size);
  err = MPI_Comm_rank(comm, &rank);

  pow2size = next_poweroftwo (size);
  pow2size >>=1;

  /* Current implementation only handles power-of-two number of processes.
     If the function was called on non-power-of-two number of processes,
     print warning and call bruck allgather algorithm with same parameters.
  */
  if (pow2size != size) {
    fprintf(stderr, "ERROR! Recoursive doubling allgather works only with po2 ranks!\n");
    goto err_hndl;
  }

  err = MPI_Type_get_extent (rdtype, &rlb, &rext);
  if (MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }

  /* Initialization step:
     - if send buffer is not MPI_IN_PLACE, copy send buffer to block 0 of
     receive buffer
  */
  if (MPI_IN_PLACE != sbuf) {
    tmpsend = (char*) sbuf;
    tmprecv = (char*) rbuf + (ptrdiff_t)rank * (ptrdiff_t)rcount * rext;
    err = copy_buffer_different_dt(tmpsend, scount, sdtype, tmprecv, rcount, rdtype);
    if (MPI_SUCCESS != err) { line = __LINE__; goto err_hndl;  }

  }

  /* Communication step:
     At every step i, rank r:
     - exchanges message with rank remote = (r ^ 2^i).

  */
  sendblocklocation = rank;
  for (distance = 0x1; distance < size; distance<<=1) {
    remote = rank ^ distance;

    if (rank < remote) {
      tmpsend = (char*)rbuf + (ptrdiff_t)sendblocklocation * (ptrdiff_t)rcount * rext;
      tmprecv = (char*)rbuf + (ptrdiff_t)(sendblocklocation + distance) * (ptrdiff_t)rcount * rext;
    } else {
      tmpsend = (char*)rbuf + (ptrdiff_t)sendblocklocation * (ptrdiff_t)rcount * rext;
      tmprecv = (char*)rbuf + (ptrdiff_t)(sendblocklocation - distance) * (ptrdiff_t)rcount * rext;
      sendblocklocation -= distance;
    }

    /* Sendreceive */
    err = MPI_Sendrecv(tmpsend, (ptrdiff_t)distance * (ptrdiff_t)rcount, rdtype, remote, 0,
                       tmprecv, (ptrdiff_t)distance * (ptrdiff_t)rcount, rdtype,
                       remote, 0, comm, MPI_STATUS_IGNORE);
    if (MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }

  }

  return MPI_SUCCESS;

err_hndl:
  fprintf(stderr, "%s:%4d\tRank %d Error occurred %d\n", __FILE__, line, rank, err);
  (void)line;  // silence compiler warning
  return err;
}

int allgather_k_bruck(const void *sbuf, size_t scount, MPI_Datatype sdtype,
                      void* rbuf, size_t rcount, MPI_Datatype rdtype, MPI_Comm comm,
                      int radix)
{
  int line = -1, rank, size, dst, src, err = MPI_SUCCESS;
  int recvcount, distance;
  ptrdiff_t rlb, rextent;
  ptrdiff_t rsize, rgap = 0;
  MPI_Request *reqs = NULL;
  request_manager_t req_manager = {NULL, 0};
  int num_reqs, max_reqs = 0;

  char *tmpsend = NULL, *tmprecv = NULL, *tmp_buf = NULL, *tmp_buf_start = NULL;

  err = MPI_Comm_size(comm, &size);
  err = MPI_Comm_rank(comm, &rank);

  // OPAL_OUTPUT((ompi_coll_base_framework.framework_output,
  //              "coll:base:allgather_intra_k_bruck radix %d rank %d", radix, rank));
  err = MPI_Type_get_extent (rdtype, &rlb, &rextent);
  if (MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }

  if (0 != rank) {
    /* Compute the temporary buffer size, including datatypes empty gaps */
    rsize = datatype_span(rdtype, (size_t)rcount * (size - rank), &rgap);
    tmp_buf = (char *) malloc(rsize);
    tmp_buf_start = tmp_buf - rgap;
  }

  // tmprecv points to the data initially on this rank, handle mpi_in_place case
  tmprecv = (char*) rbuf;
  if (MPI_IN_PLACE != sbuf) {
    tmpsend = (char*) sbuf;
    err = copy_buffer_different_dt(tmpsend, scount, sdtype, tmprecv, rcount, rdtype);
    if (MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }
  } else if (0 != rank) {
    // root data placement is at the correct poistion
    tmpsend = ((char*)rbuf) + (ptrdiff_t)rank * (ptrdiff_t)rcount * rextent;
    err = copy_buffer(tmpsend, tmprecv, rcount, rdtype);
    if (MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }
  }
  /*
     Maximum number of communication phases logk(n)
     For each phase i, rank r:
     - increase the distance and recvcount by k times
     - sends (k - 1) messages which starts at beginning of rbuf and has size
     (recvcount) to rank (r - distance * j)
     - receives (k - 1) messages of size recvcount from rank (r + distance * j)
     at location (rbuf + distance * j * rcount * rext)
     - calculate the remaining data for each of the (k - 1) messages in the last
     phase to complete all transactions
  */
  max_reqs = 2 * (radix - 1);
  reqs = alloc_reqs(&req_manager, max_reqs);
  recvcount = 1;
  tmpsend = (char*) rbuf;
  for (distance = 1; distance < size; distance *= radix) {
    num_reqs = 0;
    for (int j = 1; j < radix; j++)
    {
      if (distance * j >= size) {
        break;
      }
      src = (rank + distance * j) % size;
      dst = (rank - distance * j + size) % size;

      tmprecv = tmpsend + (ptrdiff_t)distance * j * rcount * rextent;

      if (distance <= (size / radix)) {
        recvcount = distance;
      } else {
        recvcount = (distance < (size - distance * j) ? 
                          distance:(size - distance * j));
      }

      err = MPI_Irecv(tmprecv, recvcount * rcount, rdtype, src, 
                      0, comm, &reqs[num_reqs++]);
      if (MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }
      err = MPI_Isend(tmpsend, recvcount * rcount, rdtype, dst,
                      0, comm, &reqs[num_reqs++]);
      if (MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }
    }
    err = MPI_Waitall(num_reqs, reqs, MPI_STATUSES_IGNORE);
    if (MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }
  }

  // Finalization step:        On all ranks except 0, data needs to be shifted locally
  if (0 != rank) {
    err = copy_buffer(rbuf, tmp_buf_start, ((ptrdiff_t) (size - rank) * rcount), rdtype);
    if (MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }

    tmpsend = (char*) rbuf + (ptrdiff_t) (size - rank) * rcount * rextent;
    err = copy_buffer(tmpsend, rbuf, (ptrdiff_t)rank * rcount, rdtype);
    if (MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }

    tmprecv = (char*) rbuf + (ptrdiff_t)rank * rcount * rextent;
    err = copy_buffer(tmp_buf_start, tmprecv, (ptrdiff_t)(size - rank) * rcount, rdtype);
    if (MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }
  }

  if(tmp_buf != NULL) free(tmp_buf);
  return MPI_SUCCESS;

err_hndl:
  if( NULL != reqs ) {
    cleanup_reqs(&req_manager);
  }
  fprintf(stdout,  "%s:%4d\tError occurred %d, rank %2d", __FILE__, line, err, rank);
  if(tmp_buf != NULL) {
    free(tmp_buf);
    tmp_buf = NULL;
    tmp_buf_start = NULL;
  }
  (void)line;  // silence compiler warning
  return err;
}

int allgather_ring(const void *sbuf, size_t scount, MPI_Datatype sdtype,
                   void* rbuf, size_t rcount, MPI_Datatype rdtype, MPI_Comm comm)
{
  int line = -1, rank, size, err, sendto, recvfrom, i, recvdatafrom, senddatafrom;
  ptrdiff_t rlb, rext;
  char *tmpsend = NULL, *tmprecv = NULL;

  err = MPI_Comm_size(comm, &size);
  err = MPI_Comm_rank(comm, &rank);

  err = MPI_Type_get_extent (rdtype, &rlb, &rext);
  if (MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }

  /* Initialization step:
     - if send buffer is not MPI_IN_PLACE, copy send buffer to appropriate block
     of receive buffer
  */
  tmprecv = (char*) rbuf + (ptrdiff_t)rank * (ptrdiff_t)rcount * rext;
  if (MPI_IN_PLACE != sbuf) {
    tmpsend = (char*) sbuf;
    err = copy_buffer_different_dt(tmpsend, scount, sdtype, tmprecv, rcount, rdtype);
    if (MPI_SUCCESS != err) { line = __LINE__; goto err_hndl;  }
  }

  /* Communication step:
     At every step i: 0 .. (P-1), rank r:
     - receives message from [(r - 1 + size) % size] containing data from rank
     [(r - i - 1 + size) % size]
     - sends message to rank [(r + 1) % size] containing data from rank
     [(r - i + size) % size]
     - sends message which starts at beginning of rbuf and has size
  */
  sendto = (rank + 1) % size;
  recvfrom  = (rank - 1 + size) % size;

  for (i = 0; i < size - 1; i++) {
    recvdatafrom = (rank - i - 1 + size) % size;
    senddatafrom = (rank - i + size) % size;

    tmprecv = (char*)rbuf + (ptrdiff_t)recvdatafrom * (ptrdiff_t)rcount * rext;
    tmpsend = (char*)rbuf + (ptrdiff_t)senddatafrom * (ptrdiff_t)rcount * rext;

    /* Sendreceive */
    err = MPI_Sendrecv(tmpsend, rcount, rdtype, sendto, 0,
                       tmprecv, rcount, rdtype, recvfrom, 0,
                       comm, MPI_STATUS_IGNORE);
    if (MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }

  }

  return MPI_SUCCESS;

err_hndl:
  fprintf(stderr, "%s:%4d\tError occurred %d, rank %2d", __FILE__, line, err, rank);
  (void)line;  // silence compiler warning
  return err;
}

int allgather_swing_static(const void *sbuf, size_t scount, MPI_Datatype sdtype,
                           void* rbuf, size_t rcount, MPI_Datatype rdtype, MPI_Comm comm)
{
  int line = -1, rank, size, pow2size, steps, err, remote;
  const int *s_bitmap = NULL, *r_bitmap = NULL;
  int *permutation = NULL;
  ptrdiff_t rlb, rext;
  char *tmpsend = NULL, *tmprecv = NULL;

  err = MPI_Comm_size(comm, &size);
  err = MPI_Comm_rank(comm, &rank);

  pow2size = next_poweroftwo (size);
  pow2size >>=1;

  /* Current implementation only handles power-of-two number of processes.
     If the function was called on non-power-of-two number of processes,
     print warning and call bruck allgather algorithm with same parameters.
  */
  if (pow2size != size) {
    fprintf(stderr, "ERROR! Swing static allgather works only with po2 ranks!\n");
    goto err_hndl;

  }
  
  steps = log_2(size);

  err = MPI_Type_get_extent (rdtype, &rlb, &rext);
  if (MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }

  if(get_static_bitmap(&s_bitmap, &r_bitmap, steps, size, rank) == -1){
    line = __LINE__;
    goto err_hndl;
  }

  if(get_perm_bitmap(&permutation, steps, size) == -1){
    line = __LINE__;
    goto err_hndl;
  }

  /* Initialization step:
     - if send buffer is not MPI_IN_PLACE, copy send buffer to block  of
     receive buffer
  */
  if (MPI_IN_PLACE != sbuf) {
    tmpsend = (char*) sbuf;
    tmprecv = (char*) rbuf + (ptrdiff_t)r_bitmap[steps - 1] * (ptrdiff_t)rcount * rext;

    err = copy_buffer_different_dt(tmpsend, scount, sdtype, tmprecv, rcount, rdtype);
    if (MPI_SUCCESS != err) { line = __LINE__; goto err_hndl;  }
  }


  /* Communication step:
     At every step i, rank r:
     - exchanges message with rank remote = (r ^ 2^i).

  */
  size_t step_scount = rcount;
  for(int step = steps - 1; step >= 0; step--) {
    remote = pi(rank, step, size);

    tmpsend = (char*)rbuf + (ptrdiff_t)r_bitmap[step] * (ptrdiff_t) rcount * rext;
    tmprecv = (char*)rbuf + (ptrdiff_t)s_bitmap[step] * (ptrdiff_t) rcount * rext;

    /* Sendreceive */
    err = MPI_Sendrecv(tmpsend, step_scount, rdtype, remote, 0, 
                       tmprecv, step_scount, rdtype, remote, 0,
                       comm, MPI_STATUS_IGNORE);
    if (MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }
    step_scount *= 2;

  }
  
  if(reorder_blocks(rbuf, rcount * rext, permutation, size) != MPI_SUCCESS){
    line = __LINE__;
    goto err_hndl;
  }

  return MPI_SUCCESS;

err_hndl:
  fprintf(stderr, "%s:%4d\tRank %d Error occurred %d\n", __FILE__, line, rank, err);
  (void)line;  // silence compiler warning
  return err;
}
