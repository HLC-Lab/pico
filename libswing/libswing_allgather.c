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
  int line = -1, rank, size, err = MPI_SUCCESS;
  int remote, distance, sendblocklocation;
  ptrdiff_t rlb, rext;
  char *tmpsend = NULL, *tmprecv = NULL;

  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  /*
   * Current implementation only handles power-of-two number of processes.
   */
  if(!is_power_of_two(size)) {
    SWING_DEBUG_PRINT("ERROR! Recoursive doubling allgather works only with po2 ranks!");
    goto err_hndl;
  }

  err = MPI_Type_get_extent (rdtype, &rlb, &rext);
  if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }

  /* Initialization step:
     - if send buffer is not MPI_IN_PLACE, copy send buffer to block 0 of
     receive buffer
  */
  if(MPI_IN_PLACE != sbuf) {
    tmpsend = (char*) sbuf;
    tmprecv = (char*) rbuf + (ptrdiff_t)rank * (ptrdiff_t)rcount * rext;

#ifdef CUDA_AWARE
    err = copy_buffer_different_dt_cuda(tmpsend, scount, sdtype, tmprecv, rcount, rdtype);
#else
    err = copy_buffer_different_dt(tmpsend, scount, sdtype, tmprecv, rcount, rdtype);
#endif

    if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl;  }

  }

  /* Communication step:
     At every step i, rank r:
     - exchanges message with rank remote = (r ^ 2^i).

  */
  sendblocklocation = rank;
  for(distance = 0x1; distance < size; distance<<=1) {
    remote = rank ^ distance;

    if(rank < remote) {
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
    if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }

  }

  return MPI_SUCCESS;

err_hndl:
  SWING_DEBUG_PRINT("\n%s:%4d\tRank %d Error occurred %d\n\n", __FILE__, line, rank, err);
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

  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  // OPAL_OUTPUT((ompi_coll_base_framework.framework_output,
  //              "coll:base:allgather_intra_k_bruck radix %d rank %d", radix, rank));
  err = MPI_Type_get_extent (rdtype, &rlb, &rextent);
  if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }

  if(0 != rank) {
    /* Compute the temporary buffer size, including datatypes empty gaps */
    rsize = datatype_span(rdtype, (size_t)rcount * (size - rank), &rgap);
    
#ifdef CUDA_AWARE
    SWING_CUDA_CHECK(cudaMalloc((void**)&tmp_buf, rsize));
    SWING_CUDA_CHECK(cudaMemset(tmp_buf, 0, rsize));
#else
    tmp_buf = (char *) malloc(rsize);
#endif

    tmp_buf_start = tmp_buf - rgap;
  }

  // tmprecv points to the data initially on this rank, handle mpi_in_place case
  tmprecv = (char*) rbuf;
  if(MPI_IN_PLACE != sbuf) {
    tmpsend = (char*) sbuf;

#ifdef CUDA_AWARE
    err = copy_buffer_different_dt_cuda(tmpsend, scount, sdtype, tmprecv, rcount, rdtype);
#else
    err = copy_buffer_different_dt(tmpsend, scount, sdtype, tmprecv, rcount, rdtype);
#endif

    if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }
  } else if(0 != rank) {
    // root data placement is at the correct poistion
    tmpsend = ((char*)rbuf) + (ptrdiff_t)rank * (ptrdiff_t)rcount * rextent;
    err = copy_buffer(tmpsend, tmprecv, rcount, rdtype);
    if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }
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
  for(distance = 1; distance < size; distance *= radix) {
    num_reqs = 0;
    for(int j = 1; j < radix; j++)
    {
      if(distance * j >= size) {
        break;
      }
      src = (rank + distance * j) % size;
      dst = (rank - distance * j + size) % size;

      tmprecv = tmpsend + (ptrdiff_t)distance * j * rcount * rextent;

      if(distance <= (size / radix)) {
        recvcount = distance;
      } else {
        recvcount = (distance < (size - distance * j) ? 
                          distance:(size - distance * j));
      }

      err = MPI_Irecv(tmprecv, recvcount * rcount, rdtype, src, 
                      0, comm, &reqs[num_reqs++]);
      if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }
      err = MPI_Isend(tmpsend, recvcount * rcount, rdtype, dst,
                      0, comm, &reqs[num_reqs++]);
      if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }
    }
    err = MPI_Waitall(num_reqs, reqs, MPI_STATUSES_IGNORE);
    if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }
  }

  // Finalization step:        On all ranks except 0, data needs to be shifted locally
  if(0 != rank) {
    err = copy_buffer(rbuf, tmp_buf_start, ((ptrdiff_t) (size - rank) * rcount), rdtype);
    if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }

    tmpsend = (char*) rbuf + (ptrdiff_t) (size - rank) * rcount * rextent;
    err = copy_buffer(tmpsend, rbuf, (ptrdiff_t)rank * rcount, rdtype);
    if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }

    tmprecv = (char*) rbuf + (ptrdiff_t)rank * rcount * rextent;
    err = copy_buffer(tmp_buf_start, tmprecv, (ptrdiff_t)(size - rank) * rcount, rdtype);
    if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }
  }

  if(tmp_buf != NULL) free(tmp_buf);
  return MPI_SUCCESS;

err_hndl:
  if( NULL != reqs ) {
    cleanup_reqs(&req_manager);
  }
  SWING_DEBUG_PRINT( "\n%s:%4d\tError occurred %d, rank %2d\n\n", __FILE__, line, err, rank);
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
  if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }

  /* Initialization step:
     - if send buffer is not MPI_IN_PLACE, copy send buffer to appropriate block
     of receive buffer
  */
  tmprecv = (char*) rbuf + (ptrdiff_t)rank * (ptrdiff_t)rcount * rext;
  if(MPI_IN_PLACE != sbuf) {
    tmpsend = (char*) sbuf;

#ifdef CUDA_AWARE
    err = copy_buffer_different_dt_cuda(tmpsend, scount, sdtype, tmprecv, rcount, rdtype);
#else
    err = copy_buffer_different_dt(tmpsend, scount, sdtype, tmprecv, rcount, rdtype);
#endif

    if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl;  }
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

  for(i = 0; i < size - 1; i++) {
    recvdatafrom = (rank - i - 1 + size) % size;
    senddatafrom = (rank - i + size) % size;

    tmprecv = (char*)rbuf + (ptrdiff_t)recvdatafrom * (ptrdiff_t)rcount * rext;
    tmpsend = (char*)rbuf + (ptrdiff_t)senddatafrom * (ptrdiff_t)rcount * rext;

    /* Sendreceive */
    err = MPI_Sendrecv(tmpsend, rcount, rdtype, sendto, 0,
                       tmprecv, rcount, rdtype, recvfrom, 0,
                       comm, MPI_STATUS_IGNORE);
    if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }

  }

  return MPI_SUCCESS;

err_hndl:
  SWING_DEBUG_PRINT("\n%s:%4d\tError occurred %d, rank %2d\n\n", __FILE__, line, err, rank);
  (void)line;  // silence compiler warning
  return err;
}

int allgather_swing_static_memcpy(const void *sbuf, size_t scount, MPI_Datatype sdtype,
                           void* rbuf, size_t rcount, MPI_Datatype rdtype, MPI_Comm comm)
{
  int line = -1, rank, size, steps, err = MPI_SUCCESS, remote;
  const int *s_bitmap = NULL, *r_bitmap = NULL;
  int *permutation = NULL;
  ptrdiff_t rlb, rext;
  char *tmpsend = NULL, *tmprecv = NULL;

  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  /*
   * Current implementation only handles power-of-two number of processes.
   */
  steps = log_2(size);
  if(!is_power_of_two(size) || steps < 1) {
    SWING_DEBUG_PRINT("ERROR! Swing static allgather works only with po2 ranks!");
    return MPI_ERR_ARG;
  }
  

  err = MPI_Type_get_extent (rdtype, &rlb, &rext);
  if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }

  if(get_static_bitmap(&s_bitmap, &r_bitmap, steps, size, rank) == -1 ||
     get_perm_bitmap(&permutation, steps, size) == -1){
    line = __LINE__;
    goto err_hndl;
  }

  /* Initialization step:
     - if send buffer is not MPI_IN_PLACE, copy send buffer to block  of
     receive buffer
  */
  if(MPI_IN_PLACE != sbuf) {
    tmpsend = (char*) sbuf;
    tmprecv = (char*) rbuf + (ptrdiff_t)r_bitmap[steps - 1] * (ptrdiff_t)rcount * rext;

#ifdef CUDA_AWARE
    err = copy_buffer_different_dt_cuda(tmpsend, scount, sdtype, tmprecv, rcount, rdtype);
#else
    err = copy_buffer_different_dt(tmpsend, scount, sdtype, tmprecv, rcount, rdtype);
#endif
    if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl;  }
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
    if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }
    step_scount *= 2;

  }
  
  if(reorder_blocks(rbuf, rcount * rext, permutation, size) != MPI_SUCCESS){
    line = __LINE__;
    goto err_hndl;
  }

  return MPI_SUCCESS;

err_hndl:
  SWING_DEBUG_PRINT("\n%s:%4d\tRank %d Error occurred %d\n\n", __FILE__, line, rank, err);
  (void)line;  // silence compiler warning
  return err;
}


int allgather_swing_static_send(const void *sbuf, size_t scount, MPI_Datatype sdtype,
                                void* rbuf, size_t rcount, MPI_Datatype rdtype, MPI_Comm comm)
{
  int line = -1, rank, size, steps, err = MPI_SUCCESS, remote;
  const int *s_bitmap = NULL, *r_bitmap = NULL;
  int *permutation = NULL;
  ptrdiff_t rlb, rext;
  char *tmpsend = NULL, *tmprecv = NULL;

  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);


  /*
   * Current implementation only handles power-of-two number of processes.
   */
  steps = log_2(size);
  if(!is_power_of_two(size) || steps < 1) {
    SWING_DEBUG_PRINT("ERROR! Swing static allgather works only with po2 ranks!");
    return MPI_ERR_ARG;
  }
  

  err = MPI_Type_get_extent (rdtype, &rlb, &rext);
  if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }

  if(get_static_bitmap(&s_bitmap, &r_bitmap, steps, size, rank) == -1 ||
     get_perm_bitmap(&permutation, steps, size) == -1){
    line = __LINE__;
    goto err_hndl;
  }

  /* Initialization step:
   * - if I gather the result for another rank, I send my buffer to that rank
   *   and I receive the data from the rank at the inverse permutation
   * - if I gather the result for myself, I copy the data from the send buffer
   */
  if(permutation[rank] != rank){
    tmprecv = (char*) rbuf + (ptrdiff_t)permutation[rank] * (ptrdiff_t)rcount * rext;
    err = MPI_Sendrecv(sbuf, scount, sdtype, get_sender(permutation, size, rank), 0,
                       tmprecv, rcount, rdtype, permutation[rank], 0,
                       comm, MPI_STATUS_IGNORE);
    if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }
  }
  else{
    tmprecv = (char*) rbuf + (ptrdiff_t)permutation[rank] * (ptrdiff_t)rcount * rext;

    err = copy_buffer_different_dt(sbuf, scount, sdtype, tmprecv, rcount, rdtype);
    if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl;  }
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
    if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }
    step_scount *= 2;

  }

  return MPI_SUCCESS;

err_hndl:
  SWING_DEBUG_PRINT("\n%s:%4d\tRank %d Error occurred %d\n\n", __FILE__, line, rank, err);
  (void)line;  // silence compiler warning
  return err;
}


int allgather_swing_remap_memcpy(const void *sbuf, size_t scount, MPI_Datatype sdtype,
                           void* rbuf, size_t rcount, MPI_Datatype rdtype, MPI_Comm comm)
{
  int line = -1, rank, size, steps, err = MPI_SUCCESS;
  int vrank, remote, vremote, sendblocklocation, distance;
  int *remap = NULL;
  ptrdiff_t rlb, rext;
  char *tmpsend = NULL, *tmprecv = NULL;

  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  /*
   * Current implementation only handles power-of-two number of processes.
   */
  steps = log_2(size);
  if(!is_power_of_two(size) || steps < 1) {
    SWING_DEBUG_PRINT("ERROR! Swing static allgather works only with po2 ranks!");
    return MPI_ERR_ARG;
  }
  
  if(get_remap_bitmap(&remap, steps, size) == -1){
    line = __LINE__;
    goto err_hndl;
  }

  err = MPI_Type_get_extent (rdtype, &rlb, &rext);
  if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }


  vrank = (int) remap_rank((uint32_t) size, (uint32_t) rank);
  /* Initialization step:
     - if send buffer is not MPI_IN_PLACE, copy send buffer to block  of
     receive buffer
  */
  if(MPI_IN_PLACE != sbuf) {
    tmpsend = (char*) sbuf;
    tmprecv = (char*) rbuf + (ptrdiff_t)vrank * (ptrdiff_t)rcount * rext;

#ifdef CUDA_AWARE
    err = copy_buffer_different_dt_cuda(tmpsend, scount, sdtype, tmprecv, rcount, rdtype);
#else
    err = copy_buffer_different_dt(tmpsend, scount, sdtype, tmprecv, rcount, rdtype);
#endif
    if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl;  }
  }

  /* Communication step:
     At every step i, rank r:
     - exchanges message with rank remote = (r ^ 2^i).

  */
  distance = 0x1;
  sendblocklocation = vrank;
  for(int step = steps - 1; step >= 0; step--) {
    size_t step_scount = rcount * distance;
    remote = pi(rank, step, size);
    vremote = (int) remap_rank((uint32_t) size, (uint32_t) remote);

    if(vrank < vremote){
      tmpsend = (char*)rbuf + (ptrdiff_t)sendblocklocation * (ptrdiff_t)rcount * rext;
      tmprecv = (char*)rbuf + (ptrdiff_t)(sendblocklocation + distance) * (ptrdiff_t)rcount * rext;
    } else {
      tmpsend = (char*)rbuf + (ptrdiff_t)sendblocklocation * (ptrdiff_t)rcount * rext;
      tmprecv = (char*)rbuf + (ptrdiff_t)(sendblocklocation - distance) * (ptrdiff_t)rcount * rext;
      sendblocklocation -= distance;
    }

    /* Sendreceive */
    err = MPI_Sendrecv(tmpsend, step_scount, rdtype, remote, 0, 
                       tmprecv, step_scount, rdtype, remote, 0,
                       comm, MPI_STATUS_IGNORE);
    if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }
    distance <<=1;
  } 

  if(reorder_blocks(rbuf, rcount * rext, remap, size) != MPI_SUCCESS){
    line = __LINE__;
    goto err_hndl;
  }

  return MPI_SUCCESS;

err_hndl:
  SWING_DEBUG_PRINT("\n%s:%4d\tRank %d Error occurred %d\n\n", __FILE__, line, rank, err);
  (void)line;  // silence compiler warning
  return err;
}


int allgather_swing_remap_send(const void *sbuf, size_t scount, MPI_Datatype sdtype,
                           void* rbuf, size_t rcount, MPI_Datatype rdtype, MPI_Comm comm)
{
  int line = -1, rank, size, steps, err = MPI_SUCCESS;
  int vrank, remote, vremote, sendblocklocation, distance;
  int *remap = NULL;
  ptrdiff_t rlb, rext;
  char *tmpsend = NULL, *tmprecv = NULL;

  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  /*
   * Current implementation only handles power-of-two number of processes.
   */
  steps = log_2(size);
  if(!is_power_of_two(size) || steps < 1) {
    SWING_DEBUG_PRINT("ERROR! Swing static allgather works only with po2 ranks!");
    return MPI_ERR_ARG;
  }
  
  if(get_remap_bitmap(&remap, steps, size) == -1){
    line = __LINE__;
    goto err_hndl;
  }

  err = MPI_Type_get_extent (rdtype, &rlb, &rext);
  if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }

  /* Initialization step:
   * - if I gather the result for another rank, I send my buffer to that rank
   *   and I receive the data from the rank at the inverse permutation
   * - if I gather the result for myself, I copy the data from the send buffer
   */
  vrank = (int) remap_rank((uint32_t) size, (uint32_t) rank);
  if(vrank != rank){
    tmprecv = (char*) rbuf + (ptrdiff_t)vrank * (ptrdiff_t)rcount * rext;
    err = MPI_Sendrecv(sbuf, scount, sdtype, get_sender(remap, size, rank), 0,
                       tmprecv, rcount, rdtype, vrank, 0,
                       comm, MPI_STATUS_IGNORE);
    if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }
  }
  else{
    tmpsend = (char*) sbuf;
    tmprecv = (char*) rbuf + (ptrdiff_t)vrank * (ptrdiff_t)rcount * rext;
#ifdef CUDA_AWARE
    err = copy_buffer_different_dt_cuda(tmpsend, scount, sdtype, tmprecv, rcount, rdtype);
#else
    err = copy_buffer_different_dt(tmpsend, scount, sdtype, tmprecv, rcount, rdtype);
#endif
    if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl;  }
  }

  /* Communication step:
     At every step i, rank r:
     - exchanges message with rank remote = (r ^ 2^i).
  */
  distance = 0x1;
  sendblocklocation = vrank;
  for(int step = steps - 1; step >= 0; step--) {
    size_t step_scount = rcount * distance;
    remote = pi(rank, step, size);
    vremote = (int) remap_rank((uint32_t) size, (uint32_t) remote);

    if(vrank < vremote){
      tmpsend = (char*)rbuf + (ptrdiff_t)sendblocklocation * (ptrdiff_t)rcount * rext;
      tmprecv = (char*)rbuf + (ptrdiff_t)(sendblocklocation + distance) * (ptrdiff_t)rcount * rext;
    } else {
      tmpsend = (char*)rbuf + (ptrdiff_t)sendblocklocation * (ptrdiff_t)rcount * rext;
      tmprecv = (char*)rbuf + (ptrdiff_t)(sendblocklocation - distance) * (ptrdiff_t)rcount * rext;
      sendblocklocation -= distance;
    }

    /* Sendreceive */
    err = MPI_Sendrecv(tmpsend, step_scount, rdtype, remote, 0, 
                       tmprecv, step_scount, rdtype, remote, 0,
                       comm, MPI_STATUS_IGNORE);
    if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }
    distance <<=1;
  } 

  return MPI_SUCCESS;

err_hndl:
  SWING_DEBUG_PRINT("\n%s:%4d\tRank %d Error occurred %d\n\n", __FILE__, line, rank, err);
  (void)line;  // silence compiler warning
  return err;
}


int allgather_swing_no_remap(const void *sbuf, size_t scount, MPI_Datatype sdtype,
                           void* rbuf, size_t rcount, MPI_Datatype rdtype, MPI_Comm comm)
{
  int line = -1, rank, size, steps, err = MPI_SUCCESS, remote;
  int mask, my_first, recv_index, send_index;
  int send_count, recv_count, extra_send, extra_recv, extra_tag;
  ptrdiff_t rlb, rext;
  char *tmpsend = NULL, *tmprecv = NULL;

  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  /*
   * Current implementation only handles power-of-two number of processes.
   */
  steps = log_2(size);
  if(!is_power_of_two(size) || steps < 1) {
    SWING_DEBUG_PRINT("ERROR! Swing static allgather works only with po2 ranks!");
    return MPI_ERR_ARG;
  }

  err = MPI_Type_get_extent (rdtype, &rlb, &rext);
  if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }

  /* Initialization step:
     - if send buffer is not MPI_IN_PLACE, copy send buffer to block  of
     receive buffer
  */
  if(MPI_IN_PLACE != sbuf) {
    tmpsend = (char*) sbuf;
    tmprecv = (char*) rbuf + (ptrdiff_t)rank * (ptrdiff_t)rcount * rext;

#ifdef CUDA_AWARE
    err = copy_buffer_different_dt_cuda(tmpsend, scount, sdtype, tmprecv, rcount, rdtype);
#else
    err = copy_buffer_different_dt(tmpsend, scount, sdtype, tmprecv, rcount, rdtype);
#endif
    if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl;  }
  }


  /* Communication step.
   *  At every step i, rank r:
   *  - communication peer is calculated by pi(rank, step, size)
   *  - if the step is even, even ranks send the next `mask` blocks and
   *  odd ranks send the previous `mask` blocks.
   *  - if the step is odd, even ranks send the previous `mask` blocks and
   *  odd ranks send the next `mask` blocks.
   */
  mask = 0x1;
  my_first = rank;
  extra_tag = 1;
  for(int step = 0; step < steps; step++) {
    MPI_Request req;
    remote = pi(rank, step, size);
    send_index = my_first;

    // Calculate the send and receive indexes by alternating send/recv direction.
    if ((step & 1) == (rank & 1)) {
        recv_index = (send_index + mask + size) % size;
    } else {
        recv_index = (send_index - mask + size) % size;
        my_first = recv_index;
    }

    // Control if the previously calculated indexes imply out of bound
    // send/recv. If so, split the communication with an extra send/recv.
    extra_recv = (recv_index + mask > size) ? ((recv_index + mask) - size) : 0;
    recv_count = mask - extra_recv;

    extra_send = (send_index + mask > size) ? ((send_index + mask) - size) : 0;
    send_count = mask - extra_send;

    // warparound communication
    if (extra_recv != 0){
      tmprecv = (char*)rbuf;
      err = MPI_Irecv(tmprecv, extra_recv * rcount, rdtype, remote, extra_tag, comm, &req);
      if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }
    }
    if (extra_send != 0){
      tmpsend = (char*)rbuf;
      err = MPI_Send(tmpsend, extra_send * rcount, rdtype, remote, extra_tag, comm);
      if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }
    }

    // Simple case: no wrap-around
    tmpsend = (char*)rbuf + (ptrdiff_t)send_index * (ptrdiff_t)rcount * rext;
    tmprecv = (char*)rbuf + (ptrdiff_t)recv_index * (ptrdiff_t)rcount * rext;

    err = MPI_Sendrecv(tmpsend, send_count * rcount, rdtype, remote, 0, 
                       tmprecv, recv_count * rcount, rdtype, remote, 0,
                       comm, MPI_STATUS_IGNORE);
    if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }
    
    if (extra_recv != 0) {
      err = MPI_Wait(&req, MPI_STATUS_IGNORE);
      if (MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }
    }

    mask <<= 1;
  }

  return MPI_SUCCESS;

err_hndl:
  SWING_DEBUG_PRINT("\n%s:%4d\tRank %d Error occurred %d\n\n", __FILE__, line, rank, err);
  (void)line;  // silence compiler warning
  return err;
}


int allgather_swing_no_remap_dtype(const void *sbuf, size_t scount, MPI_Datatype sdtype,
                           void* rbuf, size_t rcount, MPI_Datatype rdtype, MPI_Comm comm)
{
  int line = -1, rank, size, steps, err = MPI_SUCCESS, remote;
  int mask, my_first, recv_index, send_index;
  int send_count, recv_count, extra_send, extra_recv;
  ptrdiff_t rlb, rext;
  char *tmpsend = NULL, *tmprecv = NULL;

  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  /*
   * Current implementation only handles power-of-two number of processes.
   */
  steps = log_2(size);
  if(!is_power_of_two(size) || steps < 1) {
    SWING_DEBUG_PRINT("ERROR! Swing static allgather works only with po2 ranks!");
    return MPI_ERR_ARG;
  }

  err = MPI_Type_get_extent (rdtype, &rlb, &rext);
  if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }

  /* Initialization step:
     - if send buffer is not MPI_IN_PLACE, copy send buffer to block  of
     receive buffer
  */
  if(MPI_IN_PLACE != sbuf) {
    tmpsend = (char*) sbuf;
    tmprecv = (char*) rbuf + (ptrdiff_t)rank * (ptrdiff_t)rcount * rext;

#ifdef CUDA_AWARE
    err = copy_buffer_different_dt_cuda(tmpsend, scount, sdtype, tmprecv, rcount, rdtype);
#else
    err = copy_buffer_different_dt(tmpsend, scount, sdtype, tmprecv, rcount, rdtype);
#endif
    if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl;  }
  }


  /* Communication step.
   *  At every step i, rank r:
   *  - communication peer is calculated by pi(rank, step, size)
   *  - if the step is even, even ranks send the next `mask` blocks and
   *  odd ranks send the previous `mask` blocks.
   *  - if the step is odd, even ranks send the previous `mask` blocks and
   *  odd ranks send the next `mask` blocks.
   */
  mask = 0x1;
  my_first = rank;
  for(int step = 0; step < steps; step++) {
    MPI_Datatype send_dtype = MPI_DATATYPE_NULL, recv_dtype = MPI_DATATYPE_NULL;
    remote = pi(rank, step, size);
    send_index = my_first;

    // Calculate the send and receive indexes by alternating send/recv direction
    if ((step & 1) == (rank & 1)) {
        recv_index = (send_index + mask + size) % size;
    } else {
        recv_index = (send_index - mask + size) % size;
        my_first = recv_index;
    }

    // Control if the previously calculated indexes imply out of bound
    // send/recv.
    extra_recv = (recv_index + mask > size) ? ((recv_index + mask) - size) : 0;
    recv_count = mask - extra_recv;
    extra_send = (send_index + mask > size) ? ((send_index + mask) - size) : 0;
    send_count = mask - extra_send;

    if (extra_recv == 0 && extra_send == 0){
      // Simple case: no wrap-around, use a simple MPI_Sendrecv
      tmpsend = (char*)rbuf + (ptrdiff_t)send_index * (ptrdiff_t)rcount * rext;
      tmprecv = (char*)rbuf + (ptrdiff_t)recv_index * (ptrdiff_t)rcount * rext;

      err = MPI_Sendrecv(tmpsend, send_count * rcount, rdtype, remote, 0, 
                        tmprecv, recv_count * rcount, rdtype, remote, 0,
                        comm, MPI_STATUS_IGNORE);
    }
    else{
      // Handles warp around communication with derived datatypes
      tmpsend = (char*)rbuf;
      tmprecv = (char*)rbuf;
      if (extra_recv > 0){
        int recv_blocklengths[2] = {extra_recv * rcount, recv_count * rcount};
        int recv_displacements[2] = {0, recv_index * rcount};
        MPI_Type_indexed(2, recv_blocklengths, recv_displacements, rdtype, &recv_dtype);
      } else {
        MPI_Type_contiguous(recv_count * rcount, rdtype, &recv_dtype);
        tmprecv = (char *)rbuf + (ptrdiff_t)recv_index * (ptrdiff_t)rcount * rext;
      }
      MPI_Type_commit(&recv_dtype);

      if (extra_send > 0){
        int send_blocklengths[2] = {extra_send * rcount, send_count * rcount};
        int send_displacements[2] = {0, send_index * rcount};
        MPI_Type_indexed(2, send_blocklengths, send_displacements, rdtype, &send_dtype);
      } else {
        MPI_Type_contiguous(send_count * rcount, rdtype, &send_dtype);
        tmpsend = (char *)rbuf + (ptrdiff_t)send_index * (ptrdiff_t)rcount * rext;
      }
      MPI_Type_commit(&send_dtype);
      
      err = MPI_Sendrecv(tmpsend, 1, send_dtype, remote, 0, 
                        tmprecv, 1, recv_dtype, remote, 0,
                        comm, MPI_STATUS_IGNORE);

      MPI_Type_free(&send_dtype);
      MPI_Type_free(&recv_dtype);
    }

    // this controls the error message of both the MPI_Sendrecv
    if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }

    mask <<= 1;
  }

  return MPI_SUCCESS;

err_hndl:
  SWING_DEBUG_PRINT("\n%s:%4d\tRank %d Error occurred %d\n\n", __FILE__, line, rank, err);
  (void)line;  // silence compiler warning
  return err;
}


// ---------------------------------------------------
// MODIFICATIONS INTRODUCTED BY LORENZO
// 
// The following implementations are not implemented in the framework yet.
//
// ---------------------------------------------------

// TOCCA METTERE TUTTO IN CUDA

static inline int permute_blocks(void *buffer, size_t block_size, int *block_permutation, int num_blocks) {

  char* tmp_buffer;
#ifdef CUDA_AWARE
  SWING_CUDA_CHECK(cudaMalloc((void**)&tmp_buffer, block_size * num_blocks));
  SWING_CUDA_CHECK(cudaMemset(tmp_buffer, 0, block_size * num_blocks));
#else
  tmp_buffer = (char*) malloc(block_size * num_blocks);
#endif

  if (!tmp_buffer) {
      fprintf(stderr, "Memory allocation failed\n");
      return MPI_ERR_NO_MEM;
  }

  for (int i = 0; i < num_blocks; ++i) {
      memcpy(tmp_buffer + block_permutation[i] * block_size, (char*)buffer + i * block_size, block_size);
  }

  memcpy(buffer, tmp_buffer, block_size * num_blocks);
  free(tmp_buffer);
  return MPI_SUCCESS;
}

// AUXILIARY FUNCTION USED TO FIND PERMUTATIONS

int allgather_swing_find_permutation(const void *sbuf, size_t scount, MPI_Datatype sdtype, 
  void* rbuf, size_t rcount, MPI_Datatype rdtype, MPI_Comm comm) {

  int rank, size, step, steps, send_rank, recv_rank;
  MPI_Aint lb, extent;
  char *sendbuf_off = (char*) sbuf, *recvbuf_off = (char*) rbuf;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Type_get_extent(sdtype, &lb, &extent);

  memcpy(recvbuf_off, sendbuf_off, rcount * extent);

  steps = log_2(size);
  for(step = 0; step < steps; ++step) {

      int powStep = 1 << step;;
      int negpowStep = -1 << (step+1);

      if(rank % 2 == 0){
          send_rank = (int)((rank + (1-1*negpowStep)/3) + size) % size; 
          recv_rank = send_rank; 
      } else {
          send_rank = (int)((rank - (1-1*negpowStep)/3) + size) % size;
          recv_rank = send_rank; 
      }   

      sendbuf_off = (char*) sbuf;
      recvbuf_off = (char*) rbuf + (ptrdiff_t) powStep * (ptrdiff_t) rcount * extent;
  

      MPI_Sendrecv(sendbuf_off, rcount * powStep, rdtype, send_rank, 0,
      recvbuf_off, rcount * powStep, rdtype, recv_rank, 0, comm, MPI_STATUS_IGNORE);

  }

  return MPI_SUCCESS;
}

// ALLGATHER IMPLEMENTATION USING PERMUTATION PRECOMPUTED

int allgather_swing_permute_require(const void *sbuf, size_t scount, MPI_Datatype sdtype, 
  void* rbuf, size_t rcount, MPI_Datatype rdtype, MPI_Comm comm, int* permutation) {

  int rank, size, step, steps, send_rank, recv_rank;
  MPI_Aint lb, extent;
  char *sendbuf_off = (char*) sbuf, *recvbuf_off = (char*) rbuf;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Type_get_extent(sdtype, &lb, &extent);

  memcpy(recvbuf_off, sendbuf_off, rcount * extent);

  steps = log_2(size);
  for(step = 0; step < steps; ++step) {

      int powStep = 1 << step;;
      int negpowStep = -1 << (step+1);

      if(rank % 2 == 0){
          send_rank = (int)((rank + (1-1*negpowStep)/3) + size) % size; 
          recv_rank = send_rank; 
      } else {
          send_rank = (int)((rank - (1-1*negpowStep)/3) + size) % size;
          recv_rank = send_rank; 
      }   

      sendbuf_off = (char*) sbuf;
      recvbuf_off = (char*) rbuf + (ptrdiff_t) powStep * (ptrdiff_t) rcount * extent;
  

      MPI_Sendrecv(sendbuf_off, rcount * powStep, rdtype, send_rank, 0,
      recvbuf_off, rcount * powStep, rdtype, recv_rank, 0, comm, MPI_STATUS_IGNORE);

  }
  
  reorder_blocks(rbuf, rcount * extent, permutation, size);

  return MPI_SUCCESS;
}