/*
 * Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
 * Licensed under the MIT License
 */

#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <unistd.h>

#include "libbine.h"
#include "libbine_utils.h"
#include "libbine_utils_bitmaps.h"



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
    BINE_DEBUG_PRINT("ERROR! Recoursive doubling allgather works only with po2 ranks!");
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

    err = COPY_BUFF_DIFF_DT(tmpsend, scount, sdtype, tmprecv, rcount, rdtype);

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
  BINE_DEBUG_PRINT("\n%s:%4d\tRank %d Error occurred %d\n\n", __FILE__, line, rank, err);
  (void)line;  // silence compiler warning
  return err;
}

int allgather_k_bruck(const void *sbuf, size_t scount, MPI_Datatype sdtype,
                      void* rbuf, size_t rcount, MPI_Datatype rdtype, MPI_Comm comm)
{
  int line = -1, rank, size, dst, src, err = MPI_SUCCESS;
  int recvcount, distance, radix = 2;
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
    BINE_CUDA_CHECK(cudaMalloc((void**)&tmp_buf, rsize));
    BINE_CUDA_CHECK(cudaMemset(tmp_buf, 0, rsize));
#else
    tmp_buf = (char *) malloc(rsize);
#endif

    tmp_buf_start = tmp_buf - rgap;
  }

  // tmprecv points to the data initially on this rank, handle mpi_in_place case
  tmprecv = (char*) rbuf;
  if(MPI_IN_PLACE != sbuf) {
    tmpsend = (char*) sbuf;

    err = COPY_BUFF_DIFF_DT(tmpsend, scount, sdtype, tmprecv, rcount, rdtype);

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
  BINE_DEBUG_PRINT( "\n%s:%4d\tError occurred %d, rank %2d\n\n", __FILE__, line, err, rank);
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

    err = COPY_BUFF_DIFF_DT(tmpsend, scount, sdtype, tmprecv, rcount, rdtype);

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
  BINE_DEBUG_PRINT("\n%s:%4d\tError occurred %d, rank %2d\n\n", __FILE__, line, err, rank);
  (void)line;  // silence compiler warning
  return err;
}


// Taken from OpenMPI coll/base
/*
 * ompi_coll_base_allgather_intra_sparbit
 *
 * Function:     allgather using O(log(N)) steps.
 * Accepts:      Same arguments as MPI_Allgather
 * Returns:      MPI_SUCCESS or error code
 *
 * Description: Proposal of an allgather algorithm similar to Bruck but with inverted distances
 *              and non-decreasing exchanged data sizes. Described in "Sparbit: a new
 *              logarithmic-cost and data locality-aware MPI Allgather algorithm".
 *
 * Memory requirements:  
 *              Additional memory for N requests. 
 *
 * Example on 6 nodes, with l representing the highest power of two smaller than N, in this case l =
 * 4 (more details can be found on the paper):
 *  Initial state
 *    #     0      1      2      3      4      5
 *         [0]    [ ]    [ ]    [ ]    [ ]    [ ]
 *         [ ]    [1]    [ ]    [ ]    [ ]    [ ]
 *         [ ]    [ ]    [2]    [ ]    [ ]    [ ]
 *         [ ]    [ ]    [ ]    [3]    [ ]    [ ]
 *         [ ]    [ ]    [ ]    [ ]    [4]    [ ]
 *         [ ]    [ ]    [ ]    [ ]    [ ]    [5]
 *   Step 0: Each process sends its own block to process r + l and receives another from r - l.
 *    #     0      1      2      3      4      5
 *         [0]    [ ]    [ ]    [ ]    [0]    [ ]
 *         [ ]    [1]    [ ]    [ ]    [ ]    [1]
 *         [2]    [ ]    [2]    [ ]    [ ]    [ ]
 *         [ ]    [3]    [ ]    [3]    [ ]    [ ]
 *         [ ]    [ ]    [4]    [ ]    [4]    [ ]
 *         [ ]    [ ]    [ ]    [5]    [ ]    [5]
 *   Step 1: Each process sends its own block to process r + l/2 and receives another from r - l/2.
 *   The block received on the previous step is ignored to avoid a future double-write.  
 *    #     0      1      2      3      4      5
 *         [0]    [ ]    [0]    [ ]    [0]    [ ]
 *         [ ]    [1]    [ ]    [1]    [ ]    [1]
 *         [2]    [ ]    [2]    [ ]    [2]    [ ]
 *         [ ]    [3]    [ ]    [3]    [ ]    [3]
 *         [4]    [ ]    [4]    [ ]    [4]    [ ]
 *         [ ]    [5]    [ ]    [5]    [ ]    [5]
 *   Step 1: Each process sends all the data it has (3 blocks) to process r + l/4 and similarly
 *   receives all the data from process r - l/4. 
 *    #     0      1      2      3      4      5
 *         [0]    [0]    [0]    [0]    [0]    [0]
 *         [1]    [1]    [1]    [1]    [1]    [1]
 *         [2]    [2]    [2]    [2]    [2]    [2]
 *         [3]    [3]    [3]    [3]    [3]    [3]
 *         [4]    [4]    [4]    [4]    [4]    [4]
 *         [5]    [5]    [5]    [5]    [5]    [5]
 */

int allgather_sparbit(const void *sbuf, size_t scount, MPI_Datatype sdtype, void* rbuf,
                      size_t rcount, MPI_Datatype rdtype, MPI_Comm comm)
{
  /* ################# VARIABLE DECLARATION, BUFFER CREATION AND PREPARATION FOR THE ALGORITHM ######################## */

  /* list of variable declaration */
  int rank = 0, comm_size = 0, comm_log = 0, exclusion = 0, data_expected = 1, transfer_count = 0;
  int sendto, recvfrom, send_disp, recv_disp;
  uint32_t last_ignore, ignore_steps, distance = 1;

  int err = 0;
  int line = -1;

  ptrdiff_t rlb, rext;

  char *tmpsend = NULL, *tmprecv = NULL;

  MPI_Request *requests = NULL;

  /* algorithm choice information printing */
  MPI_Comm_size(comm, &comm_size);
  MPI_Comm_rank(comm, &rank);

  err = MPI_Type_get_extent(rdtype, &rlb, &rext);
  if (MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }

  /* if the MPI_IN_PLACE condition is not set, copy the send buffer to the receive buffer to perform the sends (all the data is extracted and forwarded from the recv buffer)*/
  /* tmprecv and tmpsend are used as abstract pointers to simplify send and receive buffer choice */
  tmprecv = (char *) rbuf;
  if(MPI_IN_PLACE != sbuf){
    tmpsend = (char *) sbuf;
    err = copy_buffer_different_dt(tmpsend, scount, sdtype, tmprecv + (ptrdiff_t) rank * rcount * rext, rcount, rdtype);
    if (MPI_SUCCESS != err) { line = __LINE__; goto err_hndl;  }
  }
  tmpsend = tmprecv;

  requests = (MPI_Request *) malloc(comm_size * sizeof(MPI_Request));
  
  /* ################# ALGORITHM LOGIC ######################## */

  /* calculate log2 of the total process count */
  comm_log = log_2(comm_size);
  distance <<= comm_log - 1;

  last_ignore = __builtin_ctz(comm_size);
  ignore_steps = (~((uint32_t) comm_size >> last_ignore) | 1) << last_ignore;

  /* perform the parallel binomial tree distribution steps */
  for (int i = 0; i < comm_log; ++i) {
    sendto = (rank + distance) % comm_size;  
    recvfrom = (rank - distance + comm_size) % comm_size;  
    exclusion = (distance & ignore_steps) == distance;

    for (transfer_count = 0; transfer_count < data_expected - exclusion; transfer_count++) {
      send_disp = (rank - 2 * transfer_count * distance + comm_size) % comm_size;
      recv_disp = (rank - (2 * transfer_count + 1) * distance + comm_size) % comm_size;

       /* Since each process sends several non-contiguos blocks of data, each block sent (and therefore each send and recv call) needs a different tag. */
       /* As base OpenMPI only provides one tag for allgather, we are forced to use a tag space from other components in the send and recv calls */
      err = MPI_Isend(tmpsend + (ptrdiff_t) send_disp * scount * rext, scount, rdtype, sendto, send_disp, comm, requests + transfer_count);
      if (MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }

      err = MPI_Irecv(tmprecv + (ptrdiff_t) recv_disp * rcount * rext, rcount, rdtype, recvfrom, recv_disp, comm, requests + data_expected - exclusion + transfer_count);
      if (MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }
     }

    err = MPI_Waitall(transfer_count * 2, requests, MPI_STATUSES_IGNORE);
    distance >>= 1; 
     /* calculates the data expected for the next step, based on the current number of blocks and eventual exclusions */
    data_expected = (data_expected << 1) - exclusion;
    exclusion = 0;
  }

  free(requests);

  return MPI_SUCCESS;

err_hndl:
  BINE_DEBUG_PRINT("\n%s:%4d\tError occurred %d, rank %2d\n\n", __FILE__, line, err, rank);
  (void)line;  // silence compiler warning
  return err;
}

int allgather_bine_block_by_block(const void *sbuf, size_t scount, MPI_Datatype sdtype,
                           void* rbuf, size_t rcount, MPI_Datatype rdtype, MPI_Comm comm){
  int line = -1, rank, size, steps, err = MPI_SUCCESS, remote;
  int *s_bitmap = NULL, *r_bitmap = NULL;
  ptrdiff_t rlb, rext;
  char *tmpsend = NULL, *tmprecv = NULL;
  MPI_Request *requests = NULL;

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  steps = log_2(size);
  if(!is_power_of_two(size) || steps < 1) {
    BINE_DEBUG_PRINT("ERROR! bine static allgather works only with po2 ranks!");
    return MPI_ERR_ARG;
  }

  err = MPI_Type_get_extent (rdtype, &rlb, &rext);
  if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }

  if(MPI_IN_PLACE != sbuf) {
    tmpsend = (char*) sbuf;
    tmprecv = (char*) rbuf + (ptrdiff_t)rank * (ptrdiff_t)rcount * rext;

    err = copy_buffer_different_dt(tmpsend, scount, sdtype, tmprecv, rcount, rdtype);
    if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl;  }
  }

  s_bitmap = (int *) malloc(size * sizeof(int));
  r_bitmap = (int *) malloc(size * sizeof(int));
  requests = (MPI_Request *) malloc(size * sizeof(MPI_Request));
  if(s_bitmap == NULL || r_bitmap == NULL || requests == NULL){
    line = __LINE__;
    err = MPI_ERR_NO_MEM;
    goto err_hndl;
  }

  for(int step = steps - 1; step >= 0; step--) {
    int num_reqs = 0;
    remote = pi(rank, step, size);

    memset(s_bitmap, 0, size * sizeof(int));
    memset(r_bitmap, 0, size * sizeof(int));
    get_indexes(rank, step, steps, size, r_bitmap);
    get_indexes(remote, step, steps, size, s_bitmap);

    for(int block = 0; block < size; block++){
      if(s_bitmap[block] != 0){
        tmpsend = (char*)rbuf + (ptrdiff_t)block * (ptrdiff_t)rcount * rext;
        err = MPI_Isend(tmpsend, rcount, rdtype, remote, block, comm, requests + num_reqs);
        if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }
        num_reqs++;
      }
      if(r_bitmap[block] != 0){
        tmprecv = (char*)rbuf + (ptrdiff_t)block * (ptrdiff_t)rcount * rext;
        err = MPI_Irecv(tmprecv, rcount, rdtype, remote, block, comm, requests + num_reqs);
        if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }
        num_reqs++;
      }
    }

    err = MPI_Waitall(num_reqs, requests, MPI_STATUSES_IGNORE);
    if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }
  }


  free(s_bitmap);
  free(r_bitmap);
  free(requests);

  return MPI_SUCCESS;

err_hndl:
  BINE_DEBUG_PRINT("\n%s:%4d\tError occurred %d, rank %2d\n\n", __FILE__, line, err, rank);
  (void)line;  // silence compiler warning
  if(requests != NULL) free(requests);
  if(s_bitmap != NULL) free(s_bitmap);
  if(r_bitmap != NULL) free(r_bitmap);
  return err;
}


int allgather_bine_block_by_block_any_even(const void *sendbuf, size_t sendcount, MPI_Datatype sendtype,
                                            void* recvbuf, size_t recvcount, MPI_Datatype recvtype, MPI_Comm comm)
{
  assert(sendcount == recvcount); // TODO: Implement the case where sendcount != recvcount
  assert(sendtype == recvtype); // TODO: Implement the case where sendtype != recvtype
  int size, rank, dtsize, err = MPI_SUCCESS;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  MPI_Type_size(recvtype, &dtsize);
  MPI_Request *requests = NULL;
  memcpy((char*) recvbuf + sendcount * rank * dtsize, sendbuf, sendcount * dtsize);

  int inverse_mask = 0x1 << (int) (log_2(size) - 1);
  int step = 0;

  requests = (MPI_Request *) malloc(2 * size * sizeof(MPI_Request));
  while(inverse_mask > 0){
    int partner, req_count = 0;
    if(rank % 2 == 0){
      partner = mod(rank + negabinary_to_binary((inverse_mask << 1) - 1), size); 
    }else{
      partner = mod(rank - negabinary_to_binary((inverse_mask << 1) - 1), size); 
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

        err = MPI_Isend((char*) recvbuf + block_to_send*sendcount*dtsize, sendcount, sendtype, partner_send, 0, comm, &requests[req_count++]);
        if(MPI_SUCCESS != err) { goto err_hndl; }

        err = MPI_Irecv((char*) recvbuf + block_to_recv*recvcount*dtsize, recvcount, recvtype, partner_recv, 0, comm, &requests[req_count++]);
        if(MPI_SUCCESS != err) { goto err_hndl; }
      }
    }
    err = MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
    inverse_mask >>= 1;
    step++;
  }

  free(requests);
  return MPI_SUCCESS;

err_hndl:
  if (requests != NULL) free(requests);
  return err;
}

int allgather_bine_permute_static(const void *sbuf, size_t scount, MPI_Datatype sdtype,
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
    BINE_DEBUG_PRINT("ERROR! bine static allgather works only with po2 ranks!");
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

    err = copy_buffer_different_dt(tmpsend, scount, sdtype, tmprecv, rcount, rdtype);
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
  BINE_DEBUG_PRINT("\n%s:%4d\tRank %d Error occurred %d\n\n", __FILE__, line, rank, err);
  (void)line;  // silence compiler warning
  return err;
}


int allgather_bine_send_static(const void *sbuf, size_t scount, MPI_Datatype sdtype,
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
    BINE_DEBUG_PRINT("ERROR! bine static allgather works only with po2 ranks!");
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
  BINE_DEBUG_PRINT("\n%s:%4d\tRank %d Error occurred %d\n\n", __FILE__, line, rank, err);
  (void)line;  // silence compiler warning
  return err;
}


int allgather_bine_permute_remap(const void *sbuf, size_t scount, MPI_Datatype sdtype,
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
    BINE_DEBUG_PRINT("ERROR! bine static allgather works only with po2 ranks!");
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

    err = copy_buffer_different_dt(tmpsend, scount, sdtype, tmprecv, rcount, rdtype);
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
  BINE_DEBUG_PRINT("\n%s:%4d\tRank %d Error occurred %d\n\n", __FILE__, line, rank, err);
  (void)line;  // silence compiler warning
  return err;
}


int allgather_bine_send_remap(const void *sbuf, size_t scount, MPI_Datatype sdtype,
                           void* rbuf, size_t rcount, MPI_Datatype rdtype, MPI_Comm comm)
{
  int line = -1, rank, size, steps, err = MPI_SUCCESS;
  int vrank, remote, vremote, sendblocklocation, distance;
  ptrdiff_t rlb, rext;
  char *tmpsend = NULL, *tmprecv = NULL;

  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  /*
   * Current implementation only handles power-of-two number of processes.
   */
  steps = log_2(size);
  if(!is_power_of_two(size) || steps < 1) {
    BINE_DEBUG_PRINT("ERROR! bine static allgather works only with po2 ranks!");
    return MPI_ERR_ARG;
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
    err = MPI_Sendrecv(sbuf, scount, sdtype, get_sender_rec(size, rank), 0,
                       tmprecv, rcount, rdtype, vrank, 0,
                       comm, MPI_STATUS_IGNORE);
    if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }
  }
  else{
    tmpsend = (char*) sbuf;
    tmprecv = (char*) rbuf + (ptrdiff_t)vrank * (ptrdiff_t)rcount * rext;

    err = copy_buffer_different_dt(tmpsend, scount, sdtype, tmprecv, rcount, rdtype);
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
  BINE_DEBUG_PRINT("\n%s:%4d\tRank %d Error occurred %d\n\n", __FILE__, line, rank, err);
  (void)line;  // silence compiler warning
  return err;
}


int allgather_bine_2_blocks(const void *sbuf, size_t scount, MPI_Datatype sdtype,
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
    BINE_DEBUG_PRINT("ERROR! bine static allgather works only with po2 ranks!");
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

    err = COPY_BUFF_DIFF_DT(tmpsend, scount, sdtype, tmprecv, rcount, rdtype);
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
  BINE_DEBUG_PRINT("\n%s:%4d\tRank %d Error occurred %d\n\n", __FILE__, line, rank, err);
  (void)line;  // silence compiler warning
  return err;
}


int allgather_bine_2_blocks_dtype(const void *sbuf, size_t scount, MPI_Datatype sdtype,
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
    BINE_DEBUG_PRINT("ERROR! bine static allgather works only with po2 ranks!");
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

    err = COPY_BUFF_DIFF_DT(tmpsend, scount, sdtype, tmprecv, rcount, rdtype);
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
  BINE_DEBUG_PRINT("\n%s:%4d\tRank %d Error occurred %d\n\n", __FILE__, line, rank, err);
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

// static inline int permute_blocks(void *buffer, size_t block_size, int *block_permutation, int num_blocks) {
//
//   char* tmp_buffer;
// #ifdef CUDA_AWARE
//   BINE_CUDA_CHECK(cudaMalloc((void**)&tmp_buffer, block_size * num_blocks));
//   BINE_CUDA_CHECK(cudaMemset(tmp_buffer, 0, block_size * num_blocks));
// #else
//   tmp_buffer = (char*) malloc(block_size * num_blocks);
// #endif
//
//   if (!tmp_buffer) {
//       fprintf(stderr, "Memory allocation failed\n");
//       return MPI_ERR_NO_MEM;
//   }
//
//   for (int i = 0; i < num_blocks; ++i) {
//       memcpy(tmp_buffer + block_permutation[i] * block_size, (char*)buffer + i * block_size, block_size);
//   }
//
//   memcpy(buffer, tmp_buffer, block_size * num_blocks);
//   free(tmp_buffer);
//   return MPI_SUCCESS;
// }
//
// // AUXILIARY FUNCTION USED TO FIND PERMUTATIONS
//
// int allgather_bine_find_permutation(const void *sbuf, size_t scount, MPI_Datatype sdtype, 
//   void* rbuf, size_t rcount, MPI_Datatype rdtype, MPI_Comm comm) {
//
//   int rank, size, step, steps, send_rank, recv_rank;
//   MPI_Aint lb, extent;
//   char *sendbuf_off = (char*) sbuf, *recvbuf_off = (char*) rbuf;
//
//   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//   MPI_Comm_size(MPI_COMM_WORLD, &size);
//   MPI_Type_get_extent(sdtype, &lb, &extent);
//
//   memcpy(recvbuf_off, sendbuf_off, rcount * extent);
//
//   steps = log_2(size);
//   for(step = 0; step < steps; ++step) {
//
//       int powStep = 1 << step;;
//       int negpowStep = -1 << (step+1);
//
//       if(rank % 2 == 0){
//           send_rank = (int)((rank + (1-1*negpowStep)/3) + size) % size; 
//           recv_rank = send_rank; 
//       } else {
//           send_rank = (int)((rank - (1-1*negpowStep)/3) + size) % size;
//           recv_rank = send_rank; 
//       }   
//
//       sendbuf_off = (char*) sbuf;
//       recvbuf_off = (char*) rbuf + (ptrdiff_t) powStep * (ptrdiff_t) rcount * extent;
//   
//
//       MPI_Sendrecv(sendbuf_off, rcount * powStep, rdtype, send_rank, 0,
//       recvbuf_off, rcount * powStep, rdtype, recv_rank, 0, comm, MPI_STATUS_IGNORE);
//
//   }
//
//   return MPI_SUCCESS;
// }
//
// // ALLGATHER IMPLEMENTATION USING PERMUTATION PRECOMPUTED
//
// int allgather_bine_permute_require(const void *sbuf, size_t scount, MPI_Datatype sdtype, 
//   void* rbuf, size_t rcount, MPI_Datatype rdtype, MPI_Comm comm, int* permutation) {
//
//   int rank, size, step, steps, send_rank, recv_rank;
//   MPI_Aint lb, extent;
//   char *sendbuf_off = (char*) sbuf, *recvbuf_off = (char*) rbuf;
//
//   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//   MPI_Comm_size(MPI_COMM_WORLD, &size);
//   MPI_Type_get_extent(sdtype, &lb, &extent);
//
//   memcpy(recvbuf_off, sendbuf_off, rcount * extent);
//
//   steps = log_2(size);
//   for(step = 0; step < steps; ++step) {
//
//       int powStep = 1 << step;;
//       int negpowStep = -1 << (step+1);
//
//       if(rank % 2 == 0){
//           send_rank = (int)((rank + (1-1*negpowStep)/3) + size) % size; 
//           recv_rank = send_rank; 
//       } else {
//           send_rank = (int)((rank - (1-1*negpowStep)/3) + size) % size;
//           recv_rank = send_rank; 
//       }   
//
//       sendbuf_off = (char*) sbuf;
//       recvbuf_off = (char*) rbuf + (ptrdiff_t) powStep * (ptrdiff_t) rcount * extent;
//   
//
//       MPI_Sendrecv(sendbuf_off, rcount * powStep, rdtype, send_rank, 0,
//       recvbuf_off, rcount * powStep, rdtype, recv_rank, 0, comm, MPI_STATUS_IGNORE);
//
//   }
//   
//   reorder_blocks(rbuf, rcount * extent, permutation, size);
//
//   return MPI_SUCCESS;
// }
