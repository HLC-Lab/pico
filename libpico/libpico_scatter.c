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
 * Open MPI linear scatter function copied from 
 * https://github.com/open-mpi/ompi/blob/3a2e90895e15822912002be1e9aea8032c4c0bae/ompi/mca/coll/base/coll_base_scatter.c#L223
 */
int scatter_linear(const void *sbuf, size_t scount, MPI_Datatype sdtype, void *rbuf,
                   size_t rcount, MPI_Datatype rdtype, int root, MPI_Comm comm)
{
  int i, rank, size, err;
  MPI_Aint lb, s_ext, r_ext;
  MPI_Aint s_block_span, r_block_span;  /* bytes spanned per rank on send/recv sides */
  char *tmp_buf;

  /* Initialize */
  err = MPI_Comm_size(comm, &size);
  err = MPI_Comm_rank(comm, &rank);

  /* If not root, receive data. */
  if (rank != root) {
    err = MPI_Recv(rbuf, (int)rcount, rdtype, root, 0, comm, MPI_STATUS_IGNORE);
    return err;
  }

  /* I am the root, loop sending data. */
  /* Root: compute strides/spans */
  MPI_Type_get_extent(sdtype, &lb, &s_ext);
  MPI_Type_get_extent(rdtype, &lb, &r_ext);
  assert(lb == 0); /* should be true for basic types */

  s_block_span = (MPI_Aint)scount * s_ext;  /* stride between neighbors in sbuf */
  r_block_span = (MPI_Aint)rcount * r_ext;  /* max we should touch in rbuf    */

  tmp_buf = (char *)sbuf;
  for (i = 0; i < size; ++i) {
    tmp_buf = (char *)sbuf + (MPI_Aint)i * s_block_span;

    if (i == rank) {
      if (rbuf == MPI_IN_PLACE) { continue; }

      MPI_Aint n = s_block_span < r_block_span ? s_block_span : r_block_span;
      memcpy(rbuf, tmp_buf, (size_t)n);
    } else {
      err = MPI_Send(tmp_buf, (int)scount, sdtype, i, 0, comm);
      if (MPI_SUCCESS != err) return err;
    }
  }
  /* All done */

  return MPI_SUCCESS;
}

int scatter_bine(const void *sendbuf, size_t sendcount, MPI_Datatype dt,
                  void *recvbuf, size_t recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)
{
  assert(sendcount == recvcount); // TODO: Implement the case where sendcount != recvcount
  assert(dt == recvtype); // TODO: Implement the case where sendtype != recvtype

  int size, rank, dtsize, err = MPI_SUCCESS;
  int vrank, halving_direction, mask, recvd = 0, is_leaf = 0;
  int sbuf_offset, vrank_nb;
  size_t min_resident_block, max_resident_block;
  char *tmpbuf = NULL, *sbuf = NULL, *rbuf = NULL;

  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  MPI_Type_size(dt, &dtsize);

  vrank = mod(rank - root, size); // mod computes math modulo rather than reminder
  halving_direction = 1; // Down -- send bottom half
  if(rank % 2){
    halving_direction = -1; // Up -- send top half
  }
  // The gather started with these directions. Thus this will
  // be the direction they ended up with if we have an odd number
  // of steps. If not, invert.
  if((int) log_2(size) % 2 == 0){
    halving_direction *= -1;
  }

  // I need to do the opposite of what I did in the gather.
  // Thus, I need to know where min_resident_block and max_resident_block
  // ended up after the last step.
  // Even ranks added 2^0, 2^2, 2^4, ... to max_resident_block
  //   and subtracted 2^1, 2^3, 2^5, ... from min_resident_block
  // Odd ranks subtracted 2^0, 2^2, 2^4, ... from min_resident_block
  //      and added 2^1, 2^3, 2^5, ... to max_resident_block
  if(rank % 2 == 0){    
    max_resident_block = mod((rank + 0x55555555) & ((0x1 << (int) log_2(size)) - 1), size);
    min_resident_block = mod((rank - 0xAAAAAAAA) & ((0x1 << (int) log_2(size)) - 1), size);
  }else{
    min_resident_block = mod((rank - 0x55555555) & ((0x1 << (int) log_2(size)) - 1), size);
    max_resident_block = mod((rank + 0xAAAAAAAA) & ((0x1 << (int) log_2(size)) - 1), size);    
  }

  mask = 0x1 << (int) (log_2(size) - 1);
  sbuf_offset = rank;
  if (root == rank){
    recvd = 1;
    sbuf = (char*) sendbuf;
  }

  vrank_nb = binary_to_negabinary(vrank);
  while(mask > 0){
    size_t top_start, top_end, bottom_start, bottom_end;
    size_t send_start, send_end, recv_start, recv_end;

    int partner = vrank_nb ^ ((mask << 1) - 1);
    partner = mod(negabinary_to_binary(partner) + root, size);
    int mask_lsbs = (mask << 1) - 1; // Mask with num_steps - step + 1 LSBs set to 1
    int lsbs = vrank_nb & mask_lsbs; // Extract k LSBs
    int equal_lsbs = (lsbs == 0 || lsbs == mask_lsbs);

    top_start = min_resident_block;
    top_end = mod(min_resident_block + mask - 1, size);
    bottom_start = mod(top_end + 1, size);
    bottom_end = max_resident_block;
    if(halving_direction == 1){
      // Send bottom half [..., size - 1]
      send_start = bottom_start;
      send_end = bottom_end;
      recv_start = top_start;
      recv_end = top_end;
      max_resident_block = mod(max_resident_block - mask, size);
    }else{
      // Send top half [0, ...]
      send_start = top_start;
      send_end = top_end;
      recv_start = bottom_start;
      recv_end = bottom_end;
      min_resident_block = mod(min_resident_block + mask, size);
    }

    if(recvd){
      if(send_end >= send_start){
        err = MPI_Send((char*) sbuf + send_start * sendcount * dtsize, sendcount * (send_end - send_start + 1)  , dt, partner, 0, comm);
        if(err != MPI_SUCCESS){ goto err_hndl; }
      }else{
        err = MPI_Send((char*) sbuf + send_start * sendcount * dtsize, sendcount * ((size - 1) - send_start + 1), dt, partner, 0, comm);
        if(err != MPI_SUCCESS){ goto err_hndl; }
        err = MPI_Send((char*) sbuf                                  , sendcount * (send_end + 1)               , dt, partner, 0, comm);
        if(err != MPI_SUCCESS){ goto err_hndl; }
      }
    }else if(equal_lsbs){
      // Setup the buffers to be used from now on
      // How large should the tmpbuf be?
      // It must be large enough to hold a number of blocks 
      // equal to the number of children in the tree rooted in me.
      size_t num_blocks = mod((recv_end - recv_start + 1), size);
      if(recv_start == recv_end){
        // I am a leaf and this is the last step, I do not need a tmpbuf
        rbuf = (char*) recvbuf;
        is_leaf = 1;
      }else{
        tmpbuf = (char*) malloc(recvcount * num_blocks * dtsize);
        if(tmpbuf == NULL){ err = MPI_ERR_NO_MEM; goto err_hndl; }
        sbuf = (char*) tmpbuf;
        rbuf = (char*) tmpbuf;

        // Adjust min and max resident blocks
        min_resident_block = 0;
        max_resident_block = num_blocks - 1;

        sbuf_offset = mod(rank - recv_start, size);
      }
      if(recv_end >= recv_start){ 
        err = MPI_Recv((char*) rbuf, recvcount * num_blocks, dt, partner, 0, comm, MPI_STATUS_IGNORE);
        if(err != MPI_SUCCESS){ goto err_hndl; }
      } else {
        err = MPI_Recv((char*) rbuf, recvcount * ((size - 1) - recv_start + 1), dt, partner, 0, comm, MPI_STATUS_IGNORE);
        if(err != MPI_SUCCESS){ goto err_hndl; }
        err = MPI_Recv((char*) rbuf + (recvcount * ((size - 1) - recv_start + 1)) * dtsize, recvcount * (recv_end + 1), dt, partner, 0, comm, MPI_STATUS_IGNORE);
        if(err != MPI_SUCCESS){ goto err_hndl; }
      }
      recvd = 1;
    }
    mask >>= 1;
    halving_direction *= -1;
  }

  if(!is_leaf){
    memcpy((char*) recvbuf, (char*) sbuf + sbuf_offset * recvcount * dtsize, recvcount * dtsize);
  }

err_hndl:
  if(tmpbuf != NULL){
    free(tmpbuf);
  }
  return err;
}
