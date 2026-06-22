/*
 * Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
 * Licensed under the MIT License
 */

#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <assert.h>

#include "libpico.h"
#include "libpico_utils.h"



/**
 * Open MPI linear gather function copied from 
 * https://github.com/open-mpi/ompi/blob/3a2e90895e15822912002be1e9aea8032c4c0bae/ompi/mca/coll/base/coll_base_gather.c#L371
 */
int gather_linear(const void *sbuf, size_t scount, MPI_Datatype sdtype,
          void *rbuf, size_t rcount, MPI_Datatype rdtype,
          int root, MPI_Comm comm)
{
  int i, rank, size, err;
  MPI_Aint lb, s_ext, r_ext;
  MPI_Aint s_block_span, r_block_span;  /* byte span of one rank's block */
  char *tmp_buf;

  /* Init */
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  /* Non-root: send and return */
  if (rank != root) {
    return MPI_Send(sbuf, (int)scount, sdtype, root, 0, comm);
  }

  /* Root: compute spans/stride */
  MPI_Type_get_extent(sdtype, &lb, &s_ext);
  MPI_Type_get_extent(rdtype, &lb, &r_ext);
  assert(lb == 0); // should be true for basic types

  s_block_span = (MPI_Aint)scount * s_ext;
  r_block_span = (MPI_Aint)rcount * r_ext;

  for (i = 0; i < size; ++i) {
    tmp_buf = (char *)rbuf + (MPI_Aint)i * r_block_span;

    if (i == rank) {
      if (sbuf == MPI_IN_PLACE) { continue; }

      MPI_Aint n = (s_block_span < r_block_span) ? s_block_span : r_block_span;
      memcpy(tmp_buf, sbuf, (size_t)n);
    } else {
      err = MPI_Recv(tmp_buf, (int)rcount, rdtype, i, 0, comm, MPI_STATUS_IGNORE);
      if (MPI_SUCCESS != err) return err;
    }
  }

  return MPI_SUCCESS;
}

int gather_bine(const void *sendbuf, size_t sendcount, MPI_Datatype dt, void *recvbuf, size_t recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm){
  assert(sendcount == recvcount && dt == recvtype);
  int size, rank, dtsize, err = MPI_SUCCESS;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  MPI_Type_size(dt, &dtsize);

  if(rank != root){
    recvbuf = (char *) malloc(recvcount * size * dtsize);
    if (recvbuf == NULL) { err = MPI_ERR_NO_MEM; goto err_hndl; }
  }

  memcpy((char*) recvbuf + rank * recvcount * dtsize, sendbuf, recvcount * dtsize);

  // I have the blocks in range [min_block_resident, max_block_resident]
  size_t min_block_resident = rank, max_block_resident = rank;
  int vrank = mod(rank - root, size); // mod computes math modulo rather than reminder
  int extension_direction = 1; // Down
  if(rank % 2){
    extension_direction = -1; // Up
  }
  int mask = 0x1;
  while(mask < size){
    int partner = binary_to_negabinary(vrank) ^ ((mask << 1) - 1);
    partner = mod(negabinary_to_binary(partner) + root, size);
    int mask_lsbs = (mask << 2) - 1; // Mask with step + 2 LSBs set to 1
    int lsbs = binary_to_negabinary(vrank) & mask_lsbs; // Extract k LSBs
    int equal_lsbs = (lsbs == 0 || lsbs == mask_lsbs);

    if(!equal_lsbs || ((mask << 1) >= size && (rank != root))){
      if(max_block_resident >= min_block_resident){
        // Single send
        err = MPI_Send((char*) recvbuf + min_block_resident*recvcount*dtsize, recvcount*(max_block_resident - min_block_resident + 1), dt, partner, 0, comm);
        if (err != MPI_SUCCESS) { goto err_hndl; }
      }else{
        // Wrapped send
        err = MPI_Send((char*) recvbuf + min_block_resident*recvcount*dtsize, recvcount*((size - 1) - min_block_resident + 1), dt, partner, 0, comm);
        if (err != MPI_SUCCESS) { goto err_hndl; }
        err = MPI_Send((char*) recvbuf                                      , recvcount*(max_block_resident + 1)             , dt, partner, 0, comm);
        if (err != MPI_SUCCESS) { goto err_hndl; }
      }
      break;
    }else{
      // Determine if I extend the data I have up or down
      size_t recv_start, recv_end; // Receive [recv_start, recv_end]
      if(extension_direction == 1){
        recv_start = mod(max_block_resident + 1, size);
        recv_end = mod(max_block_resident + mask, size);
        max_block_resident = recv_end;
      }else{
        recv_end = mod(min_block_resident - 1, size);
        recv_start = mod(min_block_resident - mask, size);
        min_block_resident = recv_start;
      }
      if(recv_end >= recv_start){ 
        // Single recv
        err = MPI_Recv((char*) recvbuf + recv_start*recvcount*dtsize, recvcount*(recv_end - recv_start + 1), dt, partner, 0, comm, MPI_STATUS_IGNORE);
        if (err != MPI_SUCCESS) { goto err_hndl; }
      }else{
        // Wrapped recv
        err = MPI_Recv((char*) recvbuf + recv_start*recvcount*dtsize, recvcount*((size - 1) - recv_start + 1), dt, partner, 0, comm, MPI_STATUS_IGNORE);
        if (err != MPI_SUCCESS) { goto err_hndl; }
        err = MPI_Recv((char*) recvbuf                              , recvcount*(recv_end + 1)               , dt, partner, 0, comm, MPI_STATUS_IGNORE);
        if (err != MPI_SUCCESS) { goto err_hndl; }
      }

      extension_direction *= -1;
    }
    mask <<= 1;
  }
  if(rank != root){
    free(recvbuf);
  }
  return MPI_SUCCESS;

err_hndl:
  if(rank != root){
    if (recvbuf != NULL) free(recvbuf);
  }
  return err;
}

