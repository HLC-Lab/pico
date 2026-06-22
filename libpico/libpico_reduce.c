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

#define BINE_MIN(a, b) ((a) < (b) ? (a) : (b))

int reduce_bine_lat(const void *sendbuf, void *recvbuf, size_t count,
                     MPI_Datatype dt, MPI_Op op, int root, MPI_Comm comm)
{
  int size, rank, dtsize, vrank, mask, err = MPI_SUCCESS;
  int partner, mask_lsbs, lsbs, equal_lsbs;
  char* tmpbuf = NULL;
  size_t buf_size;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  MPI_Type_size(dt, &dtsize);

  if(count == 0) { return MPI_SUCCESS; }

  if(!is_power_of_two(size)) { err = MPI_ERR_SIZE; goto err_hndl; }

  buf_size = count * dtsize;

  tmpbuf = (char *) malloc(buf_size);
  if (tmpbuf == NULL) { err = MPI_ERR_NO_MEM; goto err_hndl; }

  if(rank != root){
    recvbuf = (char *) malloc(buf_size);
    if (recvbuf == NULL) { err = MPI_ERR_NO_MEM; goto err_hndl; }
  }

  err = copy_buffer((char *)sendbuf, (char *)recvbuf, count, dt);
  if(MPI_SUCCESS != err) { goto err_hndl; }

  vrank = mod(rank - root, size); // mod computes math modulo rather than reminder
  mask = 0x1;
  int btnb_vrank = binary_to_negabinary(vrank);
  while(mask < size){
    partner = btnb_vrank ^ ((mask << 1) - 1);
    partner = mod(negabinary_to_binary(partner) + root, size);
    mask_lsbs = (mask << 2) - 1; // Mask with step + 2 LSBs set to 1
    lsbs = btnb_vrank & mask_lsbs; // Extract k LSBs
    equal_lsbs = (lsbs == 0 || lsbs == mask_lsbs);

    if(!equal_lsbs || ((mask << 1) >= size && (rank != root))){
      err = MPI_Send(recvbuf, count, dt, partner, 0, comm);
      if (err != MPI_SUCCESS) { goto err_hndl; }
      break;
    }else{
      err = MPI_Recv(tmpbuf, count, dt, partner, 0, comm, MPI_STATUS_IGNORE);
      if (err != MPI_SUCCESS) { goto err_hndl; }
      err = MPI_Reduce_local(tmpbuf, recvbuf, count, dt, op);
      if (err != MPI_SUCCESS) { goto err_hndl; }
    }
    mask <<= 1;
  }

  free(tmpbuf);
  if(rank != root){
    free(recvbuf);
  }

  return MPI_SUCCESS;

err_hndl:
  if(tmpbuf != NULL) free(tmpbuf);
  if(rank != root){
    if (recvbuf != NULL) free(recvbuf);
  }
  return err;
}


int reduce_bine_bdw(const void *sendbuf, void *recvbuf, size_t count,
                     MPI_Datatype dt, MPI_Op op, int root, MPI_Comm comm)
{
  assert(root == 0); // TODO: Generalize
  int size, rank, dtsize, err = MPI_SUCCESS, steps, step;
  int count_per_rank, rem, mask = 0x1, inverse_mask;
  int block_first_mask, remapped_rank, receiving_mask;
  int *rindex = NULL, *sindex = NULL, *rcount = NULL, *scount = NULL;
  char* resbuf = NULL, *tmpbuf = NULL;
  size_t buf_size;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  MPI_Type_size(dt, &dtsize);

  steps = log_2(size);
  if(!is_power_of_two(size)) { err = MPI_ERR_SIZE; goto err_hndl; }

  count_per_rank = count / size;
  rem = count % size;

  buf_size = count * dtsize;
  tmpbuf = (char *) malloc(buf_size);
  if (tmpbuf == NULL) { err = MPI_ERR_NO_MEM; goto err_hndl; }

  if (rank == root) {
    resbuf = recvbuf;
  } else {
    resbuf = (char *) malloc(buf_size);
    if (resbuf == NULL) { err = MPI_ERR_NO_MEM; goto err_hndl; }
  }

  err = copy_buffer((char *)sendbuf, resbuf, count, dt);
  if(MPI_SUCCESS != err) { goto err_hndl; }

  mask = 0x1;
  inverse_mask = 0x1 << (int) (log_2(size) - 1);
  block_first_mask = ~(inverse_mask - 1);
  remapped_rank = remap_rank(size, rank);

  /***** Reduce_scatter *****/
  rindex = malloc(sizeof(int) * steps);
  sindex = malloc(sizeof(int) * steps);
  rcount = malloc(sizeof(int) * steps);
  scount = malloc(sizeof(int) * steps);
  step = 0;
  while(mask < size){
    int partner;
    int nbtb = negabinary_to_binary((mask << 1) - 1);
    if(rank % 2 == 0){
      partner = mod(rank + nbtb, size); 
    }else{
      partner = mod(rank - nbtb, size); 
    }

    // Compute send block boundaries inline
    int send_block_first = remap_rank(size, partner) & block_first_mask;
    int send_block_last = send_block_first + inverse_mask - 1;
    sindex[step] = count_per_rank * send_block_first + (send_block_first < rem ? send_block_first : rem);
    scount[step] = count_per_rank * (send_block_last - send_block_first + 1)
                   + (BINE_MIN(send_block_last, rem) - BINE_MIN(send_block_first, rem))
                   + (send_block_last < rem ? 1 : 0);

    // Compute recv block boundaries inline
    int recv_block_first = remapped_rank & block_first_mask;
    int recv_block_last = recv_block_first + inverse_mask - 1;
    rindex[step] = count_per_rank * recv_block_first + (recv_block_first < rem ? recv_block_first : rem);
    rcount[step] = count_per_rank * (recv_block_last - recv_block_first + 1)
                  + (BINE_MIN(recv_block_last, rem) - BINE_MIN(recv_block_first, rem))
                  + (recv_block_last < rem ? 1 : 0);

    err = MPI_Sendrecv(resbuf + sindex[step] * dtsize, scount[step], dt, partner, step,
                       tmpbuf + rindex[step] * dtsize, rcount[step], dt, partner, step,
                       comm, MPI_STATUS_IGNORE);
    if (err != MPI_SUCCESS) { goto err_hndl; }
    err = MPI_Reduce_local(tmpbuf + rindex[step] * dtsize, resbuf + rindex[step] * dtsize, rcount[step], dt, op);
    if (err != MPI_SUCCESS) { goto err_hndl; }

    mask <<= 1;
    inverse_mask >>= 1;
    block_first_mask >>= 1;
    step++;
  }

  /***** Gather *****/
  mask >>= 1;
  inverse_mask = 0x1;
  block_first_mask = ~0x0;
  // I send in the step corresponding to the position (starting from right)
  // of the first 1 in my remapped rank -- this indicates the step when the data reaches me in a scatter
  receiving_mask = 0x1 << (ffs(remapped_rank) - 1); // ffs starts counting from 1, thus -1
  step = steps - 1;
  while(mask > 0){
    int partner;
    int nbtb = negabinary_to_binary((mask << 1) - 1);
    if(rank % 2 == 0){
      partner = mod(rank + nbtb, size); 
    }else{
      partner = mod(rank - nbtb, size); 
    }

    // Only the one with 0 in the i-th bit starting from the left (i is the step) survives
    if(inverse_mask & receiving_mask){
      err = MPI_Send(resbuf + rindex[step] * dtsize, rcount[step], dt, partner, steps + step, comm);
      if (err != MPI_SUCCESS) { goto err_hndl; }
      break;
    }else{
      // Something similar for the block to recv.
      // I receive my partner's block, but aligned to the power of two
      err = MPI_Recv(resbuf + sindex[step] * dtsize, scount[step], dt, partner, steps + step, comm, MPI_STATUS_IGNORE);
      if(err != MPI_SUCCESS) { goto err_hndl; }
    }

    mask >>= 1;
    inverse_mask <<= 1;
    block_first_mask <<= 1;
    step--;
  }

  free(rindex);
  free(sindex);
  free(rcount);
  free(scount);
  free(tmpbuf);
  if(rank != root){
    free(resbuf);
  }

  return MPI_SUCCESS;

err_hndl:
  if(rindex != NULL) free(rindex);
  if(sindex != NULL) free(sindex);
  if(rcount != NULL) free(rcount);
  if(scount != NULL) free(scount);
  if(tmpbuf != NULL) free(tmpbuf);
  if (rank != root){
    if (resbuf != NULL) free(resbuf);
  }
  return err;
}

