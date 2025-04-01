#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>

#include "libswing.h"
#include "libswing_utils.h"

int scatter_swing(const void *sendbuf, size_t sendcount, MPI_Datatype dt, void *recvbuf, size_t recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm){
  assert(sendcount == recvcount); // TODO: Implement the case where sendcount != recvcount
  assert(dt == recvtype); // TODO: Implement the case where sendtype != recvtype

  int size, rank, dtsize, err = MPI_SUCCESS;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  MPI_Type_size(dt, &dtsize);
  // Tempbuffer
  char *tmpbuf, *sbuf, *rbuf;
  if(rank != root){
    tmpbuf = (char*) malloc(sendcount*size*dtsize);
    sbuf = (char*) tmpbuf;
    rbuf = (char*) tmpbuf;
  }else{
    sbuf = (char*) sendbuf;
  }

  int vrank = mod(rank - root, size); // mod computes math modulo rather than reminder
  int halving_direction = 1; // Down -- send bottom half
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
  size_t min_resident_block, max_resident_block;
  if(rank % 2 == 0){    
    max_resident_block = mod(rank + 0x55555555 & ((0x1 << (int) log_2(size)) - 1), size);
    min_resident_block = mod(rank - 0xAAAAAAAA & ((0x1 << (int) log_2(size)) - 1), size);
  }else{
    min_resident_block = mod(rank - 0x55555555 & ((0x1 << (int) log_2(size)) - 1), size);
    max_resident_block = mod(rank + 0xAAAAAAAA & ((0x1 << (int) log_2(size)) - 1), size);    
  }
  
  int mask = 0x1 << (int) (log_2(size) - 1);
  int recvd = (root == rank);
  while(mask > 0){
    int partner = binary_to_negabinary(vrank) ^ ((mask << 1) - 1);
    partner = mod(negabinary_to_binary(partner) + root, size);    
    int mask_lsbs = (mask << 1) - 1; // Mask with num_steps - step + 1 LSBs set to 1
    int lsbs = binary_to_negabinary(vrank) & mask_lsbs; // Extract k LSBs
    int equal_lsbs = (lsbs == 0 || lsbs == mask_lsbs);

    size_t top_start, top_end, bottom_start, bottom_end;
    top_start = min_resident_block;
    top_end = mod(min_resident_block + mask - 1, size);
    bottom_start = mod(top_end + 1, size);
    bottom_end = max_resident_block;
    size_t send_start, send_end, recv_start, recv_end;
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
      // Single send
      err = MPI_Send((char*) sbuf + send_start*sendcount*dtsize, sendcount*(send_end - send_start + 1), dt, partner, 0, comm);
      if(err != MPI_SUCCESS){ goto err_hndl; }
      }else{
      // Wrapped send
      err = MPI_Send((char*) sbuf + send_start*sendcount*dtsize, sendcount*((size - 1) - send_start + 1), dt, partner, 0, comm);
      if(err != MPI_SUCCESS){ goto err_hndl; }
      err = MPI_Send((char*) sbuf                              , sendcount*(send_end + 1)               , dt, partner, 0, comm);
      if(err != MPI_SUCCESS){ goto err_hndl; }
      }
    }else if(equal_lsbs){
    if(recv_end >= recv_start){ 
      // Single recv
      err = MPI_Recv((char*) rbuf + recv_start*recvcount*dtsize, recvcount*(recv_end - recv_start + 1), dt, partner, 0, comm, MPI_STATUS_IGNORE);
      if(err != MPI_SUCCESS){ goto err_hndl; }
    }else{
      // Wrapped recv
      err = MPI_Recv((char*) rbuf + recv_start*recvcount*dtsize, recvcount*((size - 1) - recv_start + 1), dt, partner, 0, comm, MPI_STATUS_IGNORE);
      if(err != MPI_SUCCESS){ goto err_hndl; }
      err = MPI_Recv((char*) rbuf                              , recvcount*(recv_end + 1)               , dt, partner, 0, comm, MPI_STATUS_IGNORE);
      if(err != MPI_SUCCESS){ goto err_hndl; }
    }
    recvd = 1;
    }
    mask >>= 1;
    halving_direction *= -1;
  }

  memcpy(recvbuf, (char*) sbuf + rank*recvcount*dtsize, recvcount*dtsize);
  if(rank != root){
    free(tmpbuf);
  }
  return MPI_SUCCESS;

err_hndl:
  if(tmpbuf != NULL && rank != root) free(tmpbuf);
  return err;
}
