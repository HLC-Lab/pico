#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>

#include "libswing.h"
#include "libswing_utils.h"

int reduce_swing_lat(const void *sendbuf, void *recvbuf, size_t count, MPI_Datatype dt, MPI_Op op, int root, MPI_Comm comm){
  int size, rank, dtsize, vrank, mask, err = MPI_SUCCESS;
  int partner, mask_lsbs, lsbs, equal_lsbs;
  char* tmpbuf = NULL;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  MPI_Type_size(dt, &dtsize);

  if(!is_power_of_two(size)) { err = MPI_ERR_SIZE; goto err_hndl; }

  if(count == 0) { return MPI_SUCCESS; }

  tmpbuf = (char *) malloc(count * (size_t) dtsize);
  if (tmpbuf == NULL) { err = MPI_ERR_NO_MEM; goto err_hndl; }
  if(rank != root){
    recvbuf = (char *) malloc(count * (size_t) dtsize);
    if (recvbuf == NULL) { err = MPI_ERR_NO_MEM; goto err_hndl; }
  }

  if(sendbuf != MPI_IN_PLACE) {
    err = copy_buffer((char *)sendbuf, (char *)recvbuf, count, dt);
    if(MPI_SUCCESS != err) { goto err_hndl; }
  }

  vrank = mod(rank - root, size); // mod computes math modulo rather than reminder
  mask = 0x1;
  while(mask < size){
    partner = binary_to_negabinary(vrank) ^ ((mask << 1) - 1);
    partner = mod(negabinary_to_binary(partner) + root, size);
    mask_lsbs = (mask << 2) - 1; // Mask with step + 2 LSBs set to 1
    lsbs = binary_to_negabinary(vrank) & mask_lsbs; // Extract k LSBs
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
  if(rank != root){
    if (recvbuf != NULL) free(recvbuf);
  }
  return err;
}


int reduce_swing_bdw(const void *sendbuf, void *recvbuf, size_t count, MPI_Datatype dt, MPI_Op op, int root, MPI_Comm comm){
  assert(root == 0); // TODO: Generalize
  int size, rank, dtsize, err = MPI_SUCCESS;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  MPI_Type_size(dt, &dtsize);
  int* displs = (int*) malloc(size*sizeof(int));
  int* recvcounts = (int*) malloc(size*sizeof(int));
  int count_per_rank = count / size;
  int rem = count % size;
  for(int i = 0; i < size; i++){
    displs[i] = count_per_rank*i + (i < rem ? i : rem);
    recvcounts[i] = count_per_rank + (i < rem ? 1 : 0);
  }
  
  void* tmpbuf = malloc(count*dtsize);
  void* resbuf;
  
  if(rank == root){
    resbuf = recvbuf;
  }else{
    resbuf = malloc(count*dtsize);
  }
  memcpy(resbuf, sendbuf, count*dtsize);

  int mask = 0x1;
  int inverse_mask = 0x1 << (int) (log_2(size) - 1);
  int block_first_mask = ~(inverse_mask - 1);
  int vrank = (rank % 2) ? rank : -rank;
  int remapped_rank = remap_rank(size, rank);
  
  /***** Reduce_scatter *****/
  while(mask < size){
    int partner;
    if(rank % 2 == 0){
      partner = mod(rank + negabinary_to_binary((mask << 1) - 1), size); 
    }else{
      partner = mod(rank - negabinary_to_binary((mask << 1) - 1), size); 
    }

    // For sure I need to send my (remapped) partner's data
    // the actual start block however must be aligned to 
    // the power of two
    int send_block_first = remap_rank(size, partner) & block_first_mask;
    int send_block_last = send_block_first + inverse_mask - 1;
    int send_count = displs[send_block_last] - displs[send_block_first] + recvcounts[send_block_last];
    // Something similar for the block to recv.
    // I receive my block, but aligned to the power of two
    int recv_block_first = remapped_rank & block_first_mask;
    int recv_block_last = recv_block_first + inverse_mask - 1;
    int recv_count = displs[recv_block_last] - displs[recv_block_first] + recvcounts[recv_block_last];

    err = MPI_Sendrecv((char*) resbuf + displs[send_block_first]*dtsize, send_count, dt, partner, 0,
                 (char*) tmpbuf + displs[recv_block_first]*dtsize, recv_count, dt, partner, 0, comm, MPI_STATUS_IGNORE);
    if (err != MPI_SUCCESS) { goto err_hndl; }
    err = MPI_Reduce_local((char*) tmpbuf + displs[recv_block_first]*dtsize, (char*) resbuf + displs[recv_block_first]*dtsize, recv_count, dt, op);
    if (err != MPI_SUCCESS) { goto err_hndl; }

    mask <<= 1;
    inverse_mask >>= 1;
    block_first_mask >>= 1;
  }

  /***** Gather *****/
  mask >>= 1;
  inverse_mask = 0x1;
  block_first_mask = ~0x0;
  int receiving_mask;
  // I send in the step corresponding to the position (starting from right)
  // of the first 1 in my remapped rank -- this indicates the step when the data reaches me in a scatter
  receiving_mask = 0x1 << (ffs(remapped_rank) - 1); // ffs starts counting from 1, thus -1
  
  while(mask > 0){
    int partner;
    if(rank % 2 == 0){
      partner = mod(rank + negabinary_to_binary((mask << 1) - 1), size); 
    }else{
      partner = mod(rank - negabinary_to_binary((mask << 1) - 1), size); 
    }

    // Only the one with 0 in the i-th bit starting from the left (i is the step) survives
    if(inverse_mask & receiving_mask){
      int send_block_first = remapped_rank & block_first_mask;
      int send_block_last = send_block_first + inverse_mask - 1;
      int send_count = displs[send_block_last] - displs[send_block_first] + recvcounts[send_block_last];    
      err = MPI_Send((char*) resbuf + displs[send_block_first]*dtsize, send_count, dt, partner, 0, comm);
      if (err != MPI_SUCCESS) { goto err_hndl; }
      break;
    }else{
      // Something similar for the block to recv.
      // I receive my partner's block, but aligned to the power of two
      int recv_block_first = remap_rank(size, partner) & block_first_mask;
      int recv_block_last = recv_block_first + inverse_mask - 1;
      int recv_count = displs[recv_block_last] - displs[recv_block_first] + recvcounts[recv_block_last];
      err = MPI_Recv((char*) resbuf + displs[recv_block_first]*dtsize, recv_count, dt, partner, 0, comm, MPI_STATUS_IGNORE);
      if(err != MPI_SUCCESS) { goto err_hndl; }
    }

    mask >>= 1;
    inverse_mask <<= 1;
    block_first_mask <<= 1;
  }

  free(tmpbuf);
  if(rank != root){
    free(resbuf);
  }
  free(displs);
  return MPI_SUCCESS;

err_hndl:
  if(tmpbuf != NULL) free(tmpbuf);
  if(displs != NULL) free(displs);
  if (rank != root){
    if (resbuf != NULL) free(resbuf);
  }
  return err;
}
