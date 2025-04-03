#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>

#include "libswing.h"
#include "libswing_utils.h"

int alltoall_swing(const void *sendbuf, size_t s_count, MPI_Datatype s_dtype,
                   void *recvbuf, size_t r_count, MPI_Datatype r_dtype, MPI_Comm comm)
{
  assert(s_count == r_count);
  assert(s_dtype == r_dtype);
  int rank, size, dtsize, err = MPI_SUCCESS;
  int inverse_mask, mask = 0x1, block_first_mask;
  size_t num_resident_blocks, num_resident_blocks_next, min_block_s, max_block_s;
  size_t sbuf_size, tmpbuf_size, tmpbuf_size_real;
  char *tmpbuf = NULL;
  uint *resident_block, *resident_block_next;
  // resident_block[i] contains the id of a block that is resident in the current rank (for i < num_resident_blocks)
  // resident_block_next[i] contains the id of a block that is resident in the current rank in the next step (for i < num_resident_blocks_next)
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  MPI_Type_size(s_dtype, &dtsize);

  num_resident_blocks = size;
  num_resident_blocks_next = 0;
  sbuf_size = s_count * dtsize;
  tmpbuf_size = sbuf_size * size;
  tmpbuf_size_real = tmpbuf_size + sizeof(uint) * size + sizeof(uint) * size;

  tmpbuf = (char *) malloc(tmpbuf_size_real);
  if(tmpbuf == NULL){
    err = MPI_ERR_NO_MEM;
    goto err_hndl;
  }
  resident_block = (uint *) (tmpbuf + tmpbuf_size);
  resident_block_next = (uint *) (tmpbuf + tmpbuf_size + sizeof(uint) * size);

  // At the beginning I only have my blocks
  for(size_t i = 0; i < size; i++){
    resident_block[i] = i;
  }

  memcpy(tmpbuf, sendbuf, tmpbuf_size);

  // We use recvbuf to receive/send the data, and tmpbuf to organize the data to send at the next step
  // By doing so, we avoid a copy form tmpbuf to recvbuf at the end
  inverse_mask = 0x1 << (int) (log_2(size) - 1);
  block_first_mask = ~(inverse_mask - 1);

  while(mask < size){
    int partner;
    int ntbn = negabinary_to_binary((mask << 1) -1);
    if(rank % 2 == 0){
      partner = mod(rank + ntbn, size); 
    } else {
      partner = mod(rank - ntbn, size); 
    }
    min_block_s = remap_rank(size, partner) & block_first_mask;
    max_block_s = min_block_s + inverse_mask - 1;

    size_t block_recvd_cnt = 0, block_send_cnt = 0;
    size_t offset_send = 0, offset_keep = 0;
    num_resident_blocks_next = 0;
    for(size_t i = 0; i < size; i++){
      uint block = resident_block[i % num_resident_blocks];
      // Shall I send this block? Check the negabinary thing  
      uint remap_block = remap_rank(size, block);
      size_t offset = i * sbuf_size;

      // I move to the beginning of tmpbuf the blocks I want to keep,
      // and I move to recvbuf the blocks I want to send.
      if(remap_block >= min_block_s && remap_block <= max_block_s){
        memcpy((char*) recvbuf + offset_send, tmpbuf + offset, sbuf_size);
        offset_send += sbuf_size;
        block_send_cnt++;
      }else{
        // Copy the blocks we are not sending to the second half of recvbuf
        if(offset != offset_keep){
          memcpy(tmpbuf + offset_keep, tmpbuf + offset, sbuf_size);
        }
        offset_keep += sbuf_size;
        block_recvd_cnt++;

        resident_block_next[num_resident_blocks_next] = block;
        num_resident_blocks_next++;
      }
    }
    assert(block_recvd_cnt == size/2);
    assert(block_send_cnt == size/2);
    num_resident_blocks /= 2;

    // I receive data in the second half of tmpbuf (the first half contains the blocks I am keeping from previous iteration)
    err = MPI_Sendrecv((char*) recvbuf, s_count * block_send_cnt, s_dtype, partner, 0,
                       tmpbuf + (size / 2) * sbuf_size, s_count * block_send_cnt, s_dtype, partner, 0, 
                       comm, MPI_STATUS_IGNORE);
    if(err != MPI_SUCCESS) { goto err_hndl; }

    // Update resident blocks
    memcpy(resident_block, resident_block_next, sizeof(uint) * num_resident_blocks);

    mask <<= 1;
    inverse_mask >>= 1;
    block_first_mask >>= 1;
  }

  // Now I need to permute tmpbuf into recvbuf
  // Since I always received the new block on the right, and moved the blocks
  // I wanted to keep to the left, they are now sorted in the same order they reached this
  // rank from their corresponding source ranks. 
  // I.e., I should consider the "reverse tree" (with this rank at the bottom and all the other ranks on top),
  // which represent how the data arrived here.
  // This tree is basically the opposite that I used to send the data
  // I should consider the decreasing tree, and viceversa.
  for(size_t i = 0; i < size; i++){
    int rotated_i = 0;
    if((rank % 2) == 0){
      rotated_i = mod(i - rank, size);
    } else {
      rotated_i = mod(rank - i, size);
    }
    int repr = 0;
    if(in_range(rotated_i, log_2(size))){
      repr = binary_to_negabinary(rotated_i);
    }else{
      repr = binary_to_negabinary(rotated_i - size);
    }
    int index = remap_distance_doubling(repr);

    size_t offset_src = index * sbuf_size;
    size_t offset_dst = i * sbuf_size;
    memcpy((char*) recvbuf + offset_dst, tmpbuf + offset_src, sbuf_size);
  }

  free(tmpbuf);
  return MPI_SUCCESS;

err_hndl:
  if(tmpbuf != NULL) free(tmpbuf);
  return err;
}
