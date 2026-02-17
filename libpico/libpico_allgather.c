/*
 * Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
 * Licensed under the MIT License
 */

#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <unistd.h>

#include "libpico.h"
#include "libpico_utils.h"

#ifdef PICO_MPI_CUDA_AWARE
#include "support_kernel.h"
#endif

int allgather_recursivedoubling_hierarchy_local_parallel(const void *sbuf, size_t scount, MPI_Datatype sdtype,
                                             void *rbuf, size_t rcount, MPI_Datatype rdtype, MPI_Comm comm)
{
  int line, rank, size, err = MPI_SUCCESS;
  int peer, distance, node_offset, local_rank, node_size, node_rank, send_block_location, dist_mask;
  int node_sub_group, remapped_node_rank, remaining_node, node_data_to_send, node_data_to_recv, remapped_peer;
  int local_sub_group, local_data_to_send, local_data_to_recv, remapped_local_rank, remaning_local, req_index;
  ptrdiff_t rlb, rext;
  int task_on_node = pico_task_on_node();
  MPI_Request send_reqs[task_on_node], recv_reqs[task_on_node];
  char *tmprecv = NULL, *tmpsend = NULL, *tmprecv_buff = NULL;

  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  PICO_TAG_BEGIN("setup");

  err = MPI_Type_get_extent(rdtype, &rlb, &rext);
  if (MPI_SUCCESS != err)
  {
    line = __LINE__;
    goto err_hndl;
  }

#if defined PICO_MPI_CUDA_AWARE && !defined GPU_NATIVE_SUPPORT
  tmprecv_buff = (char *)calloc(size * rcount, rext);
  if (tmprecv_buff == NULL)
  {
    line = __LINE__;
    err = MPI_ERR_NO_MEM;
    goto err_hndl;
  }

  tmpsend = (char *)sbuf;
  tmprecv = tmprecv_buff + (ptrdiff_t)rank * (ptrdiff_t)rcount * rext;
  BINE_CUDA_CHECK(cudaMemcpy(tmprecv, tmpsend, rcount * rext, cudaMemcpyDeviceToHost));
#else
  tmprecv_buff = rbuf;

  // Setup buffer for mpi if not in place
  if (MPI_IN_PLACE != sbuf)
  {
    tmpsend = (char *)sbuf;
    tmprecv = tmprecv_buff + (ptrdiff_t)rank * (ptrdiff_t)rcount * rext;

    err = COPY_BUFF_DIFF_DT(tmpsend, scount, sdtype, tmprecv, rcount, rdtype);
    if (MPI_SUCCESS != err)
    {
      line = __LINE__;
      goto err_hndl;
    }
  }
#endif
  pico_get_group_config(&node_size, &node_rank, &node_offset, &local_rank, task_on_node, size, rank);
  PICO_TAG_END("setup");

  PICO_TAG_BEGIN("local_comm");
  // local allgather
  PICO_TAG_BEGIN("local_comm/setup");
  local_sub_group = floor_power_of_two(task_on_node);
  remaning_local = task_on_node - local_sub_group;
  send_block_location = rank;
  PICO_TAG_END("local_comm/setup");

  PICO_TAG_BEGIN("local_comm/recv_from_excluded_rank");
  // share data betwin excluded local rank
  if (local_sub_group != task_on_node)
  {
    if ((local_rank >> 1) < remaning_local && local_rank & 1)
    {
      tmpsend = (char *)tmprecv_buff + (ptrdiff_t)send_block_location * (ptrdiff_t)rcount * rext;
      err = MPI_Send(tmpsend, rcount, rdtype, rank - 1, 0, comm);
    }
    else if ((local_rank >> 1) < remaning_local && !(local_rank & 1))
    {
      tmprecv = (char *)tmprecv_buff + (ptrdiff_t)(send_block_location + 1) * (ptrdiff_t)rcount * rext;
      err = MPI_Recv(tmprecv, rcount, rdtype, rank + 1, 0, comm, MPI_STATUS_IGNORE);
    }

    if (MPI_SUCCESS != err)
    {
      line = __LINE__;
      goto err_hndl;
    }
  }
  PICO_TAG_END("local_comm/recv_from_excluded_rank");

  req_index = 0;
  if (!(local_rank & 1) || local_rank >= remaning_local << 1 || local_sub_group == task_on_node)
  {
    PICO_TAG_BEGIN("local_comm/exchange");
    remapped_local_rank = (local_rank >> 1) < remaning_local && local_sub_group != task_on_node ? local_rank >> 1 : local_rank - remaning_local;
    local_data_to_send = remapped_local_rank < remaning_local ? 2 : 1;
    tmpsend = (char *)tmprecv_buff + (ptrdiff_t)rank * (ptrdiff_t)rcount * rext;

    for (int i = 0; i < local_sub_group; i++)
    {
      if (i == remapped_local_rank)
        continue;

      remapped_peer = (i < remaning_local && local_sub_group != task_on_node ? (i * 2) : (i + remaning_local));
      local_data_to_recv = remapped_peer < remaning_local ? 2 : 1;
      peer = node_offset + remapped_peer;
      tmprecv = (char *)tmprecv_buff + (ptrdiff_t)(peer) * (ptrdiff_t)rcount * rext;

      /* Sendreceive */
      err = MPI_Isend(tmpsend, (ptrdiff_t)local_data_to_send * (ptrdiff_t)rcount, rdtype, peer, 0, comm, &send_reqs[req_index]);
      if (MPI_SUCCESS != err)
      {
        line = __LINE__;
        goto err_hndl;
      }

      err = MPI_Irecv(tmprecv, (ptrdiff_t)local_data_to_recv * (ptrdiff_t)rcount, rdtype, peer, 0, comm, &recv_reqs[req_index]);
      if (MPI_SUCCESS != err)
      {
        line = __LINE__;
        goto err_hndl;
      }

      req_index++;
    }
    PICO_TAG_END("local_comm/exchange");
  }

  PICO_TAG_BEGIN("local_comm/wait_recv");
  err = MPI_Waitall(req_index, recv_reqs, MPI_STATUSES_IGNORE);
  if (MPI_SUCCESS != err)
  {
    line = __LINE__;
    goto err_hndl;
  }
  PICO_TAG_END("local_comm/wait_recv");

  PICO_TAG_BEGIN("local_comm/wait_send");
  err = MPI_Waitall(req_index, send_reqs, MPI_STATUSES_IGNORE);
  if (MPI_SUCCESS != err)
  {
    line = __LINE__;
    goto err_hndl;
  }
  PICO_TAG_END("local_comm/wait_send");

  PICO_TAG_BEGIN("local_comm/send_to_excluded_rank");
  if (local_sub_group != task_on_node)
  {
    if ((local_rank >> 1) < remaning_local && !(local_rank & 1))
    {
      tmpsend = (char *)tmprecv_buff + (ptrdiff_t)node_offset * (ptrdiff_t)rcount * rext;
      err = MPI_Send(tmpsend, rcount * (local_rank + 1), rdtype, rank + 1, 0, comm);
      if (MPI_SUCCESS != err)
      {
        line = __LINE__;
        goto err_hndl;
      }
      tmpsend = (char *)tmprecv_buff + (ptrdiff_t)(rank + 1) * (ptrdiff_t)rcount * rext;
      err = MPI_Send(tmpsend, rcount * (task_on_node - (local_rank + 1)), rdtype, rank + 1, 0, comm);
      if (MPI_SUCCESS != err)
      {
        line = __LINE__;
        goto err_hndl;
      }
    }
    else if ((local_rank >> 1) < remaning_local && local_rank & 1)
    {
      tmprecv = (char *)tmprecv_buff + (ptrdiff_t)node_offset * (ptrdiff_t)rcount * rext;
      err = MPI_Recv(tmprecv, rcount * local_rank, rdtype, rank - 1, 0, comm, MPI_STATUS_IGNORE);
      if (MPI_SUCCESS != err)
      {
        line = __LINE__;
        goto err_hndl;
      }
      tmprecv = (char *)tmprecv_buff + (ptrdiff_t)rank * (ptrdiff_t)rcount * rext;
      err = MPI_Recv(tmprecv, rcount * (task_on_node - local_rank), rdtype, rank - 1, 0, comm, MPI_STATUS_IGNORE);
      if (MPI_SUCCESS != err)
      {
        line = __LINE__;
        goto err_hndl;
      }
    }
  }
  PICO_TAG_END("local_comm/send_to_excluded_rank");
  // end local comunication
  PICO_TAG_END("local_comm");

  PICO_TAG_BEGIN("global_comm");
  PICO_TAG_BEGIN("global_comm/setup");
  // global allgather
  send_block_location = node_offset;
  node_sub_group = floor_power_of_two(node_size);
  remaining_node = node_size - node_sub_group;
  dist_mask = ~0;
  PICO_TAG_END("global_comm/setup");

  // share data betwin extra node and node in the group
  PICO_TAG_BEGIN("global_comm/recv_from_excluded_node");
  if (node_sub_group != node_size)
  {
    if ((node_rank >> 1) < remaining_node && node_rank & 1)
    {
      tmpsend = (char *)tmprecv_buff + (ptrdiff_t)send_block_location * (ptrdiff_t)rcount * rext;
      err = MPI_Send(tmpsend, task_on_node * rcount, rdtype, ((node_rank - 1) * task_on_node) + local_rank, 0, comm);
    }
    else if ((node_rank >> 1) < remaining_node && !(node_rank & 1))
    {
      tmprecv = (char *)tmprecv_buff + (ptrdiff_t)(send_block_location + task_on_node) * (ptrdiff_t)rcount * rext;
      err = MPI_Recv(tmprecv, task_on_node * rcount, rdtype, ((node_rank - 1) * task_on_node) + local_rank, 0, comm, MPI_STATUS_IGNORE);
    }

    if (MPI_SUCCESS != err)
    {
      line = __LINE__;
      goto err_hndl;
    }
  }
  PICO_TAG_END("global_comm/recv_from_excluded_node");

  // exchange data in sub group
  PICO_TAG_BEGIN("global_comm/exchange");
  if (!(node_rank & 1) || node_rank >= (remaining_node << 1) || node_size == node_sub_group)
  {
    remapped_node_rank = (node_rank >> 1) < remaining_node && node_size != node_sub_group ? node_rank / 2 : node_rank - remaining_node;

    for (distance = 0x1; distance < node_sub_group; distance <<= 1)
    {
      remapped_peer = remapped_node_rank ^ distance;
      node_data_to_recv = (distance + min(distance, max(remaining_node - (remapped_peer & dist_mask), 0))) * task_on_node;
      node_data_to_send = (distance + min(distance, max(remaining_node - (remapped_node_rank & dist_mask), 0))) * task_on_node;
      dist_mask <<= 1;
      
      if (remapped_node_rank < remapped_peer)
      {
        tmpsend = (char *)tmprecv_buff + (ptrdiff_t)send_block_location * (ptrdiff_t)rcount * rext;
        tmprecv = (char *)tmprecv_buff + (ptrdiff_t)(send_block_location + node_data_to_send) * (ptrdiff_t)rcount * rext;
      }
      else
      {
        tmpsend = (char *)tmprecv_buff + (ptrdiff_t)send_block_location * (ptrdiff_t)rcount * rext;
        tmprecv = (char *)tmprecv_buff + (ptrdiff_t)(send_block_location - node_data_to_recv) * (ptrdiff_t)rcount * rext;
        send_block_location -= node_data_to_recv;
      }

      peer = remapped_peer < remaining_node && node_size != node_sub_group ? remapped_peer * 2 : remapped_peer + remaining_node;
      peer = peer * task_on_node + local_rank;

      /* Sendreceive */
      PICO_TAG_BEGIN("global_comm/exchange/send_recv");
      err = MPI_Sendrecv(tmpsend, node_data_to_send * rcount, rdtype, peer, 0,
                         tmprecv, node_data_to_recv * rcount, rdtype,
                         peer, 0, comm, MPI_STATUS_IGNORE);
      PICO_TAG_END("global_comm/exchange/send_recv");
      if (MPI_SUCCESS != err)
      {
        line = __LINE__;
        goto err_hndl;
      }
    }
  }
  PICO_TAG_END("global_comm/exchange");

  // share data back to extra node
  PICO_TAG_BEGIN("global_comm/send_to_excluded_node");
  if (node_sub_group != node_size)
  {
    if ((node_rank >> 1) < remaining_node && !(node_rank & 1))
    {
      err = MPI_Send(tmprecv_buff, (node_rank + 1) * task_on_node * rcount, rdtype, ((node_rank + 1) * task_on_node) + local_rank, 0, comm);
      if (MPI_SUCCESS != err)
      {
        line = __LINE__;
        goto err_hndl;
      }
      tmpsend = (char *)tmprecv_buff + (ptrdiff_t)(node_rank + 1) * task_on_node * (ptrdiff_t)rcount * rext;
      err = MPI_Send(tmpsend, (node_size - (node_rank + 1)) * task_on_node * rcount, rdtype, ((node_rank + 1) * task_on_node) + local_rank, 0, comm);
      if (MPI_SUCCESS != err)
      {
        line = __LINE__;
        goto err_hndl;
      }
    }
    else if ((node_rank >> 1) < remaining_node && node_rank & 1)
    {
      err = MPI_Recv(tmprecv_buff, node_rank * task_on_node * rcount, rdtype, ((node_rank - 1) * task_on_node) + local_rank, 0, comm, MPI_STATUS_IGNORE);
      if (MPI_SUCCESS != err)
      {
        line = __LINE__;
        goto err_hndl;
      }
      tmprecv = (char *)tmprecv_buff + (ptrdiff_t)node_rank * task_on_node * (ptrdiff_t)rcount * rext;
      err = MPI_Recv(tmprecv, (node_size - node_rank) * task_on_node * rcount, rdtype, ((node_rank - 1) * task_on_node) + local_rank, 0, comm, MPI_STATUS_IGNORE);
      if (MPI_SUCCESS != err)
      {
        line = __LINE__;
        goto err_hndl;
      }
    }
  }
  PICO_TAG_END("global_comm/send_to_excluded_node");
  PICO_TAG_END("global_comm");
  // end global

#if defined PICO_MPI_CUDA_AWARE && !defined GPU_NATIVE_SUPPORT
  BINE_CUDA_CHECK(cudaMemcpy(rbuf, tmprecv_buff, size * rcount * rext, cudaMemcpyHostToDevice));
  if (tmprecv_buff != NULL)
  {
    free(tmprecv_buff);
  }
#endif

  return MPI_SUCCESS;
err_hndl:
  BINE_DEBUG_PRINT("\n%s:%4d\tRank %d Error occurred %d\n\n", __FILE__, line, rank, err);
  (void)line; // silence compiler warning
  return err;
}


int allgather_recursivedoubling_hierarchy(const void *sbuf, size_t scount, MPI_Datatype sdtype,
                                             void *rbuf, size_t rcount, MPI_Datatype rdtype, MPI_Comm comm)
{
  int line, rank, size, err = MPI_SUCCESS;
  int peer, distance, node_offset, local_rank, node_size, node_rank, send_block_location, dist_mask;
  int node_sub_group, remapped_node_rank, remaining_node, node_data_to_send, node_data_to_recv, remapped_peer;
  int local_sub_group, local_data_to_send, local_data_to_recv, remapped_local_rank, remaning_local;
  ptrdiff_t rlb, rext;
  char *tmprecv = NULL, *tmpsend = NULL, *tmprecv_buff = NULL;
  int task_on_node = pico_task_on_node();

  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  PICO_TAG_BEGIN("setup");
  err = MPI_Type_get_extent(rdtype, &rlb, &rext);
  if (MPI_SUCCESS != err)
  {
    line = __LINE__;
    goto err_hndl;
  }

#if defined PICO_MPI_CUDA_AWARE && !defined GPU_NATIVE_SUPPORT
  tmprecv_buff = (char *)calloc(size * rcount, rext);
  if (tmprecv_buff == NULL)
  {
    line = __LINE__;
    err = MPI_ERR_NO_MEM;
    goto err_hndl;
  }

  tmpsend = (char *)sbuf;
  tmprecv = tmprecv_buff + (ptrdiff_t)rank * (ptrdiff_t)rcount * rext;
  BINE_CUDA_CHECK(cudaMemcpy(tmprecv, tmpsend, rcount * rext, cudaMemcpyDeviceToHost));
#else
  tmprecv_buff = rbuf;

  // Setup buffer for mpi if not in place
  if (MPI_IN_PLACE != sbuf)
  {
    tmpsend = (char *)sbuf;
    tmprecv = tmprecv_buff + (ptrdiff_t)rank * (ptrdiff_t)rcount * rext;

    err = COPY_BUFF_DIFF_DT(tmpsend, scount, sdtype, tmprecv, rcount, rdtype);
    if (MPI_SUCCESS != err)
    {
      line = __LINE__;
      goto err_hndl;
    }
  }
#endif
  pico_get_group_config(&node_size, &node_rank, &node_offset, &local_rank, task_on_node, size, rank);
  PICO_TAG_END("setup");

  PICO_TAG_BEGIN("local_comm");
  // local allgather
  PICO_TAG_BEGIN("local_comm/setup");
  local_sub_group = floor_power_of_two(task_on_node);
  remaning_local = task_on_node - local_sub_group;
  send_block_location = rank;
  PICO_TAG_END("local_comm/setup");

  PICO_TAG_BEGIN("local_comm/recv_from_excluded_rank");
  // share data betwin excluded local rank
  if (local_sub_group != task_on_node)
  {
    if ((local_rank >> 1) < remaning_local && local_rank & 1)
    {
      tmpsend = (char *)tmprecv_buff + (ptrdiff_t)send_block_location * (ptrdiff_t)rcount * rext;
      err = MPI_Send(tmpsend, rcount, rdtype, rank - 1, 0, comm);
    }
    else if ((local_rank >> 1) < remaning_local && !(local_rank & 1))
    {
      tmprecv = (char *)tmprecv_buff + (ptrdiff_t)(send_block_location + 1) * (ptrdiff_t)rcount * rext;
      err = MPI_Recv(tmprecv, rcount, rdtype, rank + 1, 0, comm, MPI_STATUS_IGNORE);
    }

    if (MPI_SUCCESS != err)
    {
      line = __LINE__;
      goto err_hndl;
    }
  }
  PICO_TAG_END("local_comm/recv_from_excluded_rank");

  PICO_TAG_BEGIN("local_comm/exchange");
  if (!(local_rank & 1) || local_rank >= remaning_local << 1 || local_sub_group == task_on_node)
  {
    remapped_local_rank = (local_rank >> 1) < remaning_local && local_sub_group != task_on_node ? local_rank >> 1 : local_rank - remaning_local;
    dist_mask = ~0;

    for (distance = 0x1; distance < local_sub_group; distance <<= 1)
    {
      remapped_peer = (remapped_local_rank ^ distance);
      local_data_to_recv = distance + min(distance, max(remaning_local - (remapped_peer & dist_mask), 0));
      local_data_to_send = distance + min(distance, max(remaning_local - (remapped_local_rank & dist_mask), 0));
      dist_mask <<= 1;

      if (remapped_local_rank < remapped_peer)
      {
        tmpsend = (char *)tmprecv_buff + (ptrdiff_t)send_block_location * (ptrdiff_t)rcount * rext;
        tmprecv = (char *)tmprecv_buff + (ptrdiff_t)(send_block_location + local_data_to_send) * (ptrdiff_t)rcount * rext;
      }
      else
      {
        tmpsend = (char *)tmprecv_buff + (ptrdiff_t)send_block_location * (ptrdiff_t)rcount * rext;
        tmprecv = (char *)tmprecv_buff + (ptrdiff_t)(send_block_location - local_data_to_recv) * (ptrdiff_t)rcount * rext;
        send_block_location -= local_data_to_recv;
      }

      peer = node_offset + (remapped_peer < remaning_local && local_sub_group != task_on_node ? (remapped_peer * 2) : (remapped_peer + remaning_local));

      /* Sendreceive */
      PICO_TAG_BEGIN("local_comm/exchange/send_recv");
      err = MPI_Sendrecv(tmpsend, (ptrdiff_t)local_data_to_send * (ptrdiff_t)rcount, rdtype, peer, 0,
                         tmprecv, (ptrdiff_t)local_data_to_recv * (ptrdiff_t)rcount, rdtype,
                         peer, 0, comm, MPI_STATUS_IGNORE);
      PICO_TAG_END("local_comm/exchange/send_recv");
      if (MPI_SUCCESS != err)
      {
        line = __LINE__;
        goto err_hndl;
      }
    }
  }
  PICO_TAG_END("local_comm/exchange");

  PICO_TAG_BEGIN("local_comm/send_to_excluded_rank");
  if (local_sub_group != task_on_node)
  {
    if ((local_rank >> 1) < remaning_local && !(local_rank & 1))
    {
      tmpsend = (char *)tmprecv_buff + (ptrdiff_t)node_offset * (ptrdiff_t)rcount * rext;
      err = MPI_Send(tmpsend, rcount * (local_rank + 1), rdtype, rank + 1, 0, comm);
      if (MPI_SUCCESS != err)
      {
        line = __LINE__;
        goto err_hndl;
      }
      tmpsend = (char *)tmprecv_buff + (ptrdiff_t)(rank + 1) * (ptrdiff_t)rcount * rext;
      err = MPI_Send(tmpsend, rcount * (task_on_node - (local_rank + 1)), rdtype, rank + 1, 0, comm);
      if (MPI_SUCCESS != err)
      {
        line = __LINE__;
        goto err_hndl;
      }
    }
    else if ((local_rank >> 1) < remaning_local && local_rank & 1)
    {
      tmprecv = (char *)tmprecv_buff + (ptrdiff_t)node_offset * (ptrdiff_t)rcount * rext;
      err = MPI_Recv(tmprecv, rcount * local_rank, rdtype, rank - 1, 0, comm, MPI_STATUS_IGNORE);
      if (MPI_SUCCESS != err)
      {
        line = __LINE__;
        goto err_hndl;
      }
      tmprecv = (char *)tmprecv_buff + (ptrdiff_t)rank * (ptrdiff_t)rcount * rext;
      err = MPI_Recv(tmprecv, rcount * (task_on_node - local_rank), rdtype, rank - 1, 0, comm, MPI_STATUS_IGNORE);
      if (MPI_SUCCESS != err)
      {
        line = __LINE__;
        goto err_hndl;
      }
    }
  }
  PICO_TAG_END("local_comm/send_to_excluded_rank");
  PICO_TAG_END("local_comm");
  // end local comunication

  PICO_TAG_BEGIN("global_comm");
  // global allgather
  PICO_TAG_BEGIN("global_comm/setup");
  send_block_location = node_offset;
  node_sub_group = floor_power_of_two(node_size);
  remaining_node = node_size - node_sub_group;
  dist_mask = ~0;
  PICO_TAG_END("global_comm/setup");

  if (rank % task_on_node == 0)
  {
    // share data betwin extra node and node in the group
    PICO_TAG_BEGIN("global_comm/recv_from_excluded_node");
    if (node_sub_group != node_size)
    {
      if ((node_rank >> 1) < remaining_node && node_rank & 1)
      {
        tmpsend = (char *)tmprecv_buff + (ptrdiff_t)send_block_location * (ptrdiff_t)rcount * rext;
        err = MPI_Send(tmpsend, task_on_node * rcount, rdtype, (node_rank - 1) * task_on_node, 0, comm);
      }
      else if ((node_rank >> 1) < remaining_node && !(node_rank & 1))
      {
        tmprecv = (char *)tmprecv_buff + (ptrdiff_t)(send_block_location + task_on_node) * (ptrdiff_t)rcount * rext;
        err = MPI_Recv(tmprecv, task_on_node * rcount, rdtype, (node_rank + 1) * task_on_node, 0, comm, MPI_STATUS_IGNORE);
      }

      if (MPI_SUCCESS != err)
      {
        line = __LINE__;
        goto err_hndl;
      }
    }
    PICO_TAG_END("global_comm/recv_from_excluded_node");

    // exchange data in sub group
    PICO_TAG_BEGIN("global_comm/exchange");
    if (!(node_rank & 1) || node_rank >= (remaining_node << 1) || node_size == node_sub_group)
    {
      remapped_node_rank = (node_rank >> 1) < remaining_node && node_size != node_sub_group ? node_rank / 2 : node_rank - remaining_node;


      for (distance = 0x1; distance < node_sub_group; distance <<= 1)
      {
        remapped_peer = remapped_node_rank ^ distance;
        node_data_to_recv = (distance + min(distance, max(remaining_node - (remapped_peer & dist_mask), 0))) * task_on_node;
        node_data_to_send = (distance + min(distance, max(remaining_node - (remapped_node_rank & dist_mask), 0))) * task_on_node;
        dist_mask <<= 1;
        
        if (remapped_node_rank < remapped_peer)
        {
          tmpsend = (char *)tmprecv_buff + (ptrdiff_t)send_block_location * (ptrdiff_t)rcount * rext;
          tmprecv = (char *)tmprecv_buff + (ptrdiff_t)(send_block_location + node_data_to_send) * (ptrdiff_t)rcount * rext;
        }
        else
        {
          tmpsend = (char *)tmprecv_buff + (ptrdiff_t)send_block_location * (ptrdiff_t)rcount * rext;
          tmprecv = (char *)tmprecv_buff + (ptrdiff_t)(send_block_location - node_data_to_recv) * (ptrdiff_t)rcount * rext;
          send_block_location -= node_data_to_recv;
        }

        peer = remapped_peer < remaining_node && node_size != node_sub_group ? remapped_peer * 2 : remapped_peer + remaining_node;
        peer *= task_on_node;

        /* Sendreceive */
        PICO_TAG_BEGIN("global_comm/exchange/send_recv");
        err = MPI_Sendrecv(tmpsend, node_data_to_send * rcount, rdtype, peer, 0,
                           tmprecv, node_data_to_recv * rcount, rdtype,
                           peer, 0, comm, MPI_STATUS_IGNORE);
        PICO_TAG_END("global_comm/exchange/send_recv");
        if (MPI_SUCCESS != err)
        {
          line = __LINE__;
          goto err_hndl;
        }
      }
    }
    PICO_TAG_END("global_comm/exchange");

    // share data back to extra node
    PICO_TAG_BEGIN("global_comm/send_to_excluded_node");
    if (node_sub_group != node_size)
    {
      if ((node_rank >> 1) < remaining_node && !(node_rank & 1))
      {
        err = MPI_Send(tmprecv_buff, (node_rank + 1) * task_on_node * rcount, rdtype, (node_rank + 1) * task_on_node, 0, comm);
        if (MPI_SUCCESS != err)
        {
          line = __LINE__;
          goto err_hndl;
        }
        tmpsend = (char *)tmprecv_buff + (ptrdiff_t)(node_rank + 1) * task_on_node * (ptrdiff_t)rcount * rext;
        err = MPI_Send(tmpsend, (node_size - (node_rank + 1)) * task_on_node * rcount, rdtype, (node_rank + 1) * task_on_node, 0, comm);
        if (MPI_SUCCESS != err)
        {
          line = __LINE__;
          goto err_hndl;
        }
      }
      else if ((node_rank >> 1) < remaining_node && node_rank & 1)
      {
        err = MPI_Recv(tmprecv_buff, node_rank * task_on_node * rcount, rdtype, (node_rank - 1) * task_on_node, 0, comm, MPI_STATUS_IGNORE);
        if (MPI_SUCCESS != err)
        {
          line = __LINE__;
          goto err_hndl;
        }
        tmprecv = (char *)tmprecv_buff + (ptrdiff_t)node_rank * task_on_node * (ptrdiff_t)rcount * rext;
        err = MPI_Recv(tmprecv, (node_size - node_rank) * task_on_node * rcount, rdtype, (node_rank - 1) * task_on_node, 0, comm, MPI_STATUS_IGNORE);
        if (MPI_SUCCESS != err)
        {
          line = __LINE__;
          goto err_hndl;
        }
      }
    }
    PICO_TAG_END("global_comm/send_to_excluded_node");
  }
  PICO_TAG_END("global_comm");
  // end global

  // share data back localy
  // TODO: make all the node that recive data cominucate to other
  PICO_TAG_BEGIN("final_local_exchange");
  for (distance = 0x1; distance < local_sub_group; distance <<= 1)
  {
    if (local_rank >= (distance << 1))
      continue;

    peer = local_rank ^ distance;

    if (local_rank < peer)
    {
      MPI_Send(tmprecv_buff, size * rcount, rdtype, peer + node_offset, 0, comm);
    }
    else
    {
      MPI_Recv(tmprecv_buff, size * rcount, rdtype, peer + node_offset, 0, comm, MPI_STATUS_IGNORE);
    }
  }

  if (local_rank < remaning_local)
  {
    MPI_Send(tmprecv_buff, size * rcount, rdtype, rank + local_sub_group, 0, comm);
  }
  else if (local_rank >= local_sub_group)
  {
    MPI_Recv(tmprecv_buff, size * rcount, rdtype, rank - local_sub_group, 0, comm, MPI_STATUS_IGNORE);
  }
  PICO_TAG_END("final_local_exchange");

#if defined PICO_MPI_CUDA_AWARE && !defined GPU_NATIVE_SUPPORT
  BINE_CUDA_CHECK(cudaMemcpy(rbuf, tmprecv_buff, size * rcount * rext, cudaMemcpyHostToDevice));
  if (tmprecv_buff != NULL)
  {
    free(tmprecv_buff);
  }
#endif

  return MPI_SUCCESS;
err_hndl:
  BINE_DEBUG_PRINT("\n%s:%4d\tRank %d Error occurred %d\n\n", __FILE__, line, rank, err);
  (void)line; // silence compiler warning
  return err;
}

int allgather_recursivedoubling_any_even(const void *sbuf, size_t scount, MPI_Datatype sdtype,
                                            void *rbuf, size_t rcount, MPI_Datatype rdtype, MPI_Comm comm)
{
  int line = -1, rank, size, sub_group_size, remaining_node, err = MPI_SUCCESS;
  int peer, distance, lower_block_data, upper_block_data, peer_group, rank_group, dist_mask = ~0;
  ptrdiff_t rlb, rext;
  char *tmprecv = NULL, *tmpsend = NULL, *tmprecv_buff;

  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  PICO_TAG_BEGIN("setup");
  err = MPI_Type_get_extent(rdtype, &rlb, &rext);
  if (MPI_SUCCESS != err)
  {
    line = __LINE__;
    goto err_hndl;
  }

#ifdef PICO_MPI_CUDA_AWARE
  tmprecv_buff = (char *)calloc(size * rcount, rext);
  if (tmprecv_buff == NULL)
  {
    line = __LINE__;
    err = MPI_ERR_NO_MEM;
    goto err_hndl;
  }
#else
  tmprecv_buff = rbuf;
#endif

#ifndef PICO_MPI_CUDA_AWARE
  // Setup buffer for mpi if not in place
  if (MPI_IN_PLACE != sbuf)
  {
    tmpsend = (char *)sbuf;
    tmprecv = tmprecv_buff + (ptrdiff_t)rank * (ptrdiff_t)rcount * rext;

    err = COPY_BUFF_DIFF_DT(tmpsend, scount, sdtype, tmprecv, rcount, rdtype);
    if (MPI_SUCCESS != err)
    {
      line = __LINE__;
      goto err_hndl;
    }
  }
#else
  tmpsend = (char *)sbuf;
  tmprecv = tmprecv_buff + (ptrdiff_t)rank * (ptrdiff_t)rcount * rext;
  BINE_CUDA_CHECK(cudaMemcpy(tmprecv, tmpsend, rcount * rext, cudaMemcpyDeviceToHost));
#endif

  // findi sub group power of 2
  sub_group_size = floor_power_of_two(size);
  remaining_node = size - sub_group_size;
  lower_block_data = rank;
  upper_block_data = rank;
  PICO_TAG_END("setup");

  // bind remaining rank to rank in sub group
  PICO_TAG_BEGIN("recv_from_extra_rank");
  if (rank >= sub_group_size)
  {
    tmpsend = tmprecv_buff + (ptrdiff_t)rank * (ptrdiff_t)scount * rext;
    err = MPI_Send(tmpsend, scount, sdtype, rank - sub_group_size, 0, comm);

    if (MPI_SUCCESS != err)
    {
      line = __LINE__;
      goto err_hndl;
    }
  }
  else if (rank < remaining_node)
  {

    tmprecv = tmprecv_buff + (ptrdiff_t)(sub_group_size + rank) * (ptrdiff_t)rcount * rext;
    err = MPI_Recv(tmprecv, rcount, sdtype, sub_group_size + rank, 0, comm, MPI_STATUS_IGNORE);

    if (MPI_SUCCESS != err)
    {
      line = __LINE__;
      goto err_hndl;
    }

    upper_block_data = sub_group_size + rank;
  }
  PICO_TAG_END("recv_from_extra_rank");

  PICO_TAG_BEGIN("exchange");
  if (rank < sub_group_size)
  {
    // perform exchange in sub group
    for (distance = 0x1; distance < sub_group_size; distance <<= 1)
    {
      peer = rank ^ distance;

      if (rank < peer)
      {
        tmpsend = tmprecv_buff + (ptrdiff_t)lower_block_data * (ptrdiff_t)rcount * rext;
        tmprecv = tmprecv_buff + (ptrdiff_t)(lower_block_data + distance) * (ptrdiff_t)rcount * rext;
      }
      else
      {
        tmpsend = tmprecv_buff + (ptrdiff_t)lower_block_data * (ptrdiff_t)rcount * rext;
        tmprecv = tmprecv_buff + (ptrdiff_t)(lower_block_data - distance) * (ptrdiff_t)rcount * rext;
        lower_block_data -= distance;
      }

      /* Sendreceive */
      PICO_TAG_BEGIN("exchange/send_recv");
      err = MPI_Sendrecv(tmpsend, (ptrdiff_t)distance * (ptrdiff_t)rcount, rdtype, peer, 0,
                         tmprecv, (ptrdiff_t)distance * (ptrdiff_t)rcount, rdtype,
                         peer, 0, comm, MPI_STATUS_IGNORE);
      PICO_TAG_END("exchange/send_recv");
      if (MPI_SUCCESS != err)
      {
        line = __LINE__;
        goto err_hndl;
      }
      // calc first node of the sub group of current e peer node
      peer_group = peer & dist_mask;
      rank_group = rank & dist_mask;
      dist_mask = dist_mask << 1;

      // send and recive extra data
      PICO_TAG_BEGIN("exchange/extradata");
      if (peer_group < remaining_node && rank_group < remaining_node)
      {
        tmpsend = tmprecv_buff + (ptrdiff_t)upper_block_data * (ptrdiff_t)rcount * rext;
        tmprecv = tmprecv_buff + (ptrdiff_t)(sub_group_size + peer_group) * (ptrdiff_t)rcount * rext;
        if (peer_group < rank)
        {
          upper_block_data = sub_group_size + peer_group;
        }

        PICO_TAG_BEGIN("exchange/extradata/send_recv");
        err = MPI_Sendrecv(tmpsend, (ptrdiff_t)remining_data_to_share(remaining_node, rank_group, distance) * (ptrdiff_t)rcount, rdtype, peer, 0,
                           tmprecv, (ptrdiff_t)remining_data_to_share(remaining_node, peer_group, distance) * (ptrdiff_t)rcount, rdtype,
                           peer, 0, comm, MPI_STATUS_IGNORE);
        PICO_TAG_END("exchange/extradata/send_recv");
        if (MPI_SUCCESS != err)
        {
          line = __LINE__;
          goto err_hndl;
        }
      }
      // send extra data
      else if (rank_group < remaining_node)
      {
        tmpsend = tmprecv_buff + (ptrdiff_t)upper_block_data * (ptrdiff_t)rcount * rext;
        err = MPI_Send(tmpsend, (ptrdiff_t)remining_data_to_share(remaining_node, rank_group, distance) * (ptrdiff_t)rcount, rdtype, peer, 0, comm);
        if (MPI_SUCCESS != err)
        {
          line = __LINE__;
          goto err_hndl;
        }
      }
      // recive extra data
      else if (peer_group < remaining_node)
      {
        tmprecv = tmprecv_buff + (ptrdiff_t)(sub_group_size + peer_group) * (ptrdiff_t)rcount * rext;
        if (peer_group < rank)
        {
          upper_block_data = sub_group_size + peer_group;
        }

        err = MPI_Recv(tmprecv, (ptrdiff_t)remining_data_to_share(remaining_node, peer_group, distance) * (ptrdiff_t)rcount, rdtype, peer, 0, comm, MPI_STATUS_IGNORE);
        if (MPI_SUCCESS != err)
        {
          line = __LINE__;
          goto err_hndl;
        }
      }
      PICO_TAG_END("exchange/extradata");
    }
  }
  PICO_TAG_END("exchange");

  // return value to excluded rank
  PICO_TAG_BEGIN("send_to_excluded_rank");
  if (rank >= sub_group_size)
  {
    peer = rank - sub_group_size;

    // first blok
    err = MPI_Recv(tmprecv_buff, (ptrdiff_t)rank * rcount, sdtype, peer, 0, comm, MPI_STATUS_IGNORE);
    if (MPI_SUCCESS != err)
    {
      line = __LINE__;
      goto err_hndl;
    }

    // second block edge case for the last rank
    if (size - (rank + 1) > 0)
    {
      tmprecv = tmprecv_buff + (ptrdiff_t)(rank + 1) * (ptrdiff_t)rcount * rext;
      err = MPI_Recv(tmprecv, (ptrdiff_t)(size - (rank + 1)) * rcount, sdtype, peer, 0, comm, MPI_STATUS_IGNORE);

      if (MPI_SUCCESS != err)
      {
        line = __LINE__;
        goto err_hndl;
      }
    }
  }
  else if (rank < remaining_node)
  {
    peer = rank + sub_group_size;
    // first block
    err = MPI_Send(tmprecv_buff, (ptrdiff_t)peer * rcount, sdtype, peer, 0, comm);
    if (MPI_SUCCESS != err)
    {
      line = __LINE__;
      goto err_hndl;
    }

    // second block edge case for the last rank
    if (size - (peer + 1) > 0)
    {
      tmpsend = tmprecv_buff + (ptrdiff_t)(peer + 1) * (ptrdiff_t)rcount * rext;
      err = MPI_Send(tmpsend, (ptrdiff_t)(size - (peer + 1)) * rcount, sdtype, peer, 0, comm);

      if (MPI_SUCCESS != err)
      {
        line = __LINE__;
        goto err_hndl;
      }
    }
  }
  PICO_TAG_END("send_to_excluded_rank");

#ifdef PICO_MPI_CUDA_AWARE
  BINE_CUDA_CHECK(cudaMemcpy(rbuf, tmprecv_buff, size * rcount * rext, cudaMemcpyHostToDevice));
#endif

  return MPI_SUCCESS;
err_hndl:
  BINE_DEBUG_PRINT("\n%s:%4d\tRank %d Error occurred %d\n\n", __FILE__, line, rank, err);
  (void)line; // silence compiler warning
  return err;
}

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
    
#ifdef PICO_MPI_CUDA_AWARE
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
  if( NULL != reqs ) {
    cleanup_reqs(&req_manager);
  }
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

  PICO_TAG_BEGIN("setup");

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  steps = log_2(size);
  if(!is_power_of_two(size) || steps < 1) {
    BINE_DEBUG_PRINT("ERROR! bine static allgather works only with po2 ranks!");
    return MPI_ERR_ARG;
  }

  err = MPI_Type_get_extent (rdtype, &rlb, &rext);
  if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }

  PICO_TAG_BEGIN("setup/buffer_copy");
  if(MPI_IN_PLACE != sbuf) {
    tmpsend = (char*) sbuf;
    tmprecv = (char*) rbuf + (ptrdiff_t)rank * (ptrdiff_t)rcount * rext;

    err = COPY_BUFF_DIFF_DT(tmpsend, scount, sdtype, tmprecv, rcount, rdtype);
    if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl;  }
  }
  PICO_TAG_END("setup/buffer_copy");
  PICO_TAG_BEGIN("setup/bitmap_setup");
  s_bitmap = (int *) malloc(size * sizeof(int));
  r_bitmap = (int *) malloc(size * sizeof(int));
  requests = (MPI_Request *) malloc(size * sizeof(MPI_Request));
  if(s_bitmap == NULL || r_bitmap == NULL || requests == NULL){
    line = __LINE__;
    err = MPI_ERR_NO_MEM;
    goto err_hndl;
  }
  PICO_TAG_END("setup/bitmap_setup");

  PICO_TAG_END("setup");
  PICO_TAG_BEGIN("comunication");

  for(int step = steps - 1; step >= 0; step--) {
    int num_reqs = 0;
    remote = pi(rank, step, size);

    PICO_TAG_BEGIN("comunication/bitmap_set");
    memset(s_bitmap, 0, size * sizeof(int));
    memset(r_bitmap, 0, size * sizeof(int));
    get_indexes(rank, step, steps, size, r_bitmap);
    get_indexes(remote, step, steps, size, s_bitmap);
    PICO_TAG_END("comunication/bitmap_set");

    PICO_TAG_BEGIN("comunication/block_exchange");
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
    PICO_TAG_END("comunication/block_exchange");
    PICO_TAG_BEGIN("comunication/wait_requests");
    err = MPI_Waitall(num_reqs, requests, MPI_STATUSES_IGNORE);
    if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }
    PICO_TAG_END("comunication/wait_requests");
  }

  PICO_TAG_END("comunication");

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
  COPY_BUFF_DIFF_DT(sendbuf, sendcount, recvtype, (char*) recvbuf + sendcount * rank * dtsize, recvcount, recvtype);
  //memcpy((char*) recvbuf + sendcount * rank * dtsize, sendbuf, sendcount * dtsize);

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



int allgather_bine_send_remap(const void *sbuf, size_t scount, MPI_Datatype sdtype,
                           void* rbuf, size_t rcount, MPI_Datatype rdtype, MPI_Comm comm)
{
  int line = -1, rank, size, steps, err = MPI_SUCCESS;
  int vrank, remote, vremote, send_block_location, distance;
  ptrdiff_t rlb, rext;
  char *tmpsend = NULL, *tmprecv = NULL;

  PICO_TAG_BEGIN("setup");
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

    err = COPY_BUFF_DIFF_DT(tmpsend, scount, sdtype, tmprecv, rcount, rdtype);
    if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl;  }
  }
  PICO_TAG_END("setup");
  PICO_TAG_BEGIN("comunication");

  /* Communication step:
     At every step i, rank r:
     - exchanges message with rank remote = (r ^ 2^i).
  */
  distance = 0x1;
  send_block_location = vrank;
  for(int step = steps - 1; step >= 0; step--) {
    size_t step_scount = rcount * distance;
    remote = pi(rank, step, size);
    vremote = (int) remap_rank((uint32_t) size, (uint32_t) remote);

    if(vrank < vremote){
      tmpsend = (char*)rbuf + (ptrdiff_t)send_block_location * (ptrdiff_t)rcount * rext;
      tmprecv = (char*)rbuf + (ptrdiff_t)(send_block_location + distance) * (ptrdiff_t)rcount * rext;
    } else {
      tmpsend = (char*)rbuf + (ptrdiff_t)send_block_location * (ptrdiff_t)rcount * rext;
      tmprecv = (char*)rbuf + (ptrdiff_t)(send_block_location - distance) * (ptrdiff_t)rcount * rext;
      send_block_location -= distance;
    }

    PICO_TAG_BEGIN("comunication/send_reciv");
    /* Sendreceive */
    err = MPI_Sendrecv(tmpsend, step_scount, rdtype, remote, 0, 
                       tmprecv, step_scount, rdtype, remote, 0,
                       comm, MPI_STATUS_IGNORE);
    if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }
    PICO_TAG_END("comunication/send_reciv");
    distance <<=1;
  } 
  PICO_TAG_END("comunication");

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

  PICO_TAG_BEGIN("setup");
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
  PICO_TAG_END("setup");

  PICO_TAG_BEGIN("comunication");
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

    PICO_TAG_BEGIN("comunication/extra_comm");
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
    PICO_TAG_END("comunication/extra_comm");

    // Simple case: no wrap-around
    tmpsend = (char*)rbuf + (ptrdiff_t)send_index * (ptrdiff_t)rcount * rext;
    tmprecv = (char*)rbuf + (ptrdiff_t)recv_index * (ptrdiff_t)rcount * rext;

    PICO_TAG_BEGIN("comunication/send_reciv");
    err = MPI_Sendrecv(tmpsend, send_count * rcount, rdtype, remote, 0, 
                       tmprecv, recv_count * rcount, rdtype, remote, 0,
                       comm, MPI_STATUS_IGNORE);
    if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }
    PICO_TAG_END("comunication/send_reciv");
    
    PICO_TAG_BEGIN("comunication/wait_extra_req");
    if (extra_recv != 0) {
      err = MPI_Wait(&req, MPI_STATUS_IGNORE);
      if (MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }
    }
    PICO_TAG_END("comunication/wait_extra_req");

    mask <<= 1;
  }
  PICO_TAG_END("comunication");

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

int allgather_bine_permutation(const void *sbuf, size_t scount, MPI_Datatype sdtype,
                           void* rbuf, size_t rcount, MPI_Datatype rdtype, MPI_Comm comm){
  int line = -1, rank, size, steps, err = MPI_SUCCESS, remote, data_exchange;
  int *permutation = NULL;
  ptrdiff_t rlb, rext;
  char *tmprecv = NULL;;

  PICO_TAG_BEGIN("setup");

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
    err = COPY_BUFF_DIFF_DT(sbuf, scount, sdtype, rbuf, rcount, rdtype);
    if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl;  }
  }

  permutation = (int *) malloc(size * sizeof(int));
  if(permutation == NULL){
    line = __LINE__;
    err = MPI_ERR_NO_MEM;
    goto err_hndl;
  }

  memset(permutation, -1, size * sizeof(int));
  *(permutation + rank) = 0;
  PICO_TAG_END("setup");

  PICO_TAG_BEGIN("comunication");
  data_exchange = 1;
  for(int step = steps - 1; step >= 0; step--) {
    remote = pi(rank, step, size);

    PICO_TAG_BEGIN("comunication/permutation_calc");
    get_permutation(rank, step, steps, size, permutation, data_exchange);
    PICO_TAG_END("comunication/permutation_calc");

    tmprecv = (char*) rbuf + (ptrdiff_t)data_exchange * (ptrdiff_t)rcount * rext;

    PICO_TAG_BEGIN("comunication/send_reciv");
    err = MPI_Sendrecv(rbuf, data_exchange * rcount, rdtype, remote, 0, tmprecv, data_exchange * rcount, rdtype, remote, 0, comm, MPI_STATUS_IGNORE);
    if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }
    PICO_TAG_END("comunication/send_reciv");
    data_exchange <<= 1;
  }
  PICO_TAG_END("comunication");

  PICO_TAG_BEGIN("reorder_block");
  err = reorder_blocks_gpu(rbuf, rcount, rdtype, permutation, size);
  if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }
  PICO_TAG_END("reorder_block");

  if(permutation != NULL) 
    free(permutation);

  return MPI_SUCCESS;

err_hndl:
  BINE_DEBUG_PRINT("\n%s:%4d\tError occurred %d, rank %2d\n\n", __FILE__, line, err, rank);
  (void)line;  // silence compiler warning
  if(permutation != NULL) free(permutation);
  return err;
}

int allgather_bine_block_by_block_hierarcic_global_local(const void *sbuf, size_t scount, MPI_Datatype sdtype,
                           void* rbuf, size_t rcount, MPI_Datatype rdtype, MPI_Comm comm){
  int line = -1, rank, size, steps, err = MPI_SUCCESS, remote;
  int node_size, node_rank, node_offset, local_rank;
  int num_reqs;
  int *s_bitmap = NULL, *r_bitmap = NULL;
  ptrdiff_t rlb, rext;
  char *tmpsend = NULL, *tmprecv = NULL;
  MPI_Request *requests = NULL;
  int task_on_node = pico_task_on_node();

  PICO_TAG_BEGIN("setup");
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  pico_get_group_config(&node_size, &node_rank, &node_offset, &local_rank, task_on_node, size, rank);

  steps = log_2(node_size);
  if(!is_power_of_two(size) || (steps < 1 && node_size > 1)) {
    BINE_DEBUG_PRINT("ERROR! bine static allgather works only with po2 ranks!");
    return MPI_ERR_ARG;
  }

  err = MPI_Type_get_extent (rdtype, &rlb, &rext);
  if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }

  PICO_TAG_BEGIN("setup/buffer_copy");
  if(MPI_IN_PLACE != sbuf) {
    tmpsend = (char*) sbuf;
    tmprecv = (char*) rbuf + (ptrdiff_t)rank * (ptrdiff_t)rcount * rext;

    err = COPY_BUFF_DIFF_DT(tmpsend, scount, sdtype, tmprecv, rcount, rdtype);
    if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl;  }
  }
  PICO_TAG_END("setup/buffer_copy");
  PICO_TAG_BEGIN("setup/bitmap_setup");
  s_bitmap = (int *) malloc(node_size * sizeof(int));
  r_bitmap = (int *) malloc(node_size * sizeof(int));
  requests = (MPI_Request *) malloc(size * 2 * sizeof(MPI_Request));
  if(s_bitmap == NULL || r_bitmap == NULL || requests == NULL){
    line = __LINE__;
    err = MPI_ERR_NO_MEM;
    goto err_hndl;
  }
  PICO_TAG_END("setup/bitmap_setup");
  PICO_TAG_END("setup");

  PICO_TAG_BEGIN("global_comm");
  for(int step = steps - 1; step >= 0; step--) {
    num_reqs = 0;
    remote = pi(node_rank, step, node_size);

    PICO_TAG_BEGIN("global_comm/bitmap_set");
    memset(s_bitmap, 0, node_size * sizeof(int));
    memset(r_bitmap, 0, node_size * sizeof(int));
    get_indexes(node_rank, step, steps, node_size, r_bitmap);
    get_indexes(remote, step, steps, node_size, s_bitmap);
    PICO_TAG_END("global_comm/bitmap_set");

    remote = remote * task_on_node + local_rank;

    PICO_TAG_BEGIN("global_comm/block_exchange");
    for(int block = 0; block < node_size; block++){
      if(s_bitmap[block] != 0){
        tmpsend = (char*)rbuf + (ptrdiff_t)(block * task_on_node + local_rank) * (ptrdiff_t)rcount * rext;
        err = MPI_Isend(tmpsend, rcount, rdtype, remote, block, comm, requests + num_reqs);
        if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }
        num_reqs++;
      }
      if(r_bitmap[block] != 0){
        tmprecv = (char*)rbuf + (ptrdiff_t)(block * task_on_node + local_rank) * (ptrdiff_t)rcount * rext;
        err = MPI_Irecv(tmprecv, rcount, rdtype, remote, block, comm, requests + num_reqs);
        if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }
        num_reqs++;
      }
    }
    PICO_TAG_END("global_comm/block_exchange");

    PICO_TAG_BEGIN("global_comm/req_wait");
    err = MPI_Waitall(num_reqs, requests, MPI_STATUSES_IGNORE);
    if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }
    PICO_TAG_END("global_comm/req_wait");
  }
  PICO_TAG_END("global_comm");

  // local exchange
  PICO_TAG_BEGIN("local_comm");
  num_reqs = 0;
  for (int i = 0; i < task_on_node; i++)
  {
    if (i == local_rank)
      continue;

    for (int j = 0; j < node_size; j++)
    {
      tmpsend = (char*) rbuf + (ptrdiff_t)(j * task_on_node + local_rank) * (ptrdiff_t)rcount * rext;
      tmprecv = (char*) rbuf + (ptrdiff_t)(j * task_on_node + i) * rcount * rext;

      err = MPI_Isend(tmpsend, rcount, rdtype, node_offset + i, 0, comm, &requests[num_reqs]);
      if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }
      num_reqs++;
      err = MPI_Irecv(tmprecv, rcount, rdtype, node_offset + i, 0, comm, &requests[num_reqs]);
      if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }
      num_reqs++;
    }
  }
  PICO_TAG_BEGIN("local_comm/local_req_wait");
  err = MPI_Waitall(num_reqs, requests, MPI_STATUS_IGNORE);
  if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }
  PICO_TAG_END("local_comm/local_req_wait");
  PICO_TAG_END("local_comm");

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

int allgather_bine_send_remap_hierarcic_global_local(const void *sbuf, size_t scount, MPI_Datatype sdtype,
                           void* rbuf, size_t rcount, MPI_Datatype rdtype, MPI_Comm comm)
{
  int line = -1, rank, size, steps, err = MPI_SUCCESS;
  int vrank, remote, vremote, send_block_location, distance;
  int node_size, node_rank, node_offset, local_rank;
  ptrdiff_t rlb, rext;
  char *tmpsend = NULL, *tmprecv = NULL;
  void *perm_buff = NULL, *global_temp = NULL;
  int task_on_node = pico_task_on_node();
  MPI_Request requests[task_on_node * 2];

  PICO_TAG_BEGIN("setup");
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  pico_get_group_config(&node_size, &node_rank, &node_offset, &local_rank, task_on_node, size, rank);

  /*
   * Current implementation only handles power-of-two number of processes.
   */
  steps = log_2(node_size);
  if(!is_power_of_two(size) || (steps < 1 && node_size > 1)) {
    BINE_DEBUG_PRINT("ERROR! bine static allgather works only with po2 ranks!");
    return MPI_ERR_ARG;
  }

  err = MPI_Type_get_extent (rdtype, &rlb, &rext);
  if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }

#ifdef PICO_MPI_CUDA_AWARE
  BINE_CUDA_CHECK(cudaMalloc((void **)&perm_buff, size * rcount * rext));
#else
  perm_buff = malloc(size * rcount * rext);
#endif
  if (perm_buff == NULL) { line = __LINE__; goto err_hndl; }

  /* Initialization step:
   * - if I gather the result for another rank, I send my buffer to that rank
   *   and I receive the data from the rank at the inverse permutation
   * - if I gather the result for myself, I copy the data from the send buffer
   */
  PICO_TAG_BEGIN("setup/data_exchange");
  vrank = (int) remap_rank((uint32_t) node_size, (uint32_t) node_rank);
  int node_to_rank = vrank * task_on_node + local_rank;
  if(vrank != node_rank) {
    tmprecv = (char*) perm_buff + (ptrdiff_t)(local_rank * node_size + vrank) * (ptrdiff_t)rcount * rext;
    err = MPI_Sendrecv(sbuf, scount, sdtype, get_sender_rec(node_size, node_rank) * task_on_node + local_rank, 0,
                       tmprecv, rcount, rdtype, node_to_rank, 0,
                       comm, MPI_STATUS_IGNORE);
    if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }
  }
  else{
    tmpsend = (char*) sbuf;
    tmprecv = (char*) perm_buff + (ptrdiff_t)(local_rank * node_size + vrank) * (ptrdiff_t)rcount * rext;

    err = COPY_BUFF_DIFF_DT(tmpsend, scount, sdtype, tmprecv, rcount, rdtype);
    if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl;  }
  }
  PICO_TAG_END("setup/data_exchange");
  PICO_TAG_END("setup");

  /* Communication step:
     At every step i, rank r:
     - exchanges message with rank remote = (r ^ 2^i).
  */
  PICO_TAG_BEGIN("gloabbal_comm");
  distance = 0x1;
  send_block_location = vrank;
  global_temp = (char*)perm_buff + (ptrdiff_t)local_rank * (ptrdiff_t)node_size * (ptrdiff_t)rcount * rext;
  for(int step = steps - 1; step >= 0; step--) {
    size_t step_scount = rcount * distance;
    remote = pi(node_rank, step, node_size);
    vremote = (int) remap_rank((uint32_t) node_size, (uint32_t) remote);
    node_to_rank = remote * task_on_node + local_rank;

    if(vrank < vremote){
      tmpsend = (char*)global_temp + (ptrdiff_t)send_block_location * (ptrdiff_t)rcount * rext;
      tmprecv = (char*)global_temp + (ptrdiff_t)(send_block_location + distance) * (ptrdiff_t)rcount * rext;
    } else {
      tmpsend = (char*)global_temp + (ptrdiff_t)send_block_location * (ptrdiff_t)rcount * rext;
      tmprecv = (char*)global_temp + (ptrdiff_t)(send_block_location - distance) * (ptrdiff_t)rcount * rext;
      send_block_location -= distance;
    }

    PICO_TAG_BEGIN("gloabbal_comm/sendrecv");
    /* Sendreceive */
    err = MPI_Sendrecv(tmpsend, step_scount, rdtype, node_to_rank, 0, 
                       tmprecv, step_scount, rdtype, node_to_rank, 0,
                       comm, MPI_STATUS_IGNORE);
    if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }
    PICO_TAG_END("gloabbal_comm/sendrecv");
    distance <<=1;
  } 
  PICO_TAG_END("gloabbal_comm");

  PICO_TAG_BEGIN("local_exchange");
  // local exchange
  int num_reqs = 0;
  tmpsend = global_temp;
  for (int i = 0; i < task_on_node; i++)
  {
    if (i == local_rank)
      continue;
      
    tmprecv = (char*)perm_buff + (ptrdiff_t)i * node_size * rcount * rext;

    err = MPI_Isend(tmpsend, node_size * rcount, rdtype, node_offset + i, 0, comm, &requests[num_reqs]);
    if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }
    num_reqs++;
    err = MPI_Irecv(tmprecv, node_size * rcount, rdtype, node_offset + i, 0, comm, &requests[num_reqs]);
    if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }
    num_reqs++;
  }
  PICO_TAG_BEGIN("local_exchange/request_wait");
  err = MPI_Waitall(num_reqs, requests, MPI_STATUS_IGNORE);
  if(MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }
  PICO_TAG_END("local_exchange/request_wait");

  PICO_TAG_BEGIN("local_exchange/reorder");
#ifdef PICO_MPI_CUDA_AWARE
  reorder_kernel_wrapper(perm_buff, rbuf, rcount, size, task_on_node, rdtype);
  BINE_CUDA_CHECK(cudaDeviceSynchronize());
#else
  for(int i = 0; i < size; i++) {
    int elem_local_rank = i / node_size;
    int elem_node_rank = i % node_size;
    COPY_BUFF_DIFF_DT(perm_buff + i * rcount * rext, rcount, rdtype, 
      rbuf + ((elem_node_rank * task_on_node + elem_local_rank) * rcount) * rext, rcount, rdtype);
  }
#endif
  PICO_TAG_END("local_exchange/reorder");

  PICO_TAG_END("local_exchange");

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
// #ifdef PICO_MPI_CUDA_AWARE
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
