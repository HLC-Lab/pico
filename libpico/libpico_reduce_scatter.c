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

#ifdef PICO_MPI_CUDA_AWARE
#include "support_kernel.h"
#endif

int reduce_scatter_recursive_doubling_hierarchical_local_parallel(const void *sbuf, void *rbuf, const int rcounts[],
                                                      MPI_Datatype dtype, MPI_Op op, MPI_Comm comm)
{
  int i, rank, size, err = MPI_SUCCESS;
  ptrdiff_t extent, true_extent, lb, recv_buffer_size, result_buffer_size, gap = 0;
  char *recv_tmp_buff, *result_tmp_buff;
  char *recv_buff_head, *result_buff_head;
  int data_sub_group, local_inverse, local_rank;
  int node_rank, node_size, peer_node;
  int peer, dist_mask, rem_data;
  int send_index = 0, recv_index = 0;
  int task_on_node = pico_task_on_node();
  size_t send_size, recv_size;
  ptrdiff_t *disps = NULL;
  MPI_Request send_req[task_on_node];
  MPI_Request recv_req[task_on_node];
  int req_index, node_offset;

  err = MPI_Comm_size(comm, &size);
  err = MPI_Comm_rank(comm, &rank);

  /* get datatype information */
  MPI_Type_get_extent(dtype, &lb, &extent);
  MPI_Type_get_true_extent(dtype, &gap, &true_extent);

  // calculate memory needed for the buffer
  pico_get_group_config(&node_size, &node_rank, &node_offset, &local_rank, task_on_node, size, rank);
  local_inverse = inverse_rank(task_on_node, local_rank);

  data_sub_group = 0;
  for (i = 0; i < node_size; i++)
  {
    data_sub_group += rcounts[local_inverse * node_size + i];
  }

  /* determinate data displacment  */
  disps = calloc(size, sizeof(ptrdiff_t));
  if (disps == NULL)
    return MPI_ERR_NO_MEM;

  disps[0] = 0;
  for (i = 0; i < (size - 1); i++)
  {
    disps[i + 1] = disps[i] + rcounts[i];
  }

  /* short cut the trivial case */
  if (0 == disps[size - 1] + rcounts[size - 1])
  {
    free(disps);
    return MPI_SUCCESS;
  }

  result_buffer_size = true_extent + extent * data_sub_group;
  recv_buffer_size = true_extent + extent * data_sub_group * (task_on_node - 1);

  if (MPI_IN_PLACE == sbuf)
  {
    sbuf = rbuf;
  }
  /* allocate temporar buffer */
#ifdef PICO_MPI_CUDA_AWARE
  BINE_CUDA_CHECK(cudaMalloc((void **)&recv_tmp_buff, recv_buffer_size));
  BINE_CUDA_CHECK(cudaMalloc((void **)&result_tmp_buff, result_buffer_size));
#else
  recv_tmp_buff = (char *)malloc(recv_buffer_size);
  result_tmp_buff = (char *)malloc(result_buffer_size);

  if (recv_tmp_buff == NULL || result_tmp_buff == NULL)
  {
    err = MPI_ERR_NO_MEM;
    goto cleanup;
  }

#endif

  recv_buff_head = recv_tmp_buff - gap;
  result_buff_head = result_tmp_buff - gap;

  /* recursive doubling local */
  recv_size = send_size = data_sub_group;

  err = COPY_BUFF_DIFF_DT(sbuf + disps[local_inverse * node_size] * extent, recv_size, dtype, result_buff_head, recv_size, dtype);
  if (err != MPI_SUCCESS)
    goto cleanup;

  req_index = 0;
  recv_index = 0;
  for (i = 0; i < task_on_node; i++)
  {
    peer = node_offset + i;
    if (peer == rank)
      continue;

    local_inverse = inverse_rank(task_on_node, i);

    err = MPI_Isend(sbuf + disps[local_inverse * node_size] * extent, send_size, dtype, peer, 0, comm, &send_req[req_index]);
    if (err != MPI_SUCCESS)
      goto cleanup;

    err = MPI_Irecv(recv_buff_head + recv_index * extent, recv_size, dtype, peer, 0, comm, &recv_req[req_index]);
    if (err != MPI_SUCCESS)
      goto cleanup;

    req_index++;
    recv_index += recv_size;
  }

  err = MPI_Waitall(req_index, recv_req, MPI_STATUSES_IGNORE);
  if (err != MPI_SUCCESS)
    goto cleanup;
#ifdef PICO_MPI_CUDA_AWARE
  err = reduce_wrapper_grops(recv_buff_head, result_buff_head, recv_size, task_on_node - 1, dtype, op);
  if (err != MPI_SUCCESS)
    goto cleanup;
  cudaDeviceSynchronize();
#else
  for (i = 0; i < task_on_node - 1; i++)
  {
    err = MPI_Reduce_local(recv_buff_head + i * recv_size * extent, result_buff_head, recv_size, dtype, op);
    if (MPI_SUCCESS != err)
    {
      goto cleanup;
    }
  }
#endif
  err = MPI_Waitall(req_index, send_req, MPI_STATUSES_IGNORE);
  if (err != MPI_SUCCESS)
    goto cleanup;
  /* recursive doubling global */
  int g_send_index, g_recv_index, g_last_index;
  send_index = recv_index = 0;
  rem_data = node_size >> 1;
  g_send_index = g_recv_index = local_inverse * node_size;
  g_last_index = g_recv_index + node_size;

  for (dist_mask = 0x1; dist_mask < node_size; dist_mask <<= 1)
  {
    peer_node = node_rank ^ dist_mask;
    peer = (peer_node * task_on_node) + local_rank;

    send_size = recv_size = 0;

    if (node_rank < peer_node)
    {
      g_send_index = g_recv_index + rem_data;
      for (i = g_send_index; i < g_last_index; ++i)
      {
        send_size += rcounts[i];
      }
      for (i = g_recv_index; i < g_send_index; ++i)
      {
        recv_size += rcounts[i];
      }
      send_index = recv_index + recv_size;
    }
    else
    {
      g_recv_index = g_send_index + rem_data;
      for (i = g_send_index; i < g_recv_index; ++i)
      {
        send_size += rcounts[i];
      }
      for (i = g_recv_index; i < g_last_index; ++i)
      {
        recv_size += rcounts[i];
      }
      recv_index = send_index + send_size;
    }

    if (recv_size > 0)
    {
      err = MPI_Irecv(recv_buff_head, recv_size, dtype, peer, 0, comm, &recv_req[0]);
      if (err != MPI_SUCCESS)
        goto cleanup;
    }
    if (send_size > 0)
    {
      err = MPI_Send(result_buff_head + send_index * extent, send_size, dtype, peer, 0, comm);
      if (err != MPI_SUCCESS)
        goto cleanup;
    }

    if (recv_size > 0)
    {
      err = MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);
      if (MPI_SUCCESS != err)
      {
        goto cleanup;
      }

      // todo: make gpu compatible
#ifdef PICO_MPI_CUDA_AWARE
      err = reduce_wrapper(recv_buff_head, result_buff_head + recv_index * extent, recv_size, dtype, op);
      if (err != MPI_SUCCESS)
        goto cleanup;
      cudaDeviceSynchronize();
#else
      MPI_Reduce_local(recv_buff_head, result_buff_head + recv_index * extent, recv_size, dtype, op);
#endif
    }

    send_index = recv_index;
    g_send_index = g_recv_index;
    g_last_index = g_recv_index + rem_data;
    rem_data >>= 1;
  }

  int inverse = inverse_rank(size, rank);
  if (rank != inverse)
  {
    /* send result to correct rank's recv buffer */
    err = MPI_Sendrecv(result_buff_head + recv_index * extent, rcounts[inverse], dtype, inverse, 0,
                       rbuf, rcounts[rank], dtype, inverse, 0,
                       comm, MPI_STATUS_IGNORE);
    if (MPI_SUCCESS != err)
    {
      goto cleanup;
    }
  }
  else
  {
    /* copy local results from results buffer into real receive buffer */
    err = COPY_BUFF_DIFF_DT(result_buff_head + recv_index * extent, rcounts[rank],
                            dtype, rbuf, rcounts[rank], dtype);
    if (MPI_SUCCESS != err)
    {
      goto cleanup;
    }
  }

cleanup:
  if (NULL != disps)
    free(disps);
#ifdef PICO_MPI_CUDA_AWARE
  if (NULL != recv_tmp_buff)
    BINE_CUDA_CHECK(cudaFree(recv_tmp_buff));
  if (NULL != result_tmp_buff)
    BINE_CUDA_CHECK(cudaFree(result_tmp_buff));
#else
  if (NULL != recv_tmp_buff)
    free(recv_tmp_buff);
  if (NULL != result_tmp_buff)
    free(result_tmp_buff);
#endif
  return err;
}

int reduce_scatter_recursive_doubling_gpu(const void *sbuf, void *rbuf, const int rcounts[],
                                          MPI_Datatype dtype, MPI_Op op, MPI_Comm comm)
{
  int i, rank, size, err = MPI_SUCCESS;
  size_t dcount;
  ptrdiff_t extent, true_extent, lb, buffer_size, gap = 0;
  ptrdiff_t *disps = NULL;
  char *recv_tmp_buff, *result_tmp_buff;
  char *recv_buff_head, *result_buff_head;

  err = MPI_Comm_size(comm, &size);
  err = MPI_Comm_rank(comm, &rank);

  /* determinate data displacment  */
  disps = calloc(size, sizeof(ptrdiff_t));
  if (disps == NULL)
    return MPI_ERR_NO_MEM;

  disps[0] = 0;
  for (i = 0; i < (size - 1); i++)
  {
    disps[i + 1] = disps[i] + rcounts[i];
  }
  dcount = disps[size - 1] + rcounts[size - 1];

  /* short cut the trivial case */
  if (0 == dcount)
  {
    free(disps);
    return MPI_SUCCESS;
  }

  /* get datatype information */
  MPI_Type_get_extent(dtype, &lb, &extent);
  MPI_Type_get_true_extent(dtype, &gap, &true_extent);

  // calculate memory needed for the buffer
  buffer_size = true_extent + extent * (dcount - 1);

  if (MPI_IN_PLACE == sbuf)
  {
    sbuf = rbuf;
  }

  /* allocate temporar buffer */
#ifdef PICO_MPI_CUDA_AWARE
  BINE_CUDA_CHECK(cudaMalloc((void **)&recv_tmp_buff, buffer_size));
  BINE_CUDA_CHECK(cudaMalloc((void **)&result_tmp_buff, buffer_size));
#else
  recv_tmp_buff = (char *)malloc(buffer_size);
  result_tmp_buff = (char *)malloc(buffer_size);

  if (recv_tmp_buff == NULL || result_tmp_buff == NULL)
  {
    err = MPI_ERR_NO_MEM;
    goto cleanup;
  }
#endif

  recv_buff_head = recv_tmp_buff - gap;
  result_buff_head = result_tmp_buff - gap;

  err = COPY_BUFF_DIFF_DT(sbuf, dcount, dtype, result_buff_head, dcount, dtype);
  if (err != MPI_SUCCESS)
    goto cleanup;

  /* recursive doubling */
  int rem_data = size >> 1;
  int dist_mask, last_index = size, peer;
  int send_index = 0, recv_index = 0;
  size_t send_size, recv_size;
  MPI_Request req;
  for (dist_mask = 0x1; dist_mask < size; dist_mask <<= 1)
  {
    peer = rank ^ dist_mask;

    send_size = recv_size = 0;

    if (rank < peer)
    {
      send_index = recv_index + rem_data;
      for (i = send_index; i < last_index; ++i)
      {
        send_size += rcounts[i];
      }
      for (i = recv_index; i < send_index; ++i)
      {
        recv_size += rcounts[i];
      }
    }
    else
    {
      recv_index = send_index + rem_data;
      for (i = send_index; i < recv_index; ++i)
      {
        send_size += rcounts[i];
      }
      for (i = recv_index; i < last_index; ++i)
      {
        recv_size += rcounts[i];
      }
    }

    if (recv_size > 0)
    {
      err = MPI_Irecv(recv_buff_head + disps[recv_index] * extent, recv_size, dtype, peer, 0, comm, &req);
      if (err != MPI_SUCCESS)
        goto cleanup;
    }
    if (send_size > 0)
    {
      err = MPI_Send(result_buff_head + disps[send_index] * extent, send_size, dtype, peer, 0, comm);
      if (err != MPI_SUCCESS)
        goto cleanup;
    }

    if (recv_size > 0)
    {
      err = MPI_Wait(&req, MPI_STATUS_IGNORE);
      if (MPI_SUCCESS != err)
      {
        goto cleanup;
      }

      // todo: make gpu compatible
#ifdef PICO_MPI_CUDA_AWARE
      err = reduce_wrapper(recv_buff_head + disps[recv_index] * extent, result_buff_head + disps[recv_index] * extent, recv_size, dtype, op);
      if (err != MPI_SUCCESS)
        goto cleanup;
      BINE_CUDA_CHECK(cudaDeviceSynchronize());
#else
      MPI_Reduce_local(recv_buff_head + disps[recv_index] * extent, result_buff_head + disps[recv_index] * extent, recv_size, dtype, op);
#endif
    }

    send_index = recv_index;
    last_index = recv_index + rem_data;
    rem_data >>= 1;
  }

  int inverse = inverse_rank(size, rank);
  if (rank != inverse)
  {
    /* send result to correct rank's recv buffer */
    err = MPI_Sendrecv(result_buff_head + disps[inverse] * extent, rcounts[inverse], dtype, inverse, 0,
                       rbuf, rcounts[rank], dtype, inverse, 0,
                       comm, MPI_STATUS_IGNORE);
    if (MPI_SUCCESS != err)
    {
      goto cleanup;
    }
  }
  else
  {
    /* copy local results from results buffer into real receive buffer */
    err = COPY_BUFF_DIFF_DT(result_buff_head + disps[rank] * extent, rcounts[rank],
                            dtype, rbuf, rcounts[rank], dtype);
    if (MPI_SUCCESS != err)
    {
      goto cleanup;
    }
  }

cleanup:
  if (NULL != disps)
    free(disps);
#ifdef PICO_MPI_CUDA_AWARE
  if (NULL != recv_tmp_buff)
    BINE_CUDA_CHECK(cudaFree(recv_tmp_buff));
  if (NULL != result_tmp_buff)
    BINE_CUDA_CHECK(cudaFree(result_tmp_buff));
#else
  if (NULL != recv_tmp_buff)
    free(recv_tmp_buff);
  if (NULL != result_tmp_buff)
    free(result_tmp_buff);
#endif
  return err;
}


int reduce_scatter_recursivehalving(const void *sbuf, void *rbuf, const int rcounts[],
                                    MPI_Datatype dtype, MPI_Op op, MPI_Comm comm)
{
  int i, rank, size, err = MPI_SUCCESS;
  int tmp_size, remain = 0, tmp_rank;
  size_t count;
  ptrdiff_t *disps = NULL;
  ptrdiff_t extent, true_extent, lb, buf_size, gap = 0;
  char *recv_buf = NULL, *recv_buf_free = NULL;
  char *result_buf = NULL, *result_buf_free = NULL;

  err = MPI_Comm_size(comm, &size);
  err = MPI_Comm_rank(comm, &rank);


  /* Find displacements and the like */
  disps = (ptrdiff_t*) malloc(sizeof(ptrdiff_t) * size);
  if(NULL == disps) return MPI_ERR_NO_MEM;

  disps[0] = 0;
  for(i = 0; i < (size - 1); ++i) {
    disps[i + 1] = disps[i] + rcounts[i];
  }
  count = disps[size - 1] + rcounts[size - 1];

  /* short cut the trivial case */
  if(0 == count) {
    free(disps);
    return MPI_SUCCESS;
  }

  /* get datatype information */
  MPI_Type_get_extent(dtype, &lb, &extent);
  MPI_Type_get_true_extent(dtype, &gap, &true_extent);

  // Calculate the total memory span
  buf_size = true_extent + extent * (count - 1);

  /* Handle MPI_IN_PLACE */
  if(MPI_IN_PLACE == sbuf) {
    sbuf = rbuf;
  }

  /* Allocate temporary receive buffer. */
  recv_buf_free = (char*) malloc(buf_size);
  recv_buf = recv_buf_free - gap;
  if(NULL == recv_buf_free) {
    err = MPI_ERR_NO_MEM;
    goto cleanup;
  }

  /* allocate temporary buffer for results */
  result_buf_free = (char*) malloc(buf_size);
  result_buf = result_buf_free - gap;

  /* copy local buffer into the temporary results */
  err = copy_buffer_different_dt(sbuf, count, dtype, result_buf, count, dtype);
  if(MPI_SUCCESS != err) goto cleanup;

  /* figure out power of two mapping: grow until larger than
     comm size, then go back one, to get the largest power of
     two less than comm size */
  tmp_size = next_poweroftwo (size);
  tmp_size >>= 1;
  remain = size - tmp_size;

  /* If comm size is not a power of two, have the first "remain"
     procs with an even rank send to rank + 1, leaving a power of
     two procs to do the rest of the algorithm */
  if(rank < 2 * remain) {
    if((rank & 1) == 0) {
      err = MPI_Send(result_buf, count, dtype, rank + 1, 0, comm);
      if(MPI_SUCCESS != err) goto cleanup;

      /* we don't participate from here on out */
      tmp_rank = -1;
    } else {
      err = MPI_Recv(recv_buf, count, dtype, rank - 1, 0, comm, MPI_STATUS_IGNORE);

      /* integrate their results into our temp results */
      MPI_Reduce_local(recv_buf, result_buf, count, dtype, op);

      /* adjust rank to be the bottom "remain" ranks */
      tmp_rank = rank / 2;
    }
  } else {
    /* just need to adjust rank to show that the bottom "even
       remain" ranks dropped out */
    tmp_rank = rank - remain;
  }

  /* For ranks not kicked out by the above code, perform the
     recursive halving */
  if(tmp_rank >= 0) {
    size_t *tmp_rcounts = NULL;
    ptrdiff_t *tmp_disps = NULL;
    int mask, send_index, recv_index, last_index;

    /* recalculate disps and rcounts to account for the
       special "remainder" processes that are no longer doing
       anything */
    tmp_rcounts = (size_t*) malloc(tmp_size * sizeof(size_t));
    if(NULL == tmp_rcounts) {
      err = MPI_ERR_NO_MEM;
      goto cleanup;
    }
    tmp_disps = (ptrdiff_t*) malloc(tmp_size * sizeof(ptrdiff_t));
    if(NULL == tmp_disps) {
      free(tmp_rcounts);
      err = MPI_ERR_NO_MEM;
      goto cleanup;
    }

    for(i = 0 ; i < tmp_size ; ++i) {
      if(i < remain) {
        /* need to include old neighbor as well */
        tmp_rcounts[i] = rcounts[i * 2 + 1] + rcounts[i * 2];
      } else {
        tmp_rcounts[i] = rcounts[i + remain];
      }
    }

    tmp_disps[0] = 0;
    for(i = 0; i < tmp_size - 1; ++i) {
      tmp_disps[i + 1] = tmp_disps[i] + tmp_rcounts[i];
    }

    /* do the recursive halving communication.  Don't use the
       dimension information on the communicator because I
       think the information is invalidated by our "shrinking"
       of the communicator */
    mask = tmp_size >> 1;
    send_index = recv_index = 0;
    last_index = tmp_size;
    while (mask > 0) {
      int tmp_peer, peer;
      size_t send_count, recv_count;
      MPI_Request request;

      tmp_peer = tmp_rank ^ mask;
      peer = (tmp_peer < remain) ? tmp_peer * 2 + 1 : tmp_peer + remain;

      /* figure out if we're sending, receiving, or both */
      send_count = recv_count = 0;
      if(tmp_rank < tmp_peer) {
        send_index = recv_index + mask;
        for(i = send_index ; i < last_index ; ++i) {
          send_count += tmp_rcounts[i];
        }
        for(i = recv_index ; i < send_index ; ++i) {
          recv_count += tmp_rcounts[i];
        }
      } else {
        recv_index = send_index + mask;
        for(i = send_index ; i < recv_index ; ++i) {
          send_count += tmp_rcounts[i];
        }
        for(i = recv_index ; i < last_index ; ++i) {
          recv_count += tmp_rcounts[i];
        }
      }

      /* actual data transfer.  Send from result_buf,
         receive into recv_buf */
      if(recv_count > 0) {
        err = MPI_Irecv(recv_buf + tmp_disps[recv_index] * extent,
                        recv_count, dtype, peer, 0, comm, &request);
        if(MPI_SUCCESS != err) {
          free(tmp_rcounts);
          free(tmp_disps);
          goto cleanup;
        }
      }
      if(send_count > 0) {
        err = MPI_Send(result_buf + tmp_disps[send_index] * extent,
                       send_count, dtype, peer, 0, comm);
        if(MPI_SUCCESS != err) {
          free(tmp_rcounts);
          free(tmp_disps);
          goto cleanup;
        }
      }

      /* if we received something on this step, push it into
         the results buffer */
      if(recv_count > 0) {
        err = MPI_Wait(&request, MPI_STATUS_IGNORE);
        if(MPI_SUCCESS != err) {
          free(tmp_rcounts);
          free(tmp_disps);
          goto cleanup;
        }

        MPI_Reduce_local(recv_buf + tmp_disps[recv_index] * extent,
                         result_buf + tmp_disps[recv_index] * extent,
                         recv_count, dtype, op);
      }

      /* update for next iteration */
      send_index = recv_index;
      last_index = recv_index + mask;
      mask >>= 1;
    }

    /* copy local results from results buffer into real receive buffer */
    if(0 != rcounts[rank]) {
      err = copy_buffer_different_dt(result_buf + disps[rank] * extent, rcounts[rank],
                                     dtype, rbuf, rcounts[rank], dtype);
      if(MPI_SUCCESS != err) {
        free(tmp_rcounts);
        free(tmp_disps);
        goto cleanup;
      }
    }

    free(tmp_rcounts);
    free(tmp_disps);
  }

  /* Now fix up the non-power of two case, by having the odd
     procs send the even procs the proper results */
  if(rank < (2 * remain)) {
    if((rank & 1) == 0) {
      if(rcounts[rank]) {
        err = MPI_Recv(rbuf, rcounts[rank], dtype, rank + 1, 0, comm, MPI_STATUS_IGNORE);
        if(MPI_SUCCESS != err) goto cleanup;
      }
    } else {
      if(rcounts[rank - 1]) {
        err = MPI_Send(result_buf + disps[rank - 1] * extent,
                       rcounts[rank - 1], dtype, rank - 1, 0, comm);
        if(MPI_SUCCESS != err) goto cleanup;
      }
    }
  }

 cleanup:
  if(NULL != disps) free(disps);
  if(NULL != recv_buf_free) free(recv_buf_free);
  if(NULL != result_buf_free) free(result_buf_free);

  return err;
}

int reduce_scatter_recursive_distance_doubling(const void *sbuf, void *rbuf, const int rcounts[],
                                               MPI_Datatype dtype, MPI_Op op, MPI_Comm comm)
{
  int i, rank, size, step, steps, err = MPI_SUCCESS;
  size_t count;
  ptrdiff_t *disps = NULL;
  ptrdiff_t extent, true_extent, lb, buf_size, gap = 0;
  char *recv_buf = NULL, *recv_buf_free = NULL;
  char *result_buf = NULL, *result_buf_free = NULL;


  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  steps = log_2(size);
  if( !is_power_of_two(size) || steps == -1 ) {
    return MPI_ERR_ARG;
  }

  /* Find displacements and the like */
  disps = (ptrdiff_t*) malloc(sizeof(ptrdiff_t) * size);
  if(NULL == disps) return MPI_ERR_NO_MEM;

  disps[0] = 0;
  for(i = 0; i < (size - 1); ++i) {
    disps[i + 1] = disps[i] + rcounts[i];
  }
  count = (size_t) disps[size - 1] + (size_t) rcounts[size - 1];

  /* short cut the trivial case */
  if(0 == count) {
    free(disps);
    return MPI_SUCCESS;
  }

  /* Handle MPI_IN_PLACE */
  if(MPI_IN_PLACE == sbuf) {
    sbuf = rbuf;
  }

  /* get datatype information */
  MPI_Type_get_extent(dtype, &lb, &extent);
  MPI_Type_get_true_extent(dtype, &gap, &true_extent);

  // Calculate the total memory span
  buf_size = true_extent + extent * (count - 1);

  /* Allocate temporary receive buffer. */
  recv_buf_free = (char*) malloc(buf_size);
  recv_buf = recv_buf_free - gap;

  /* allocate temporary buffer for results */
  result_buf_free = (char*) malloc(buf_size);
  result_buf = result_buf_free - gap;
  
  if(NULL == recv_buf_free || NULL == result_buf_free) {
    err = MPI_ERR_NO_MEM;
    goto cleanup;
  }

  /* copy local buffer into the temporary results */
  err = copy_buffer_different_dt(sbuf, count, dtype, result_buf, count, dtype);
  if(MPI_SUCCESS != err) goto cleanup;


  /* do the recursive distance doubling. */
  int w_size = size >> 1;
  int dist_mask = 0x1,last_index = size;
  int send_index = 0, recv_index = 0;
  for (step = 0; step < steps; step++) {
    int peer;
    size_t send_count, recv_count;
    MPI_Request request;

    peer = rank ^ dist_mask;

    /* figure out if we're sending, receiving, or both */
    send_count = recv_count = 0;
    if(rank < peer) {
      send_index = recv_index + w_size;
      for (i = send_index; i < last_index; ++i) {
        send_count += rcounts[i];
      }
      for (i = recv_index; i < send_index; ++i) {
        recv_count += rcounts[i];
      }
    } else {
      recv_index = send_index + w_size;
      for (i = send_index; i < recv_index; ++i) {
      send_count += rcounts[i];
      }
      for (i = recv_index; i < last_index; ++i) {
      recv_count += rcounts[i];
      }
    }

    /* actual data transfer.  Send from result_buf,
        receive into recv_buf */
    if(recv_count > 0) {
      err = MPI_Irecv(recv_buf + disps[recv_index] * extent,
                      recv_count, dtype, peer, 0, comm, &request);
      if(MPI_SUCCESS != err) { goto cleanup; }
    }
    if(send_count > 0) {
      err = MPI_Send(result_buf + disps[send_index] * extent,
                      send_count, dtype, peer, 0, comm);
      if(MPI_SUCCESS != err) { goto cleanup; }
    }

    /* if we received something on this step, push it into
        the results buffer */
    if(recv_count > 0) {
      err = MPI_Wait(&request, MPI_STATUS_IGNORE);
      if(MPI_SUCCESS != err) { goto cleanup; }

      MPI_Reduce_local(recv_buf + disps[recv_index] * extent,
                        result_buf + disps[recv_index] * extent,
                        recv_count, dtype, op);
    }


    /* update for next iteration */
    send_index = recv_index;
    last_index = recv_index + w_size;
    w_size >>= 1;
    dist_mask <<= 1;
  }

  // /* copy local results from results buffer into real receive buffer */
  // if(0 != rcounts[rank]) {
  //   err = copy_buffer_different_dt(result_buf + disps[(int) inverse_rank(size, rank)] * extent, rcounts[rank],
  //                                   dtype, rbuf, rcounts[rank], dtype);
  //   if(MPI_SUCCESS != err) { goto cleanup; }
  // }

  int inverse = inverse_rank(size, rank);
  if(rank != inverse) {
    /* send result to correct rank's recv buffer */
    err = MPI_Sendrecv(result_buf + disps[inverse] * extent, rcounts[inverse], dtype, inverse, 0,
                       rbuf, rcounts[rank], dtype, inverse, 0,
                       comm, MPI_STATUS_IGNORE);
    if(MPI_SUCCESS != err) {
      goto cleanup;
    }
  } else {
    /* copy local results from results buffer into real receive buffer */
    err = copy_buffer_different_dt(result_buf + disps[rank] * extent, rcounts[rank],
                                    dtype, rbuf, rcounts[rank], dtype);
    if(MPI_SUCCESS != err) {
      goto cleanup;
    }
  }


cleanup:
  if(NULL != disps) free(disps);
  if(NULL != recv_buf_free) free(recv_buf_free);
  if(NULL != result_buf_free) free(result_buf_free);

  return err;
}

int reduce_scatter_ring( const void *sbuf, void *rbuf, const int rcounts[],
                        MPI_Datatype dtype, MPI_Op op, MPI_Comm comm)
{
  int ret, line, rank, size, i, k, recv_from, send_to;
  int inbi;
  size_t total_count, max_block_count;
  ptrdiff_t *displs = NULL;
  char *tmpsend = NULL, *tmprecv = NULL, *accumbuf = NULL, *accumbuf_free = NULL;
  char *inbuf_free[2] = {NULL, NULL}, *inbuf[2] = {NULL, NULL};
  ptrdiff_t extent, lb, max_real_segsize, dsize, gap = 0;
  MPI_Request reqs[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};

  ret = MPI_Comm_size(comm, &size);
  ret = MPI_Comm_rank(comm, &rank);

  /* Determine the maximum number of elements per node,
     corresponding block size, and displacements array.
  */
  displs = (ptrdiff_t*) malloc(size * sizeof(ptrdiff_t));
  if(NULL == displs) { ret = -1; line = __LINE__; goto error_hndl; }
  displs[0] = 0;
  total_count = rcounts[0];
  max_block_count = rcounts[0];
  for(i = 1; i < size; i++) {
    displs[i] = total_count;
    total_count += rcounts[i];
    if(max_block_count < rcounts[i]) max_block_count = rcounts[i];
  }

  /* Special case for size == 1 */
  if(1 == size) {
    if(MPI_IN_PLACE != sbuf) {
      ret = copy_buffer((char*) sbuf, (char*) rbuf,total_count, dtype);
      if(ret < 0) { line = __LINE__; goto error_hndl; }
    }
    free(displs);
    return MPI_SUCCESS;
  }

  /* Allocate and initialize temporary buffers, we need:
     - a temporary buffer to perform reduction (size total_count) since
     rbuf can be of rcounts[rank] size.
     - up to two temporary buffers used for communication/computation overlap.
  */
  ret = MPI_Type_get_extent(dtype, &lb, &extent);
  if(MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }

  max_real_segsize = datatype_span(dtype, max_block_count, &gap);
  dsize = datatype_span(dtype, total_count, &gap);

  accumbuf_free = (char*)malloc(dsize);
  if(NULL == accumbuf_free) { ret = -1; line = __LINE__; goto error_hndl; }
  accumbuf = accumbuf_free - gap;

  inbuf_free[0] = (char*)malloc(max_real_segsize);
  if(NULL == inbuf_free[0]) { ret = -1; line = __LINE__; goto error_hndl; }
  inbuf[0] = inbuf_free[0] - gap;
  if(size > 2) {
    inbuf_free[1] = (char*)malloc(max_real_segsize);
    if(NULL == inbuf_free[1]) { ret = -1; line = __LINE__; goto error_hndl; }
    inbuf[1] = inbuf_free[1] - gap;
  }

  /* Handle MPI_IN_PLACE for size > 1 */
  if(MPI_IN_PLACE == sbuf) {
    sbuf = rbuf;
  }

  ret = copy_buffer((char*) sbuf, accumbuf, total_count, dtype);
  if(ret < 0) { line = __LINE__; goto error_hndl; }

  /* Computation loop */

  /*
     For each of the remote nodes:
     - post irecv for block (r-2) from (r-1) with wrap around
     - send block (r-1) to (r+1)
     - in loop for every step k = 2 .. n
     - post irecv for block (r - 1 + n - k) % n
     - wait on block (r + n - k) % n to arrive
     - compute on block (r + n - k ) % n
     - send block (r + n - k) % n
     - wait on block (r)
     - compute on block (r)
     - copy block (r) to rbuf
     Note that we must be careful when computing the beginning of buffers and
     for send operations and computation we must compute the exact block size.
  */
  send_to = (rank + 1) % size;
  recv_from = (rank + size - 1) % size;

  inbi = 0;
  /* Initialize first receive from the neighbor on the left */
  ret = MPI_Irecv(inbuf[inbi], max_block_count, dtype, recv_from, 0, comm, &reqs[inbi]);
  if(MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
  tmpsend = accumbuf + displs[recv_from] * extent;
  ret = MPI_Send(tmpsend, rcounts[recv_from], dtype, send_to, 0, comm);
  if(MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }

  for(k = 2; k < size; k++) {
    const int prevblock = (rank + size - k) % size;

    inbi = inbi ^ 0x1;

    /* Post irecv for the current block */
    ret = MPI_Irecv(inbuf[inbi], max_block_count, dtype, recv_from, 0, comm, &reqs[inbi]);
    if(MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }

    /* Wait on previous block to arrive */
    ret = MPI_Wait(&reqs[inbi ^ 0x1], MPI_STATUS_IGNORE);
    if(MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }

    /* Apply operation on previous block: result goes to rbuf
       rbuf[prevblock] = inbuf[inbi ^ 0x1] (op) rbuf[prevblock]
    */
    tmprecv = accumbuf + displs[prevblock] * extent;
    MPI_Reduce_local(inbuf[inbi ^ 0x1], tmprecv, rcounts[prevblock], dtype, op);

    /* send previous block to send_to */
    ret = MPI_Send(tmprecv, rcounts[prevblock], dtype, send_to, 0, comm);
    if(MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
  }

  /* Wait on the last block to arrive */
  ret = MPI_Wait(&reqs[inbi], MPI_STATUS_IGNORE);
  if(MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }

  /* Apply operation on the last block (my block)
     rbuf[rank] = inbuf[inbi] (op) rbuf[rank] */
  tmprecv = accumbuf + displs[rank] * extent;
  MPI_Reduce_local(inbuf[inbi], tmprecv, rcounts[rank], dtype, op);

  /* Copy result from tmprecv to rbuf */
  ret = copy_buffer(tmprecv, (char *)rbuf, rcounts[rank], dtype);
  if(ret < 0) { line = __LINE__; goto error_hndl; }

  if(NULL != displs) free(displs);
  if(NULL != accumbuf_free) free(accumbuf_free);
  if(NULL != inbuf_free[0]) free(inbuf_free[0]);
  if(NULL != inbuf_free[1]) free(inbuf_free[1]);

  return MPI_SUCCESS;

 error_hndl:
  BINE_DEBUG_PRINT("\n%s:%4d\tRank %d Error occurred %d\n\n", __FILE__, line, rank, ret);
  (void)line;  // silence compiler warning
  if(NULL != displs) free(displs);
  if(NULL != accumbuf_free) free(accumbuf_free);
  if(NULL != inbuf_free[0]) free(inbuf_free[0]);
  if(NULL != inbuf_free[1]) free(inbuf_free[1]);
  return ret;
}


int reduce_scatter_butterfly(const void *sbuf, void *rbuf, const int rcounts[],
                             MPI_Datatype dtype, MPI_Op op, MPI_Comm comm)
{
  char *tmpbuf[2] = {NULL, NULL}, *psend, *precv;
  ptrdiff_t *displs = NULL, index;
  ptrdiff_t span, gap, totalcount, extent, lb;
  int rank, comm_size, err = MPI_SUCCESS;
  err = MPI_Comm_size(comm, &comm_size);
  err = MPI_Comm_rank(comm, &rank);

  if(comm_size < 2)
    return MPI_SUCCESS;

  displs = malloc(sizeof(*displs) * comm_size);
  if(NULL == displs) {
    err = MPI_ERR_NO_MEM;
    goto cleanup_and_return;
  }
  displs[0] = 0;
  for(int i = 1; i < comm_size; i++) {
    displs[i] = displs[i - 1] + rcounts[i - 1];
  }
  totalcount = displs[comm_size - 1] + rcounts[comm_size - 1];

  MPI_Type_get_extent(dtype, &lb, &extent);
  span = datatype_span(dtype, totalcount, &gap);
  tmpbuf[0] = malloc(span);
  tmpbuf[1] = malloc(span);
  if(NULL == tmpbuf[0] || NULL == tmpbuf[1]) {
    err = MPI_ERR_NO_MEM;
    goto cleanup_and_return;
  }
  psend = tmpbuf[0] - gap;
  precv = tmpbuf[1] - gap;

  if(sbuf != MPI_IN_PLACE) {
    err = copy_buffer((char *) sbuf, psend, totalcount, dtype);
    if(MPI_SUCCESS != err) { goto cleanup_and_return; }
  } else {
    err = copy_buffer(rbuf, psend, totalcount, dtype);
    if(MPI_SUCCESS != err) { goto cleanup_and_return; }
  }

  /*
   * Step 1. Reduce the number of processes to the nearest lower power of two
   * p' = 2^{\floor{\log_2 p}} by removing r = p - p' processes.
   * In the first 2r processes (ranks 0 to 2r - 1), all the even ranks send
   * the input vector to their neighbor (rank + 1) and all the odd ranks recv
   * the input vector and perform local reduction.
   * The odd ranks (0 to 2r - 1) contain the reduction with the input
   * vector on their neighbors (the even ranks). The first r odd
   * processes and the p - 2r last processes are renumbered from
   * 0 to 2^{\floor{\log_2 p}} - 1. Even ranks do not participate in the
   * rest of the algorithm.
   */

  /* Find nearest power-of-two less than or equal to comm_size */
  int nprocs_pof2 = next_poweroftwo(comm_size);
  nprocs_pof2 >>= 1;
  int nprocs_rem = comm_size - nprocs_pof2;
  int log2_size = log_2(nprocs_pof2);

  int vrank = -1;
  if(rank < 2 * nprocs_rem) {
    if((rank % 2) == 0) {
      /* Even process */
      err = MPI_Send(psend, totalcount, dtype, rank + 1, 0, comm);
      if(MPI_SUCCESS != err) { goto cleanup_and_return; }
      /* This process does not participate in the rest of the algorithm */
      vrank = -1;
    } else {
      /* Odd process */
      err = MPI_Recv(precv, totalcount, dtype, rank - 1, 0, comm, MPI_STATUS_IGNORE);
      if(MPI_SUCCESS != err) { goto cleanup_and_return; }
      MPI_Reduce_local(precv, psend, totalcount, dtype, op);
      /* Adjust rank to be the bottom "remain" ranks */
      vrank = rank / 2;
    }
  } else {
    /* Adjust rank to show that the bottom "even remain" ranks dropped out */
    vrank = rank - nprocs_rem;
  }

  if(vrank != -1) {
    /*
     * Now, psend vector of size totalcount is divided into nprocs_pof2 blocks:
     * block 0:   rcounts[0] and rcounts[1] -- for process 0 and 1
     * block 1:   rcounts[2] and rcounts[3] -- for process 2 and 3
     * ...
     * block r-1: rcounts[2*(r-1)] and rcounts[2*(r-1)+1]
     * block r:   rcounts[r+r]
     * block r+1: rcounts[r+r+1]
     * ...
     * block nprocs_pof2 - 1: rcounts[r+nprocs_pof2-1]
     */
    int nblocks = nprocs_pof2, send_index = 0, recv_index = 0;
    for(int mask = 1; mask < nprocs_pof2; mask <<= 1) {
      int vpeer = vrank ^ mask;
      int peer = (vpeer < nprocs_rem) ? vpeer * 2 + 1 : vpeer + nprocs_rem;

      nblocks /= 2;
      if((vrank & mask) == 0) {
        /* Send the upper half of reduction buffer, recv the lower half */
        send_index += nblocks;
      } else {
        /* Send the upper half of reduction buffer, recv the lower half */
        recv_index += nblocks;
      }

      /* Send blocks: [send_index, send_index + nblocks - 1] */
      size_t send_count = sum_counts(rcounts, displs, nprocs_rem,
                                     send_index, send_index + nblocks - 1);
      index = (send_index < nprocs_rem) ? 2 * send_index : nprocs_rem + send_index;
      ptrdiff_t sdispl = displs[index];

      /* Recv blocks: [recv_index, recv_index + nblocks - 1] */
      size_t recv_count = sum_counts(rcounts, displs, nprocs_rem,
                                          recv_index, recv_index + nblocks - 1);
      index = (recv_index < nprocs_rem) ? 2 * recv_index : nprocs_rem + recv_index;
      ptrdiff_t rdispl = displs[index];

      err = MPI_Sendrecv(psend + sdispl * extent, send_count, dtype, peer, 0,
                         precv + rdispl * extent, recv_count, dtype, peer, 0,
                         comm, MPI_STATUS_IGNORE);
      if(MPI_SUCCESS != err) { goto cleanup_and_return; }

      if(vrank < vpeer) {
        /* precv = psend <op> precv */
        MPI_Reduce_local(psend + rdispl * extent, precv + rdispl * extent,
                         recv_count, dtype, op);
        char *p = psend;
        psend = precv;
        precv = p;
      } else {
        /* psend = precv <op> psend */
        MPI_Reduce_local(precv + rdispl * extent, psend + rdispl * extent,
                         recv_count, dtype, op);
      }
      send_index = recv_index;
    }
    /*
     * psend points to the result block [send_index]
     * Exchange results with remote process according to a mirror permutation.
     */
    int vpeer = mirror_perm(vrank, log2_size);
    int peer = (vpeer < nprocs_rem) ? vpeer * 2 + 1 : vpeer + nprocs_rem;
    index = (send_index < nprocs_rem) ? 2 * send_index : nprocs_rem + send_index;

    if(vpeer < nprocs_rem) {
      /*
       * Process has two blocks: for excluded process and own.
       * Send the first block to excluded process.
       */
      err = MPI_Send(psend + displs[index] * extent, rcounts[index], dtype,
                     peer - 1, 0, comm);
      if(MPI_SUCCESS != err) { goto cleanup_and_return; }
    }

    /* If process has two blocks, then send the second block (own block) */
    if(vpeer < nprocs_rem)
      index++;
    if(vpeer != vrank) {
      err = MPI_Sendrecv(psend + displs[index] * extent, rcounts[index], dtype, peer, 0,
                         rbuf, rcounts[rank], dtype, peer, 0, comm, MPI_STATUS_IGNORE);
      if(MPI_SUCCESS != err) { goto cleanup_and_return; }
    } else {
      err = copy_buffer(psend + displs[rank] * extent, rbuf, rcounts[rank], dtype);
      if(MPI_SUCCESS != err) { goto cleanup_and_return; }
    }

  } else {
    /* Excluded process: receive result */
    int vpeer = mirror_perm((rank + 1) / 2, log2_size);
    int peer = (vpeer < nprocs_rem) ? vpeer * 2 + 1 : vpeer + nprocs_rem;
    err = MPI_Recv(rbuf, rcounts[rank], dtype, peer, 0, comm, MPI_STATUS_IGNORE);
    if(MPI_SUCCESS != err) { goto cleanup_and_return; }
   }

cleanup_and_return:
  if(displs)
    free(displs);
  if(tmpbuf[0])
    free(tmpbuf[0]);
  if(tmpbuf[1])
    free(tmpbuf[1]);
  return err;
}

int reduce_scatter_bine_send_remap_hierarchical(const void *sendbuf, void *recvbuf, const int recvcounts[],
                                                   MPI_Datatype dt, MPI_Op op, MPI_Comm comm)
{
  int size, rank, dtsize, err = MPI_SUCCESS, partner;
  int node_size, node_rank, node_offset, local_rank;
  int recv_count, send_count;
  int task_on_node = pico_task_on_node();
  MPI_Request send_reqs[task_on_node], recv_reqs[task_on_node];
  int send_reqc, recv_reqc;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  MPI_Type_size(dt, &dtsize);

  int count = 0;
  int *displs = (int *)malloc(size * sizeof(int));
  int *step_to_send = (int *)malloc(size * sizeof(int));
  for (int i = 0; i < size; i++)
  {
    displs[i] = count;
    count += recvcounts[i];
  }

  pico_get_group_config(&node_size, &node_rank, &node_offset, &local_rank, task_on_node, size, rank);

  void *tmpbuf = NULL, *resbuf = NULL;
  int upper_index, lower_index;
  size_t buffer_size_unit;
  lower_index = local_rank * node_size;
  upper_index = lower_index + (node_size - 1);
  buffer_size_unit = displs[upper_index] - displs[lower_index] + recvcounts[upper_index];
  const char *src_location = sendbuf + (ptrdiff_t)displs[lower_index] * (ptrdiff_t)dtsize;

#ifdef PICO_MPI_CUDA_AWARE
  BINE_CUDA_CHECK(cudaMalloc((void **)&tmpbuf, buffer_size_unit * (task_on_node - 1) * dtsize));
  BINE_CUDA_CHECK(cudaMalloc((void **)&resbuf, buffer_size_unit * dtsize));
#else
  tmpbuf = malloc(buffer_size_unit * (task_on_node - 1) * dtsize);
  resbuf = malloc(buffer_size_unit * dtsize);
#endif
  if (NULL == displs || NULL == step_to_send || NULL == tmpbuf || NULL == resbuf)
  {
    err = MPI_ERR_NO_MEM;
    goto err_hndl;
  }

#ifndef PICO_MPI_CUDA_AWARE
  memcpy(resbuf, src_location, buffer_size_unit * dtsize);
#endif

  send_reqc = recv_reqc = 0;
  recv_count = buffer_size_unit;
  for (int i = 0; i < task_on_node; i++)
  {
    partner = node_offset + i;
    if (partner == rank)
      continue;

    lower_index = i * node_size;
    upper_index = lower_index + (node_size - 1);
    send_count = displs[upper_index] - displs[lower_index] + recvcounts[upper_index];
    if (send_count > 0)
    {
      err = MPI_Isend(sendbuf + (ptrdiff_t)displs[lower_index] * (ptrdiff_t)dtsize, send_count, dt, partner, 0, comm, &send_reqs[send_reqc]);
      if (MPI_SUCCESS != err)
      {
        goto err_hndl;
      }
      send_reqc++;
    }
    if (recv_count > 0)
    {
      err = MPI_Irecv(tmpbuf + (ptrdiff_t)recv_reqc * (ptrdiff_t)recv_count * (ptrdiff_t)dtsize, recv_count, dt, partner, 0, comm, &recv_reqs[recv_reqc]);
      if (MPI_SUCCESS != err)
      {
        goto err_hndl;
      }
      recv_reqc++;
    }
  }
  err = MPI_Waitall(recv_reqc, recv_reqs, MPI_STATUSES_IGNORE);
  if (MPI_SUCCESS != err)
  {
    goto err_hndl;
  }

#ifdef PICO_MPI_CUDA_AWARE
  err = reduce_wrapper_grops_inoutsplit(tmpbuf, resbuf, src_location, recv_count, task_on_node - 1, dt, op);
  if (MPI_SUCCESS != err)
  {
    goto err_hndl;
  }
  BINE_CUDA_CHECK(cudaDeviceSynchronize());
#else
  for (int i = 0; i < recv_reqc; i++)
  {
    err = MPI_Reduce_local(tmpbuf + (ptrdiff_t)i * (ptrdiff_t)recv_count * (ptrdiff_t)dtsize, resbuf, recv_count, dt, op);
    if (MPI_SUCCESS != err)
    {
      goto err_hndl;
    }
  }
#endif
  /*COPY_BUFF_DIFF_DT(resbuf, recvcounts[rank], dt, recvbuf, recvcounts[rank], dt);
  return MPI_SUCCESS; */

  // determinate the firs global rank that has been reduced 
  int res_first_node = local_rank * node_size;
  int mask = 0x1;
  int inverse_mask = 0x1 << (int)(log_2(node_size) - 1);
  int block_first_mask = ~(inverse_mask - 1);
  int remapped_rank = remap_rank(node_size, node_rank);
  int send_block_first, send_block_last, send_block_first_rem, send_block_last_rem;
  int recv_block_first = 0, recv_block_last, recv_block_first_rem = 0, recv_block_last_rem;
  char *reduction_addres = NULL;
  while (mask < node_size)
  {
    if (node_rank % 2 == 0)
    {
      partner = mod(node_rank + negabinary_to_binary((mask << 1) - 1), node_size);
    }
    else
    {
      partner = mod(node_rank - negabinary_to_binary((mask << 1) - 1), node_size);
    }

    // For sure I need to send my (remapped) partner's data
    // the actual start block however must be aligned to
    // the power of two
    send_block_first = remap_rank(node_size, partner) & block_first_mask;
    send_block_last = send_block_first + inverse_mask - 1;
    send_block_first_rem = send_block_first + res_first_node;
    send_block_last_rem = send_block_last + res_first_node;
    send_count = displs[send_block_last_rem] - displs[send_block_first_rem] + recvcounts[send_block_last_rem];
    // Something similar for the block to recv.
    // I receive my block, but aligned to the power of two
    recv_block_first = remapped_rank & block_first_mask;
    recv_block_last = recv_block_first + inverse_mask - 1;
    recv_block_first_rem = recv_block_first + res_first_node;
    recv_block_last_rem = recv_block_last + res_first_node;
    recv_count = displs[recv_block_last_rem] - displs[recv_block_first_rem] + recvcounts[recv_block_last_rem];
    err = MPI_Sendrecv((char *)resbuf + (displs[send_block_first_rem] - displs[res_first_node]) * dtsize, send_count, dt, partner * task_on_node + local_rank, 0,
                       (char *)tmpbuf, recv_count, dt, partner * task_on_node + local_rank, 0, comm, MPI_STATUS_IGNORE);
    if (MPI_SUCCESS != err)
    {
      goto err_hndl;
    }
    reduction_addres = (char *)resbuf + (displs[recv_block_first_rem] - displs[res_first_node]) * dtsize;

#ifdef PICO_MPI_CUDA_AWARE
    err = reduce_wrapper((char *)tmpbuf, reduction_addres, recv_count, dt, op);
    if (MPI_SUCCESS != err)
    {
      goto err_hndl;
    }
    BINE_CUDA_CHECK(cudaDeviceSynchronize());
#else
    err = MPI_Reduce_local((char *)tmpbuf, reduction_addres, recv_count, dt, op);
    if (MPI_SUCCESS != err)
    {
      goto err_hndl;
    }
#endif

    mask <<= 1;
    inverse_mask >>= 1;
    block_first_mask >>= 1;
  }

  // Final send
  // Whom I have been remapped to? I.e., who is going to send me my data? Just do a recv from any
  if (recv_block_first_rem != rank && node_size > 1)
  {
    MPI_Status status;
    MPI_Sendrecv(reduction_addres, recvcounts[recv_block_first_rem], dt, recv_block_first_rem, 0,
                 (char *)recvbuf, recvcounts[rank], dt, MPI_ANY_SOURCE, 0, comm, &status);
  }
  else
  {
    err = COPY_BUFF_DIFF_DT(reduction_addres, recvcounts[rank], dt, recvbuf, recvcounts[rank], dt);
    if (MPI_SUCCESS != err)
    {
      goto err_hndl;
    }
  }

  err = MPI_Waitall(send_reqc, send_reqs, MPI_STATUSES_IGNORE);
  if (MPI_SUCCESS != err)
  {
    goto err_hndl;
  }

#ifdef PICO_MPI_CUDA_AWARE
  BINE_CUDA_CHECK(cudaFree(tmpbuf));
  BINE_CUDA_CHECK(cudaFree(resbuf));
#else
  free(tmpbuf);
  free(resbuf);
#endif
  free(displs);
  free(step_to_send);
  return MPI_SUCCESS;

err_hndl:
  if (NULL != displs)
    free(displs);
  if (NULL != step_to_send)
    free(step_to_send);
  if (NULL != tmpbuf)
#ifdef PICO_MPI_CUDA_AWARE
    BINE_CUDA_CHECK(cudaFree(tmpbuf));
#else
    free(tmpbuf);
#endif
  if (NULL != resbuf)
#ifdef PICO_MPI_CUDA_AWARE
    BINE_CUDA_CHECK(cudaFree(resbuf));
#else
    free(resbuf);
#endif
  return err;
}


int reduce_scatter_bine_send_remap(const void *sendbuf, void *recvbuf, const int recvcounts[],
                                   MPI_Datatype dt, MPI_Op op, MPI_Comm comm)
{
  int size, rank, dtsize, err = MPI_SUCCESS;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  MPI_Type_size(dt, &dtsize);

  int count = 0;
  int *displs = (int *)malloc(size * sizeof(int));
  int *step_to_send = (int *)malloc(size * sizeof(int));
  for (int i = 0; i < size; i++)
  {
    displs[i] = count;
    count += recvcounts[i];
  }

  void *tmpbuf = NULL, *resbuf = NULL;
#ifdef PICO_MPI_CUDA_AWARE
  BINE_CUDA_CHECK(cudaMalloc((void **)&tmpbuf, count * dtsize));
  BINE_CUDA_CHECK(cudaMalloc((void **)&resbuf, count * dtsize));
#else
  tmpbuf = malloc(count * dtsize);
  resbuf = malloc(count * dtsize);
#endif
  if (NULL == displs || NULL == step_to_send || NULL == tmpbuf || NULL == resbuf)
  {
    err = MPI_ERR_NO_MEM;
    goto err_hndl;
  }
  COPY_BUFF_DIFF_DT(sendbuf, count, dt, resbuf, count, dt);
  // memcpy(resbuf, sendbuf, count * dtsize);

  int mask = 0x1;
  int inverse_mask = 0x1 << (int)(log_2(size) - 1);
  int block_first_mask = ~(inverse_mask - 1);
  int remapped_rank = remap_rank(size, rank);
  while (mask < size)
  {
    int partner;
    if (rank % 2 == 0)
    {
      partner = mod(rank + negabinary_to_binary((mask << 1) - 1), size);
    }
    else
    {
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
    err = MPI_Sendrecv((char *)resbuf + displs[send_block_first] * dtsize, send_count, dt, partner, 0,
                       (char *)tmpbuf + displs[recv_block_first] * dtsize, recv_count, dt, partner, 0, comm, MPI_STATUS_IGNORE);
    if (MPI_SUCCESS != err)
    {
      goto err_hndl;
    }
#ifdef PICO_MPI_CUDA_AWARE
    reduce_wrapper((char *)tmpbuf + displs[recv_block_first] * dtsize, (char *)resbuf + displs[recv_block_first] * dtsize, recv_count, dt, op);
    BINE_CUDA_CHECK(cudaDeviceSynchronize());
#else
    err = MPI_Reduce_local((char *)tmpbuf + displs[recv_block_first] * dtsize, (char *)resbuf + displs[recv_block_first] * dtsize, recv_count, dt, op);
    if (MPI_SUCCESS != err)
    {
      goto err_hndl;
    }
#endif

    mask <<= 1;
    inverse_mask >>= 1;
    block_first_mask >>= 1;
  }

  // Final send
  // Whom I have been remapped to? I.e., who is going to send me my data? Just do a recv from any
  /*
  reciv = rmap(local_rank) * gpu_on_node + node_rank
  */
  MPI_Status status;
  MPI_Sendrecv((char *)resbuf + displs[remapped_rank] * dtsize, recvcounts[remapped_rank], dt, remapped_rank, 0,
               (char *)recvbuf, recvcounts[rank], dt, MPI_ANY_SOURCE, 0,
               comm, &status);

#ifdef PICO_MPI_CUDA_AWARE
  BINE_CUDA_CHECK(cudaFree(tmpbuf));
  BINE_CUDA_CHECK(cudaFree(resbuf));
#else
  free(tmpbuf);
  free(resbuf);
#endif
  free(displs);
  free(step_to_send);
  return MPI_SUCCESS;

err_hndl:
  if (NULL != displs)
    free(displs);
  if (NULL != step_to_send)
    free(step_to_send);
  if (NULL != tmpbuf)
#ifdef PICO_MPI_CUDA_AWARE
    BINE_CUDA_CHECK(cudaFree(tmpbuf));
#else
    free(tmpbuf);
#endif
  if (NULL != resbuf)
#ifdef PICO_MPI_CUDA_AWARE
    BINE_CUDA_CHECK(cudaFree(resbuf));
#else
    free(resbuf);
#endif
  return err;
}

int reduce_scatter_bine_permute_remap(const void *sendbuf, void *recvbuf, const int recvcounts[],
                                      MPI_Datatype dt, MPI_Op op, MPI_Comm comm)
{
  int size, rank, dtsize, err = MPI_SUCCESS;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  MPI_Type_size(dt, &dtsize);
  int count = 0;
  int *displs = (int *)malloc(size * sizeof(int));
  int *step_to_send = (int *)malloc(size * sizeof(int));
  for (int i = 0; i < size; i++)
  {
    displs[i] = count;
    count += recvcounts[i];
  }

  void *tmpbuf, *resbuf;
#ifdef PICO_MPI_CUDA_AWARE
  BINE_CUDA_CHECK(cudaMalloc(&tmpbuf, count * dtsize));
  BINE_CUDA_CHECK(cudaMalloc(&resbuf, count * dtsize));
#else
  tmpbuf = malloc(count * dtsize);
  resbuf = malloc(count * dtsize);
#endif
  if (NULL == displs || NULL == step_to_send || NULL == tmpbuf || NULL == resbuf)
  {
    err = MPI_ERR_NO_MEM;
    goto err_hndl;
  }

  // Permute memcpy
  for (int i = 0; i < size; i++)
  {
    int remapped_rank = remap_rank(size, i);
    /*if (rank == 0)
    {
      printf("pos %d remaped to %d\n", get_sender_rec(size, i), i);
      fflush(stdout);
    }*/
    COPY_BUFF_DIFF_DT((char *)sendbuf + displs[i] * dtsize, recvcounts[i], dt, (char *)resbuf + displs[remapped_rank] * dtsize, recvcounts[i], dt);
    // memcpy((char *)resbuf + displs[remapped_rank] * dtsize, (char *)sendbuf + displs[i] * dtsize, recvcounts[i] * dtsize);
  }

  int mask = 0x1;
  int inverse_mask = 0x1 << (int)(log_2(size) - 1);
  int block_first_mask = ~(inverse_mask - 1);
  int remapped_rank = remap_rank(size, rank);
  while (mask < size)
  {
    int partner;
    if (rank % 2 == 0)
    {
      partner = mod(rank + negabinary_to_binary((mask << 1) - 1), size);
    }
    else
    {
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

    err = MPI_Sendrecv((char *)resbuf + displs[send_block_first] * dtsize, send_count, dt, partner, 0,
                       (char *)tmpbuf + displs[recv_block_first] * dtsize, recv_count, dt, partner, 0, comm, MPI_STATUS_IGNORE);
    if (MPI_SUCCESS != err)
    {
      goto err_hndl;
    }
#ifdef PICO_MPI_CUDA_AWARE
    reduce_wrapper((char *)tmpbuf + displs[recv_block_first] * dtsize, (char *)resbuf + displs[recv_block_first] * dtsize, recv_count, dt, op);
    BINE_CUDA_CHECK(cudaDeviceSynchronize());
#else
    err = MPI_Reduce_local((char *)tmpbuf + displs[recv_block_first] * dtsize, (char *)resbuf + displs[recv_block_first] * dtsize, recv_count, dt, op);
    if (MPI_SUCCESS != err)
    {
      goto err_hndl;
    }
#endif

    mask <<= 1;
    inverse_mask >>= 1;
    block_first_mask >>= 1;
  }

  // Final memcpy
  COPY_BUFF_DIFF_DT((char *)resbuf + displs[remapped_rank] * dtsize, recvcounts[rank], dt, recvbuf, recvcounts[rank], dt);
  // memcpy(recvbuf, (char *)resbuf + displs[remapped_rank] * dtsize, recvcounts[rank] * dtsize);

#ifdef PICO_MPI_CUDA_AWARE
  BINE_CUDA_CHECK(cudaFree(tmpbuf));
  BINE_CUDA_CHECK(cudaFree(resbuf));
#else
  free(tmpbuf);
  free(resbuf);
#endif
  free(displs);
  free(step_to_send);
  return MPI_SUCCESS;

err_hndl:
  if (NULL != displs)
    free(displs);
  if (NULL != step_to_send)
    free(step_to_send);
  if (NULL != tmpbuf)
#ifdef PICO_MPI_CUDA_AWARE
    BINE_CUDA_CHECK(cudaFree(tmpbuf));
#else
    free(tmpbuf);
#endif
  if (NULL != resbuf)
#ifdef PICO_MPI_CUDA_AWARE
    BINE_CUDA_CHECK(cudaFree(resbuf));
#else
    free(resbuf);
#endif
  return err;
}

int reduce_scatter_bine_block_by_block_hierarchical(const void *sendbuf, void *recvbuf, const int recvcounts[],
                                                       MPI_Datatype dt, MPI_Op op, MPI_Comm comm)
{
  int size, rank, dtsize, err = MPI_SUCCESS;
  int local_rank, node_size, node_offset, node_rank, elem;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  MPI_Type_size(dt, &dtsize);
  int count = 0;

  int *displs = (int *)malloc(size * sizeof(int));
  int *step_to_send = (int *)malloc(size * sizeof(int));

  int task_on_node = pico_task_on_node();
  for (int i = 0; i < size; i++)
  {
    displs[i] = count;
    count += recvcounts[i];
  }

  pico_get_group_config(&node_size, &node_rank, &node_offset, &local_rank, task_on_node, size, rank);

  void *tmpbuf, *resbuf;
#ifdef PICO_MPI_CUDA_AWARE
  BINE_CUDA_CHECK(cudaMalloc(&tmpbuf, (count / task_on_node * (task_on_node - 1)) * dtsize));
  BINE_CUDA_CHECK(cudaMalloc(&resbuf, (count / task_on_node) * dtsize));
#else
  tmpbuf = malloc((count / task_on_node * (task_on_node - 1)) * dtsize);
  resbuf = malloc((count / task_on_node) * dtsize);
#endif

  int *inverse_remapping = (int *)malloc(node_size * sizeof(int));  
  MPI_Request *send_req = (MPI_Request *)malloc(node_size * (task_on_node - 1) * sizeof(MPI_Request));
  MPI_Request *recv_req = (MPI_Request *)malloc(node_size * (task_on_node - 1) * sizeof(MPI_Request));

  if (NULL == displs || NULL == step_to_send || NULL == tmpbuf || NULL == resbuf || NULL == inverse_remapping || NULL == send_req || NULL == recv_req)
  {
    err = MPI_ERR_NO_MEM;
    goto err_hndl;
  }

  int copy_offset = 0;
  for (int i = 0; i < node_size; i++)
  {
    inverse_remapping[remap_rank(node_size, i)] = i;
    elem = local_rank + task_on_node * i;
    COPY_BUFF_DIFF_DT(sendbuf + displs[elem] * dtsize, recvcounts[elem], dt, resbuf + copy_offset * dtsize, recvcounts[elem], dt);
    copy_offset += recvcounts[elem];
  }

  int local_reduce_count = copy_offset;
  // memcpy(resbuf, sendbuf, count * dtsize);

  int recv_offset = 0;
  int recv_req_count = 0, send_req_count = 0;
  for (int i = 0; i < task_on_node; i++)
  {
    if (local_rank == i)
      continue;

    for (int j = 0; j < node_size; j++)
    {
      elem = local_rank + j * task_on_node;
      err = MPI_Irecv(tmpbuf + recv_offset * dtsize, recvcounts[elem], dt,
                      node_offset + i, 0, comm, &recv_req[recv_req_count]);
      if (MPI_SUCCESS != err)
      {
        goto err_hndl;
      }
      recv_req_count++;
      recv_offset += recvcounts[elem];      
    
      elem = i + j * task_on_node;
      err = MPI_Isend(sendbuf + displs[elem] * dtsize, recvcounts[elem], dt, node_offset + i, 0, comm, &send_req[send_req_count]);
      if (MPI_SUCCESS != err)
      {
        goto err_hndl;
      }
      send_req_count++;
    }
  }

  err = MPI_Waitall(recv_req_count, recv_req, MPI_STATUSES_IGNORE);

#ifdef PICO_MPI_CUDA_AWARE
  err = reduce_wrapper_grops(tmpbuf, resbuf, local_reduce_count, task_on_node - 1, dt, op);
  if (MPI_SUCCESS != err)
  {
    goto err_hndl;
  }
  BINE_CUDA_CHECK(cudaDeviceSynchronize());
#else
  for (int i = 0; i < task_on_node - 1; i++)
  {
    err = MPI_Reduce_local(tmpbuf + (ptrdiff_t)i * (ptrdiff_t)local_reduce_count * (ptrdiff_t)dtsize,
                           resbuf, local_reduce_count, dt, op);
    if (MPI_SUCCESS != err)
    {
      goto err_hndl;
    }
  }
#endif

  err = MPI_Waitall(send_req_count, send_req, MPI_STATUSES_IGNORE);

  int mask = 0x1;
  int inverse_mask = 0x1 << (int)(log_2(node_size) - 1);
  int block_first_mask = ~(inverse_mask - 1);
  int remapped_rank = remap_rank(node_size, node_rank);
  recv_req_count = send_req_count = 0;
  while (mask < node_size)
  {
    int partner;
    if (node_rank % 2 == 0)
    {
      partner = mod(node_rank + negabinary_to_binary((mask << 1) - 1), node_size);
    }
    else
    {
      partner = mod(node_rank - negabinary_to_binary((mask << 1) - 1), node_size);
    }

    // For sure I need to send my (remapped) partner's data
    // the actual start block however must be aligned to
    // the power of two
    int send_block_first = remap_rank(node_size, partner) & block_first_mask;
    int send_block_last = send_block_first + inverse_mask - 1;
    // Something similar for the block to recv.
    // I receive my block, but aligned to the power of two
    int recv_block_first = remapped_rank & block_first_mask;
    int recv_block_last = recv_block_first + inverse_mask - 1;

    send_req_count = recv_req_count = 0;
    for (size_t block = recv_block_first; block <= recv_block_last; block++)
    {
      if (mask << 1 >= node_size)
      {
        // Last step, receiving in recvbuf
        err = MPI_Irecv((char *)recvbuf, recvcounts[inverse_remapping[block]], dt, partner * task_on_node + local_rank, 0,
                        comm, &recv_req[recv_req_count]);
      }
      else
      {
        err = MPI_Irecv((char *)tmpbuf + displs[inverse_remapping[block]] * dtsize, recvcounts[inverse_remapping[block]], dt, partner * task_on_node + local_rank, 0,
                        comm, &recv_req[recv_req_count]);
      }
      if (MPI_SUCCESS != err)
      {
        goto err_hndl;
      }
      ++recv_req_count;
    }

    for (size_t block = send_block_first; block <= send_block_last; block++)
    {
      err = MPI_Isend((char *)resbuf + displs[inverse_remapping[block]] * dtsize, recvcounts[inverse_remapping[block]], dt, partner * task_on_node + local_rank, 0,
                      comm, &send_req[send_req_count]);
      if (MPI_SUCCESS != err)
      {
        goto err_hndl;
      }
      ++send_req_count;
    }

    int w_req = 0;
    for (size_t block = recv_block_first; block <= recv_block_last; block++)
    {
      err = MPI_Wait(&recv_req[w_req], MPI_STATUS_IGNORE);

      if (MPI_SUCCESS != err)
      {
        goto err_hndl;
      }
      if (mask << 1 >= node_size)
      {
        // Last step, received in recvbuf, aggregating from resbuf
#ifdef PICO_MPI_CUDA_AWARE
        reduce_wrapper((char *)resbuf + displs[inverse_remapping[block]] * dtsize, (char *)recvbuf, recvcounts[inverse_remapping[block]], dt, op);
#else
        err = MPI_Reduce_local((char *)resbuf + displs[inverse_remapping[block]] * dtsize, (char *)recvbuf, recvcounts[inverse_remapping[block]], dt, op);
#endif
      }
      else
      {
#ifdef PICO_MPI_CUDA_AWARE
        reduce_wrapper((char *)tmpbuf + displs[inverse_remapping[block]] * dtsize, (char *)resbuf + displs[inverse_remapping[block]] * dtsize, recvcounts[inverse_remapping[block]], dt, op);
#else
        err = MPI_Reduce_local((char *)tmpbuf + displs[inverse_remapping[block]] * dtsize, (char *)resbuf + displs[inverse_remapping[block]] * dtsize, recvcounts[inverse_remapping[block]], dt, op);
#endif
      }
      if (MPI_SUCCESS != err)
      {
        goto err_hndl;
      }
      ++w_req;
    }
#ifdef PICO_MPI_CUDA_AWARE
    BINE_CUDA_CHECK(cudaDeviceSynchronize());
#endif

    err = MPI_Waitall(send_req_count, send_req, MPI_STATUSES_IGNORE);
    if (MPI_SUCCESS != err)
    {
      goto err_hndl;
    }

    mask <<= 1;
    inverse_mask >>= 1;
    block_first_mask >>= 1;
  }

#ifdef PICO_MPI_CUDA_AWARE
  BINE_CUDA_CHECK(cudaFree(tmpbuf));
  BINE_CUDA_CHECK(cudaFree(resbuf));
#else
  free(tmpbuf);
  free(resbuf);
#endif
  free(send_req); 
  free(recv_req);
  free(inverse_remapping);
  free(step_to_send);
  free(displs);
  return MPI_SUCCESS;

err_hndl:
  if (NULL != send_req)
    free(send_req);
  if (NULL != recv_req)
    free(recv_req);
  if (NULL != displs)
    free(displs);
  if (NULL != step_to_send)
    free(step_to_send);
  if (NULL != inverse_remapping)
    free(inverse_remapping);
  if (NULL != tmpbuf)
#ifdef PICO_MPI_CUDA_AWARE
    BINE_CUDA_CHECK(cudaFree(tmpbuf));
#else
    free(tmpbuf);
#endif
  if (NULL != resbuf)
#ifdef PICO_MPI_CUDA_AWARE
    BINE_CUDA_CHECK(cudaFree(resbuf));
#else
    free(resbuf);
#endif
  return err;
}

int reduce_scatter_bine_block_by_block(const void *sendbuf, void *recvbuf, const int recvcounts[],
                                       MPI_Datatype dt, MPI_Op op, MPI_Comm comm)
{
  int size, rank, dtsize, err = MPI_SUCCESS;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  MPI_Type_size(dt, &dtsize);
  int count = 0;
  int *displs = (int *)malloc(size * sizeof(int));
  int *step_to_send = (int *)malloc(size * sizeof(int));
  int *inverse_remapping = (int *)malloc(size * sizeof(int));
  for (int i = 0; i < size; i++)
  {
    displs[i] = count;
    count += recvcounts[i];
    inverse_remapping[remap_rank(size, i)] = i;
  }

  void *tmpbuf, *resbuf;
#ifdef PICO_MPI_CUDA_AWARE
  BINE_CUDA_CHECK(cudaMalloc(&tmpbuf, count * dtsize));
  BINE_CUDA_CHECK(cudaMalloc(&resbuf, count * dtsize));
#else
  tmpbuf = malloc(count * dtsize);
  resbuf = malloc(count * dtsize);
#endif
  MPI_Request *reqs = NULL;

  if (NULL == displs || NULL == step_to_send || NULL == tmpbuf || NULL == resbuf || NULL == inverse_remapping)
  {
    err = MPI_ERR_NO_MEM;
    goto err_hndl;
  }
  COPY_BUFF_DIFF_DT(sendbuf, count, dt, resbuf, count, dt);
  // memcpy(resbuf, sendbuf, count * dtsize);

  int mask = 0x1;
  int inverse_mask = 0x1 << (int)(log_2(size) - 1);
  int block_first_mask = ~(inverse_mask - 1);
  int remapped_rank = remap_rank(size, rank);
  reqs = (MPI_Request *)malloc(size * sizeof(MPI_Request));
  while (mask < size)
  {
    int partner;
    if (rank % 2 == 0)
    {
      partner = mod(rank + negabinary_to_binary((mask << 1) - 1), size);
    }
    else
    {
      partner = mod(rank - negabinary_to_binary((mask << 1) - 1), size);
    }

    // For sure I need to send my (remapped) partner's data
    // the actual start block however must be aligned to
    // the power of two
    int send_block_first = remap_rank(size, partner) & block_first_mask;
    int send_block_last = send_block_first + inverse_mask - 1;
    // Something similar for the block to recv.
    // I receive my block, but aligned to the power of two
    int recv_block_first = remapped_rank & block_first_mask;
    int recv_block_last = recv_block_first + inverse_mask - 1;

    int next_req = 0;
    for (size_t block = recv_block_first; block <= recv_block_last; block++)
    {
      if (mask << 1 >= size)
      {
        // Last step, receiving in recvbuf
        err = MPI_Irecv((char *)recvbuf, recvcounts[inverse_remapping[block]], dt, partner, 0,
                        comm, &reqs[next_req]);
      }
      else
      {
        err = MPI_Irecv((char *)tmpbuf + displs[inverse_remapping[block]] * dtsize, recvcounts[inverse_remapping[block]], dt, partner, 0,
                        comm, &reqs[next_req]);
      }
      if (MPI_SUCCESS != err)
      {
        goto err_hndl;
      }
      ++next_req;
    }

    for (size_t block = send_block_first; block <= send_block_last; block++)
    {
      err = MPI_Isend((char *)resbuf + displs[inverse_remapping[block]] * dtsize, recvcounts[inverse_remapping[block]], dt, partner, 0,
                      comm, &reqs[next_req]);
      if (MPI_SUCCESS != err)
      {
        goto err_hndl;
      }
      ++next_req;
    }

    int w_req = 0;
    for (size_t block = recv_block_first; block <= recv_block_last; block++)
    {
      err = MPI_Wait(&reqs[w_req], MPI_STATUS_IGNORE);
      if (MPI_SUCCESS != err)
      {
        goto err_hndl;
      }
      if (mask << 1 >= size)
      {
        // Last step, received in recvbuf, aggregating from resbuf
#ifdef PICO_MPI_CUDA_AWARE
        reduce_wrapper((char *)resbuf + displs[inverse_remapping[block]] * dtsize, (char *)recvbuf, recvcounts[inverse_remapping[block]], dt, op);
#else
        err = MPI_Reduce_local((char *)resbuf + displs[inverse_remapping[block]] * dtsize, (char *)recvbuf, recvcounts[inverse_remapping[block]], dt, op);
#endif
      }
      else
      {
#ifdef PICO_MPI_CUDA_AWARE
        reduce_wrapper((char *)tmpbuf + displs[inverse_remapping[block]] * dtsize, (char *)resbuf + displs[inverse_remapping[block]] * dtsize, recvcounts[inverse_remapping[block]], dt, op);
#else
        err = MPI_Reduce_local((char *)tmpbuf + displs[inverse_remapping[block]] * dtsize, (char *)resbuf + displs[inverse_remapping[block]] * dtsize, recvcounts[inverse_remapping[block]], dt, op);
#endif
      }
      if (MPI_SUCCESS != err)
      {
        goto err_hndl;
      }
      ++w_req;
    }
#ifdef PICO_MPI_CUDA_AWARE
    BINE_CUDA_CHECK(cudaDeviceSynchronize());
#endif
    err = MPI_Waitall(next_req - w_req, &reqs[w_req], MPI_STATUSES_IGNORE);
    if (MPI_SUCCESS != err)
    {
      goto err_hndl;
    }

    mask <<= 1;
    inverse_mask >>= 1;
    block_first_mask >>= 1;
  }

  free(reqs);
#ifdef PICO_MPI_CUDA_AWARE
  BINE_CUDA_CHECK(cudaFree(tmpbuf));
  BINE_CUDA_CHECK(cudaFree(resbuf));
#else
  free(tmpbuf);
  free(resbuf);
#endif
  free(inverse_remapping);
  free(step_to_send);
  free(displs);
  return MPI_SUCCESS;

err_hndl:
  if (NULL != reqs)
    free(reqs);
  if (NULL != displs)
    free(displs);
  if (NULL != step_to_send)
    free(step_to_send);
  if (NULL != inverse_remapping)
    free(inverse_remapping);
  if (NULL != tmpbuf)
#ifdef PICO_MPI_CUDA_AWARE
    BINE_CUDA_CHECK(cudaFree(tmpbuf));
#else
    free(tmpbuf);
#endif
  if (NULL != resbuf)
#ifdef PICO_MPI_CUDA_AWARE
    BINE_CUDA_CHECK(cudaFree(resbuf));
#else
    free(resbuf);
#endif
  return err;
}

int reduce_scatter_bine_block_by_block_any_even(const void *sendbuf, void *recvbuf, const int recvcounts[],
                                                MPI_Datatype dt, MPI_Op op, MPI_Comm comm)
{
  int size, rank, dtsize, err = MPI_SUCCESS;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  MPI_Type_size(dt, &dtsize);
  int count = 0;
  int *displs = (int *)malloc(size * sizeof(int));
  for (int i = 0; i < size; i++)
  {
    displs[i] = count;
    count += recvcounts[i];
  }

  void *tmpbuf = malloc(count * dtsize);
  void *resbuf = malloc(count * dtsize);
  memcpy(resbuf, sendbuf, count * dtsize);

  int mask = 0x1;
  MPI_Request *reqs_s = (MPI_Request *)malloc(size * sizeof(MPI_Request));
  MPI_Request *reqs_r = (MPI_Request *)malloc(size * sizeof(MPI_Request));
  int *blocks_to_recv = (int *)malloc(size * sizeof(int));
  int next_req_s = 0, next_req_r = 0;
  int reverse_step = log_2(size) - 1;
  int last_recv_done = 0;
  while (mask < size)
  {
    int partner;
    if (rank % 2 == 0)
    {
      partner = mod(rank + negabinary_to_binary((mask << 1) - 1), size);
    }
    else
    {
      partner = mod(rank - negabinary_to_binary((mask << 1) - 1), size);
    }

    next_req_r = 0;
    next_req_s = 0;

    // We start from 1 because 0 never sends block 0
    for (size_t block = 1; block < size; block++)
    {
      // Get the position of the highest set bit using clz
      // That gives us the first at which block departs from 0
      int k = 31 - __builtin_clz(get_nu(block, size));
      // Check if this must be sent
      if (k == reverse_step)
      {
        // 0 would send this block
        size_t block_to_send, block_to_recv;
        if (rank % 2 == 0)
        {
          // I am even, thus I need to shift by rank position to the right
          block_to_send = mod(block + rank, size);
          // What to receive? What my partner is sending
          // Since I am even, my partner is odd, thus I need to mirror it and then shift
          block_to_recv = mod(partner - block, size);
        }
        else
        {
          // I am odd, thus I need to mirror it
          block_to_send = mod(rank - block, size);
          // What to receive? What my partner is sending
          // Since I am odd, my partner is even, thus I need to mirror it and then shift
          block_to_recv = mod(block + partner, size);
        }

        if (block_to_send != rank)
        {
          err = MPI_Isend((char *)resbuf + displs[block_to_send] * dtsize, recvcounts[block_to_send], dt, partner, 0,
                          comm, &reqs_s[next_req_s]);
          if (MPI_SUCCESS != err)
          {
            goto err_hndl;
          }
          ++next_req_s;
        }

        if (block_to_recv != partner)
        {
          blocks_to_recv[next_req_r] = block_to_recv;
          if (mask << 1 >= size)
          {
            // Last step, receiving in recvbuf
            err = MPI_Irecv((char *)recvbuf, recvcounts[block_to_recv], dt, partner, 0,
                            comm, &reqs_r[next_req_r]);
            if (MPI_SUCCESS != err)
            {
              goto err_hndl;
            }
            last_recv_done = 1;
          }
          else
          {
            err = MPI_Irecv((char *)tmpbuf + displs[block_to_recv] * dtsize, recvcounts[block_to_recv], dt, partner, 0,
                            comm, &reqs_r[next_req_r]);
            if (MPI_SUCCESS != err)
            {
              goto err_hndl;
            }
          }
          ++next_req_r;
        }
      }
    }

    for (size_t block = 0; block < next_req_r; block++)
    {
      err = MPI_Wait(&reqs_r[block], MPI_STATUS_IGNORE);
      if (MPI_SUCCESS != err)
      {
        goto err_hndl;
      }
      if (mask << 1 >= size)
      {
        // Last step, received in recvbuf, aggregating from resbuf
        err = MPI_Reduce_local((char *)resbuf + displs[blocks_to_recv[block]] * dtsize, (char *)recvbuf, recvcounts[blocks_to_recv[block]], dt, op);
        if (MPI_SUCCESS != err)
        {
          goto err_hndl;
        }
      }
      else
      {
        err = MPI_Reduce_local((char *)tmpbuf + displs[blocks_to_recv[block]] * dtsize, (char *)resbuf + displs[blocks_to_recv[block]] * dtsize, recvcounts[blocks_to_recv[block]], dt, op);
        if (MPI_SUCCESS != err)
        {
          goto err_hndl;
        }
      }
    }
    err = MPI_Waitall(next_req_s, reqs_s, MPI_STATUSES_IGNORE);
    if (MPI_SUCCESS != err)
    {
      goto err_hndl;
    }

    mask <<= 1;
    reverse_step--;
  }
  if (!last_recv_done)
  {
    memcpy(recvbuf, (char *)resbuf + displs[rank] * dtsize, recvcounts[rank] * dtsize);
  }

  free(blocks_to_recv);
  free(reqs_s);
  free(reqs_r);
  free(displs);
  free(tmpbuf);
  free(resbuf);
  return MPI_SUCCESS;

err_hndl:
  if (NULL != blocks_to_recv)
    free(blocks_to_recv);
  if (NULL != reqs_s)
    free(reqs_s);
  if (NULL != reqs_r)
    free(reqs_r);
  if (NULL != displs)
    free(displs);
  if (NULL != tmpbuf)
    free(tmpbuf);
  if (NULL != resbuf)
    free(resbuf);
  return err;
}

// NOTE: Not fully implemented
//
// int reduce_scatter_bine_dtypes(const void *sbuf, void *rbuf, const int rcounts[],
//                                     MPI_Datatype dtype, MPI_Op op, MPI_Comm comm)
// {
//   int i, rank, size, steps, w_size, err = MPI_SUCCESS;
//   size_t count;
//   ptrdiff_t *disps = NULL;
//   ptrdiff_t extent, true_extent,lb, buf_size, gap = 0;
//   char *recv_buf = NULL, *recv_buf_free = NULL;
//   char *result_buf = NULL, *result_buf_free = NULL;
//
//   MPI_Datatype sendtype, recvtype;
//   int dis[2], blklens[2], total_count, dst;
//
//   MPI_Comm_size(comm, &size);
//   MPI_Comm_rank(comm, &rank);
//
//
//   /* Find displacements and total count*/
//   disps = (ptrdiff_t*) malloc(sizeof(ptrdiff_t) * size);
//   if(NULL == disps) return MPI_ERR_NO_MEM;
//
//   disps[0] = 0;
//   for(i = 0; i < (size - 1); ++i) {
//     disps[i + 1] = disps[i] + rcounts[i];
//   }
//   count = disps[size - 1] + rcounts[size - 1];
//
//   /* short cut the trivial case */
//   if(0 == count) {
//     free(disps);
//     return MPI_SUCCESS;
//   }
//
//   /* Current implementation only handles power-of-two number of processes.*/
//   steps = log_2(size);
//   if(!is_power_of_two(size) || steps < 1) {
//     BINE_DEBUG_PRINT("ERROR! bine static reduce scatter works only with po2 ranks!");
//     return MPI_ERR_SIZE;
//   }
//
//   /* get datatype information */
//   MPI_Type_get_extent(dtype, &lb, &extent);
//   MPI_Type_get_true_extent(dtype, &gap, &true_extent);
//
//   // Calculate the total memory span
//   buf_size = true_extent + extent * (count - 1);
//
//
//   /* Handle MPI_IN_PLACE */
//   if(MPI_IN_PLACE == sbuf) {
//     sbuf = rbuf;
//   }
//
//   /* Allocate temporary receive buffer. */
//   recv_buf_free = (char*) malloc(buf_size);
//   recv_buf = recv_buf_free - gap;
//   if(NULL == recv_buf_free) {
//     err = MPI_ERR_NO_MEM;
//     goto cleanup;
//   }
//
//   /* allocate temporary buffer for results */
//   result_buf_free = (char*) malloc(buf_size);
//   result_buf = result_buf_free - gap;
//
//   /* copy local buffer into the temporary results */
//   err = copy_buffer_different_dt(sbuf, count, dtype, result_buf, count, dtype);
//   if(MPI_SUCCESS != err) goto cleanup;
//
//   /* Main communication and reduction loop */
//   w_size = size >> 1;
//   for(int step = 0; step < steps; step++){
//     int peer, send_index, recv_index;
//     size_t s_count, r_count;
//
//     /* determine the peer we will communicate with */
//     peer = pi(rank, step, size);
//
//     /* use the bitmaps to figure out what we're sending and
//       receiving */
//     send_index = s_bitmap[step];
//     recv_index = r_bitmap[step];
//     s_count = r_count = 0;
//     for(int i = 0; i < w_size; ++i) {
//       s_count += rcounts[s_bitmap[step] + i];
//       r_count += rcounts[r_bitmap[step] + i];
//     }
//
//     /* actual data transfer.  Send from result_buf,
//         receive into recv_buf */
//     err = MPI_Sendrecv(result_buf + disps[send_index] * extent, s_count, dtype, peer, 0,
//                recv_buf + disps[recv_index] * extent, r_count, dtype, peer, 0,
//                comm, MPI_STATUS_IGNORE);
//     if(MPI_SUCCESS != err) {
//       goto cleanup;
//     }
//
//     /* if we received something on this step, push it into
//         the results buffer and perform the reduction*/
//     if(r_count > 0) {
//       err = MPI_Reduce_local(recv_buf + disps[recv_index] * extent,
//                         result_buf + disps[recv_index] * extent,
//                         r_count, dtype, op);
//       if(MPI_SUCCESS != err) {
//         goto cleanup;
//       }
//     }
//     /* halve the window size for the next iteration */
//     w_size >>= 1;
//   }
//
//
//   if(rank != permutation[rank]) {
//     /* send result to correct rank's recv buffer */
//     err = MPI_Sendrecv(result_buf + disps[r_bitmap[steps - 1]] * extent,
//             rcounts[permutation[rank]], dtype, permutation[rank], 0,
//             rbuf, rcounts[rank], dtype, get_sender(permutation, size, rank), 0,
//             comm, MPI_STATUS_IGNORE);
//     if(MPI_SUCCESS != err) {
//       goto cleanup;
//     }
//   } else {
//     /* copy local results from results buffer into real receive buffer */
//     err = copy_buffer_different_dt(result_buf + disps[rank] * extent, rcounts[rank],
//                                     dtype, rbuf, rcounts[rank], dtype);
//     if(MPI_SUCCESS != err) {
//       goto cleanup;
//     }
//   }
//
//  cleanup:
//   if(NULL != disps) free(disps);
//   if(NULL != recv_buf_free) free(recv_buf_free);
//   if(NULL != result_buf_free) free(result_buf_free);
//
//   return err;
// }
