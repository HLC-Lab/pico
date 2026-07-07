#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "libpico.h"
#include "libpico_utils.h"
#ifdef PICO_MPI_CUDA_AWARE
#include "support_kernel.h"
#endif
#include <cuda_runtime.h>

// This implementation follows the distance-halving Bine butterfly pattern described in the paper
int alltoallv_bine_DH(const void *sendbuf, const int sendcounts[], const int sdispls[], MPI_Datatype sendtype,
                      void *recvbuf, const int recvcounts[], const int rdispls[], MPI_Datatype recvtype,
                      MPI_Comm comm)
{

    assert(sendtype == recvtype);

    int r, size, dtype, s, err = MPI_SUCCESS;
    char *work_buffer = NULL, *send_buffer = NULL, *keep_buffer = NULL;
    size_t header_size = 3 * sizeof(int), max_dim_buffer, dim_work = 0;

    err = MPI_Comm_rank(comm, &r);
    if (err != MPI_SUCCESS)
        goto err_hndl;

    err = MPI_Comm_size(comm, &size);
    if (err != MPI_SUCCESS)
        goto err_hndl;

    err = MPI_Type_size(sendtype, &dtype);
    if (err != MPI_SUCCESS)
        goto err_hndl;

    s = (int)log2((double)size);
    assert((1 << s) == size);

    size_t local_total_bytes = 0;
    size_t local_blocks = 0;

    for (int i = 0; i < size; i++)
    {
        size_t block_size = (size_t)sendcounts[i] * (size_t)dtype;

        if (block_size > 0)
        {
            local_total_bytes += block_size;
            local_blocks++;
        }
    }

    max_dim_buffer = local_total_bytes + local_blocks * header_size;

    if (max_dim_buffer == 0)
        max_dim_buffer = 1;

#ifdef PICO_MPI_CUDA_AWARE
    BINE_CUDA_CHECK(cudaMalloc((void **)&work_buffer, max_dim_buffer * 3));
#else
    work_buffer = malloc(max_dim_buffer * 3);
    if (work_buffer == NULL)
    {
        err = MPI_ERR_NO_MEM;
        goto err_hndl;
    }
#endif

    send_buffer = work_buffer + max_dim_buffer;
    keep_buffer = work_buffer + 2 * max_dim_buffer;
    PICO_TAG_BEGIN("init:for");
    for (int i = 0; i < size; i++)
    {
        int block_size = (int)((size_t)sendcounts[i] * (size_t)dtype);
        size_t offset = (size_t)sdispls[i] * (size_t)dtype;
        size_t packet_size = header_size + (size_t)block_size;
        if (block_size > 0)
        {
            char *rec = work_buffer + dim_work;
            int src = r;
            int dst = i;

#ifdef PICO_MPI_CUDA_AWARE
            int header[3] = {src, dst, block_size};
            BINE_CUDA_CHECK(cudaMemcpy(rec,header,header_size,cudaMemcpyHostToDevice));
            BINE_CUDA_CHECK(cudaMemcpy(rec + header_size,(const char *)sendbuf + offset,
                                       (size_t)block_size,cudaMemcpyDeviceToDevice));
#else
            memcpy(rec, &src, sizeof(int));
            memcpy(rec + sizeof(int), &dst, sizeof(int));
            memcpy(rec + 2 * sizeof(int), &block_size, sizeof(int));

            memcpy(rec + header_size,
                   (const char *)sendbuf + offset,
                   (size_t)block_size);
#endif

            dim_work += packet_size;
        }
    }
    PICO_TAG_END("init:for");
    for (int step = 0; step < s; step++)
    {
        int partner, dim_send = 0, dim_keep = 0, dim_recv = 0;

        if ((r % 2) == 0)
            partner = mod(r + (1 - (int)pow(-2, s - step)) / 3, size);
        else
            partner = mod(r - (1 - (int)pow(-2, s - step)) / 3, size);

        size_t skip = 0;
        PICO_TAG_BEGIN("step:while");
        while (skip < dim_work)
        {
            char *rec = work_buffer + skip;

            int dst, block_size, src;
#ifdef PICO_MPI_CUDA_AWARE
            int header[3];
            BINE_CUDA_CHECK(cudaMemcpy(header,rec,header_size,cudaMemcpyDeviceToHost));
            src = header[0];
            dst = header[1];
            block_size = header[2];
#else
            memcpy(&src, rec, sizeof(int));
            memcpy(&dst, rec + sizeof(int), sizeof(int));
            memcpy(&block_size, rec + 2 * sizeof(int), sizeof(int));
#endif
            size_t packet_size = header_size + (size_t)block_size;
            int logical_dst = logical_rank_for_bine_dh_root(dst, src, size);
            int logical_partner = logical_rank_for_bine_dh_root(partner, src, size);
            if (same_prefix_negabinary(logical_dst, logical_partner, s, step + 1))
            {
#ifdef PICO_MPI_CUDA_AWARE
                BINE_CUDA_CHECK(cudaMemcpy(send_buffer + dim_send,rec,packet_size,
                                           cudaMemcpyDeviceToDevice));
#else
                memcpy(send_buffer + dim_send, rec, packet_size);
#endif
                dim_send += (int)packet_size;
            }
            else
            {
#ifdef PICO_MPI_CUDA_AWARE
                BINE_CUDA_CHECK(cudaMemcpy(keep_buffer + dim_keep,rec,packet_size,
                                           cudaMemcpyDeviceToDevice));
#else
                if (dim_keep != (int)skip)
                    memmove(work_buffer + dim_keep, rec, packet_size);
#endif
                dim_keep += (int)packet_size;
            }

            skip += packet_size;
        }
        PICO_TAG_END("step:while");
        MPI_Request req;
        MPI_Status status;

        PICO_TAG_BEGIN("step:isend_data");
        err = MPI_Isend(send_buffer, dim_send, MPI_BYTE,
                        partner, 1, comm, &req);
        PICO_TAG_END("step:isend_data");
        if (err != MPI_SUCCESS)
            goto err_hndl;

        PICO_TAG_BEGIN("step:probe");
        err = MPI_Probe(partner, 1, comm, &status);
        PICO_TAG_END("step:probe");
        if (err != MPI_SUCCESS)
            goto err_hndl;

        err = MPI_Get_count(&status, MPI_BYTE, &dim_recv);
        if (err != MPI_SUCCESS)
            goto err_hndl;

        size_t needed = (size_t)dim_keep + (size_t)dim_recv;
        if (needed == 0)
            needed = 1;

        if (needed > max_dim_buffer)
        {
            PICO_TAG_BEGIN("step:realloc");

            char *old_buffer = work_buffer;

            char *new_buffer = NULL;

#ifdef PICO_MPI_CUDA_AWARE
            BINE_CUDA_CHECK(cudaMalloc((void **)&new_buffer, needed * 3));
#else
            new_buffer = malloc(needed * 3);
            if (new_buffer == NULL)
            {
                err = MPI_ERR_NO_MEM;
                goto err_hndl;
            }
#endif

            char *new_work_buffer = new_buffer;
            char *new_send_buffer = new_buffer + needed;
            char *new_keep_buffer = new_buffer + 2 * needed;
            PICO_TAG_END("step:realloc");
            PICO_TAG_BEGIN("step:memcpy");
#ifdef PICO_MPI_CUDA_AWARE
            if (dim_keep > 0)
            {
                BINE_CUDA_CHECK(cudaMemcpy(new_work_buffer,keep_buffer,(size_t)dim_keep,
                                           cudaMemcpyDeviceToDevice));
            }
#else
            memcpy(new_work_buffer, work_buffer, (size_t)dim_keep);
#endif
            PICO_TAG_END("step:memcpy");
            work_buffer = new_work_buffer;
            send_buffer = new_send_buffer;
            keep_buffer = new_keep_buffer;

            max_dim_buffer = needed;

            PICO_TAG_BEGIN("step:recv_data");
            err = MPI_Recv(work_buffer + dim_keep, dim_recv, MPI_BYTE,
                           partner, 1, comm, MPI_STATUS_IGNORE);
            PICO_TAG_END("step:recv_data");
            if (err != MPI_SUCCESS)
                goto err_hndl;

            err = MPI_Wait(&req, MPI_STATUS_IGNORE);
            if (err != MPI_SUCCESS)
                goto err_hndl;

#ifdef PICO_MPI_CUDA_AWARE
            cudaFree(old_buffer);
#else
            free(old_buffer);
#endif
        }
        else
        {
#ifdef PICO_MPI_CUDA_AWARE
            if (dim_keep > 0)
            {
                BINE_CUDA_CHECK(cudaMemcpy(work_buffer,keep_buffer,(size_t)dim_keep,
                                           cudaMemcpyDeviceToDevice));
            }
#endif

            PICO_TAG_BEGIN("step:recv_data");
            err = MPI_Recv(work_buffer + dim_keep, dim_recv, MPI_BYTE,
                           partner, 1, comm, MPI_STATUS_IGNORE);
            PICO_TAG_END("step:recv_data");

            if (err != MPI_SUCCESS)
                goto err_hndl;

            err = MPI_Wait(&req, MPI_STATUS_IGNORE);
            if (err != MPI_SUCCESS)
                goto err_hndl;
        }

        dim_work = (size_t)dim_keep + (size_t)dim_recv;
    }
    PICO_TAG_BEGIN("permutation");
    size_t skip = 0;
    while (skip < dim_work)
    {
        char *rec = work_buffer + skip;

        int src, dst, block_size;

#ifdef PICO_MPI_CUDA_AWARE
        int header[3];
        BINE_CUDA_CHECK(cudaMemcpy(header,rec,header_size,
                                   cudaMemcpyDeviceToHost));
        src = header[0];
        dst = header[1];
        block_size = header[2];
#else
        memcpy(&src, rec, sizeof(int));
        memcpy(&dst, rec + sizeof(int), sizeof(int));
        memcpy(&block_size, rec + 2 * sizeof(int), sizeof(int));
#endif

        size_t packet_size = header_size + (size_t)block_size;

        assert(dst == r);

        size_t offset = (size_t)rdispls[src] * (size_t)dtype;

        if (block_size > 0)
        {
#ifdef PICO_MPI_CUDA_AWARE
            BINE_CUDA_CHECK(cudaMemcpy((char *)recvbuf + offset,rec + header_size,(size_t)block_size,
                                       cudaMemcpyDeviceToDevice));
#else
            memcpy((char *)recvbuf + offset,
                   rec + header_size,
                   (size_t)block_size);
#endif
        }

        skip += packet_size;
    }
    PICO_TAG_END("permutation");
err_hndl:
#ifdef PICO_MPI_CUDA_AWARE
    cudaFree(work_buffer);
#else
    free(work_buffer);
#endif
    return err;
}