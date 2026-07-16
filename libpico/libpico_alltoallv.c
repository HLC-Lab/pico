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
#ifdef PICO_MPI_CUDA_AWARE
    assert(sendtype == recvtype);
    void *host_sendbuf = NULL, *host_recvbuf = NULL;
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
    size_t num_bytes_send = num_bytes_fun(sendcounts,sdispls,  dtype, size);
    size_t num_bytes_recv = num_bytes_fun(recvcounts,rdispls ,dtype, size);
    PICO_TAG_BEGIN("cudaMalloc cudaMemcpy");
    if (num_bytes_send > 0){
        BINE_CUDA_CHECK(cudaMallocHost((void **)&host_sendbuf, num_bytes_send));
        BINE_CUDA_CHECK(cudaMemcpy(host_sendbuf,sendbuf, num_bytes_send, cudaMemcpyDeviceToHost));
    }
    if (num_bytes_recv > 0)
        BINE_CUDA_CHECK(cudaMallocHost((void **)&host_recvbuf, num_bytes_recv));
    PICO_TAG_END("cudaMalloc cudaMemcpy");
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

    work_buffer = malloc(max_dim_buffer * 3);
    if (work_buffer == NULL)
    {
        err = MPI_ERR_NO_MEM;
        goto err_hndl;
    }

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
            memcpy(rec, &src, sizeof(int));
            memcpy(rec + sizeof(int), &dst, sizeof(int));
            memcpy(rec + 2 * sizeof(int), &block_size, sizeof(int));

            memcpy(rec + header_size,
                   (const char *)host_sendbuf + offset,
                   (size_t)block_size);

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
            memcpy(&src, rec, sizeof(int));
            memcpy(&dst, rec + sizeof(int), sizeof(int));
            memcpy(&block_size, rec + 2 * sizeof(int), sizeof(int));

            size_t packet_size = header_size + (size_t)block_size;
            int logical_dst = logical_rank_for_bine_dh_root(dst, src, size);
            int logical_partner = logical_rank_for_bine_dh_root(partner, src, size);
            if (same_prefix_negabinary(logical_dst, logical_partner, s, step + 1))
            {
                memcpy(send_buffer + dim_send, rec, packet_size);
                dim_send += (int)packet_size;
            }
            else
            {
                if (dim_keep != (int)skip)
                    memmove(work_buffer + dim_keep, rec, packet_size);

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

            new_buffer = malloc(needed * 3);
            if (new_buffer == NULL)
            {
                err = MPI_ERR_NO_MEM;
                goto err_hndl;
            }

            char *new_work_buffer = new_buffer;
            char *new_send_buffer = new_buffer + needed;
            char *new_keep_buffer = new_buffer + 2 * needed;
            PICO_TAG_END("step:realloc");
            PICO_TAG_BEGIN("step:memcpy");
            memcpy(new_work_buffer, work_buffer, (size_t)dim_keep);
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

            free(old_buffer);
        }
        else
        {

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
        memcpy(&src, rec, sizeof(int));
        memcpy(&dst, rec + sizeof(int), sizeof(int));
        memcpy(&block_size, rec + 2 * sizeof(int), sizeof(int));

        size_t packet_size = header_size + (size_t)block_size;

        assert(dst == r);

        size_t offset = (size_t)rdispls[src] * (size_t)dtype;

        if (block_size > 0)
        {
            memcpy((char *)host_recvbuf + offset,
                   rec + header_size,
                   (size_t)block_size);
        }

        skip += packet_size;
    }
    PICO_TAG_END("permutation");
    PICO_TAG_BEGIN("final cudaMemcpy");
    if (num_bytes_recv > 0){
        BINE_CUDA_CHECK(cudaMemcpy(recvbuf,host_recvbuf, num_bytes_recv, cudaMemcpyHostToDevice));
    }
    PICO_TAG_END("final cudaMemcpy");
err_hndl:
    free(work_buffer);
    if (host_sendbuf != NULL)
        BINE_CUDA_CHECK(cudaFreeHost(host_sendbuf));
    if (host_recvbuf != NULL)
        BINE_CUDA_CHECK(cudaFreeHost(host_recvbuf));
    return err;
#else
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

    work_buffer = malloc(max_dim_buffer * 3);
    if (work_buffer == NULL)
    {
        err = MPI_ERR_NO_MEM;
        goto err_hndl;
    }

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

            memcpy(rec, &src, sizeof(int));
            memcpy(rec + sizeof(int), &dst, sizeof(int));
            memcpy(rec + 2 * sizeof(int), &block_size, sizeof(int));

            memcpy(rec + header_size,
                   (const char *)sendbuf + offset,
                   (size_t)block_size);

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
            memcpy(&src, rec, sizeof(int));
            memcpy(&dst, rec + sizeof(int), sizeof(int));
            memcpy(&block_size, rec + 2 * sizeof(int), sizeof(int));

            size_t packet_size = header_size + (size_t)block_size;
            int logical_dst = logical_rank_for_bine_dh_root(dst, src, size);
            int logical_partner = logical_rank_for_bine_dh_root(partner, src, size);
            if (same_prefix_negabinary(logical_dst, logical_partner, s, step + 1))
            {
                memcpy(send_buffer + dim_send, rec, packet_size);
                dim_send += (int)packet_size;
            }
            else
            {
                if (dim_keep != (int)skip)
                    memmove(work_buffer + dim_keep, rec, packet_size);

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

            new_buffer = malloc(needed * 3);
            if (new_buffer == NULL)
            {
                err = MPI_ERR_NO_MEM;
                goto err_hndl;
            }

            char *new_work_buffer = new_buffer;
            char *new_send_buffer = new_buffer + needed;
            char *new_keep_buffer = new_buffer + 2 * needed;
            PICO_TAG_END("step:realloc");
            PICO_TAG_BEGIN("step:memcpy");
            memcpy(new_work_buffer, work_buffer, (size_t)dim_keep);
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

            free(old_buffer);
        }
        else
        {

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
        memcpy(&src, rec, sizeof(int));
        memcpy(&dst, rec + sizeof(int), sizeof(int));
        memcpy(&block_size, rec + 2 * sizeof(int), sizeof(int));

        size_t packet_size = header_size + (size_t)block_size;

        assert(dst == r);

        size_t offset = (size_t)rdispls[src] * (size_t)dtype;

        if (block_size > 0)
        {
            memcpy((char *)recvbuf + offset,
                   rec + header_size,
                   (size_t)block_size);
        }

        skip += packet_size;
    }
    PICO_TAG_END("permutation");
err_hndl:
    free(work_buffer);
    return err;
#endif
}

// write NCCL implementations here
#ifdef PICO_NCCL

static int ncclSpreadOutAllToAllvGPU(void *output,
                                     const void *input,
                                     const int *send_bytes,
                                     const int *recv_bytes,
                                     const int *send_byte_displs,
                                     const int *recv_byte_displs,
                                     int rank,
                                     int size,
                                     ncclComm_t comm,
                                     cudaStream_t stream)
{
    /*
     * Self-copy: local data rank -> rank.
     */
    if (send_bytes[rank] > 0)
    {
        BINE_CUDA_CHECK(cudaMemcpyAsync(
            (char *)output + recv_byte_displs[rank],
            (const char *)input + send_byte_displs[rank],
            send_bytes[rank],
            cudaMemcpyDeviceToDevice,
            stream));
    }

    /*
     * Spread-out Alltoallv:
     * each rank communicates with one send partner and one receive partner per step.
     */
    for (int step = 1; step < size; step++)
    {
        int send_partner = (rank + step) % size;
        int recv_partner = (rank - step + size) % size;

        BINE_NCCL_CHECK(ncclGroupStart());

        if (send_bytes[send_partner] > 0)
        {
            const char *send_ptr =
                (const char *)input + send_byte_displs[send_partner];

            BINE_NCCL_CHECK(ncclSend(send_ptr,
                                     send_bytes[send_partner],
                                     ncclInt8,
                                     send_partner,
                                     comm,
                                     stream));
        }

        if (recv_bytes[recv_partner] > 0)
        {
            char *recv_ptr =
                (char *)output + recv_byte_displs[recv_partner];

            BINE_NCCL_CHECK(ncclRecv(recv_ptr,
                                     recv_bytes[recv_partner],
                                     ncclInt8,
                                     recv_partner,
                                     comm,
                                     stream));
        }

        BINE_NCCL_CHECK(ncclGroupEnd());
    }

    return MPI_SUCCESS;
}

static int ncclAllToAllvGPU(void *output,
                            const void *input,
                            const int *send_bytes,
                            const int *recv_bytes,
                            const int *send_byte_displs,
                            const int *recv_byte_displs,
                            int rank,
                            int size,
                            ncclComm_t comm,
                            cudaStream_t stream)
{
    /*
     * Self-copy: local data rank -> rank.
     * We handle self locally and skip peer == rank in NCCL P2P.
     */
    if (send_bytes[rank] > 0)
    {
        BINE_CUDA_CHECK(cudaMemcpyAsync(
            (char *)output + recv_byte_displs[rank],
            (const char *)input + send_byte_displs[rank],
            send_bytes[rank],
            cudaMemcpyDeviceToDevice,
            stream));
    }

    /*
     * Fan-out Alltoallv:
     * all ncclSend/ncclRecv operations are posted inside one NCCL group.
     */
    BINE_NCCL_CHECK(ncclGroupStart());

    for (int peer = 0; peer < size; peer++)
    {
        if (peer == rank)
            continue;

        if (send_bytes[peer] > 0)
        {
            const char *send_ptr =
                (const char *)input + send_byte_displs[peer];

            BINE_NCCL_CHECK(ncclSend(send_ptr,
                                     send_bytes[peer],
                                     ncclInt8,
                                     peer,
                                     comm,
                                     stream));
        }

        if (recv_bytes[peer] > 0)
        {
            char *recv_ptr =
                (char *)output + recv_byte_displs[peer];

            BINE_NCCL_CHECK(ncclRecv(recv_ptr,
                                     recv_bytes[peer],
                                     ncclInt8,
                                     peer,
                                     comm,
                                     stream));
        }
    }

    BINE_NCCL_CHECK(ncclGroupEnd());

    return MPI_SUCCESS;
}

static int nccl_dtype_size_bytes(ncclDataType_t dtype, int *dtype_size)
{
    switch (dtype)
    {
    case ncclInt8:
    case ncclUint8:
        *dtype_size = 1;
        return MPI_SUCCESS;

    case ncclFloat16:
        *dtype_size = 2;
        return MPI_SUCCESS;

    case ncclInt32:
    case ncclUint32:
    case ncclFloat32:
        *dtype_size = 4;
        return MPI_SUCCESS;

    case ncclInt64:
    case ncclUint64:
    case ncclFloat64:
        *dtype_size = 8;
        return MPI_SUCCESS;

    default:
        fprintf(stderr, "Unsupported NCCL datatype in Alltoallv\n");
        return MPI_ERR_TYPE;
    }
}

int alltoallv_nccl_spreadout(ALLTOALLV_NCCL_ARGS)
{
    int rank, size, dtype_size;
    int err = MPI_SUCCESS;

    BINE_NCCL_CHECK(ncclCommUserRank(nccl_comm, &rank));
    BINE_NCCL_CHECK(ncclCommCount(nccl_comm, &size));

    err = nccl_dtype_size_bytes(dtype, &dtype_size);
    if (err != MPI_SUCCESS)
        return err;

    int *send_bytes = malloc(size * sizeof(int));
    int *recv_bytes = malloc(size * sizeof(int));
    int *send_byte_displs = malloc(size * sizeof(int));
    int *recv_byte_displs = malloc(size * sizeof(int));

    if (!send_bytes || !recv_bytes || !send_byte_displs || !recv_byte_displs)
    {
        free(send_bytes);
        free(recv_bytes);
        free(send_byte_displs);
        free(recv_byte_displs);
        return MPI_ERR_NO_MEM;
    }

    for (int i = 0; i < size; i++)
    {
        send_bytes[i] = scounts[i] * dtype_size;
        recv_bytes[i] = rcounts[i] * dtype_size;
        send_byte_displs[i] = sdispls[i] * dtype_size;
        recv_byte_displs[i] = rdispls[i] * dtype_size;
    }

    err = ncclSpreadOutAllToAllvGPU(rbuf,
                                    sbuf,
                                    send_bytes,
                                    recv_bytes,
                                    send_byte_displs,
                                    recv_byte_displs,
                                    rank,
                                    size,
                                    nccl_comm,
                                    stream);

    free(send_bytes);
    free(recv_bytes);
    free(send_byte_displs);
    free(recv_byte_displs);

    return err;
}

int alltoallv_nccl_fanout(ALLTOALLV_NCCL_ARGS)
{
    int rank, size, dtype_size;
    int err = MPI_SUCCESS;

    BINE_NCCL_CHECK(ncclCommUserRank(nccl_comm, &rank));
    BINE_NCCL_CHECK(ncclCommCount(nccl_comm, &size));

    err = nccl_dtype_size_bytes(dtype, &dtype_size);
    if (err != MPI_SUCCESS)
        return err;

    int *send_bytes = malloc(size * sizeof(int));
    int *recv_bytes = malloc(size * sizeof(int));
    int *send_byte_displs = malloc(size * sizeof(int));
    int *recv_byte_displs = malloc(size * sizeof(int));

    if (!send_bytes || !recv_bytes || !send_byte_displs || !recv_byte_displs)
    {
        free(send_bytes);
        free(recv_bytes);
        free(send_byte_displs);
        free(recv_byte_displs);
        return MPI_ERR_NO_MEM;
    }

    for (int i = 0; i < size; i++)
    {
        send_bytes[i] = scounts[i] * dtype_size;
        recv_bytes[i] = rcounts[i] * dtype_size;
        send_byte_displs[i] = sdispls[i] * dtype_size;
        recv_byte_displs[i] = rdispls[i] * dtype_size;
    }

    err = ncclAllToAllvGPU(rbuf,
                           sbuf,
                           send_bytes,
                           recv_bytes,
                           send_byte_displs,
                           recv_byte_displs,
                           rank,
                           size,
                           nccl_comm,
                           stream);

    free(send_bytes);
    free(recv_bytes);
    free(send_byte_displs);
    free(recv_byte_displs);

    return err;
}

#endif