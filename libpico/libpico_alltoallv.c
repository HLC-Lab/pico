#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <math.h>
#include "libpico.h"
#include "libpico_utils.h"

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

    unsigned long local_total_bytes = 0;
    unsigned long global_total_bytes = 0;

    for (int i = 0; i < size; i++)
    {
        local_total_bytes += (unsigned long)((size_t)sendcounts[i] * (size_t)dtype);
    }

    err = allreduce_bine_lat(&local_total_bytes, &global_total_bytes, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm);
    if (err != MPI_SUCCESS)
        goto err_hndl;
    max_dim_buffer = (size_t)size * header_size + (size_t)global_total_bytes;

    work_buffer = malloc(max_dim_buffer);
    send_buffer = malloc(max_dim_buffer);
    keep_buffer = malloc(max_dim_buffer);

    if (work_buffer == NULL || send_buffer == NULL || keep_buffer == NULL)
    {
        err = MPI_ERR_NO_MEM;
        goto err_hndl;
    }

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

            memcpy(rec + header_size, (const char *)sendbuf + offset, (size_t)block_size);

            dim_work += packet_size;
        }
    }

    for (int step = 0; step < s; step++)
    {
        int partner, dim_send = 0, dim_keep = 0, dim_recv = 0;

        if ((r % 2) == 0)
            partner = mod(r + (1 - (int)pow(-2, s - step)) / 3, size);
        else
            partner = mod(r - (1 - (int)pow(-2, s - step)) / 3, size);

        size_t skip = 0;
        while (skip < dim_work)
        {
            char *rec = work_buffer + skip;

            int dst, block_size , src;
            memcpy(&src, rec, sizeof(int));
            memcpy(&dst, rec + sizeof(int), sizeof(int));
            memcpy(&block_size, rec + 2 * sizeof(int), sizeof(int));

            size_t packet_size = header_size + (size_t)block_size;

            int logical_dst = logical_rank_for_bine_dh_root(dst, src, size);
            int logical_partner = logical_rank_for_bine_dh_root(partner, src, size);
            if (same_prefix_negabinary(logical_dst, logical_partner,s,step + 1))
            {
                memcpy(send_buffer + dim_send, rec, packet_size);
                dim_send += (int)packet_size;
            }
            else
            {
                memcpy(keep_buffer + dim_keep, rec, packet_size);
                dim_keep += (int)packet_size;
            }

            skip += packet_size;
        }

        err = MPI_Sendrecv(&dim_send, 1, MPI_INT, partner, 0, &dim_recv, 1, MPI_INT, partner, 0, comm, MPI_STATUS_IGNORE);
        if (err != MPI_SUCCESS)
            goto err_hndl;

        err = MPI_Sendrecv(send_buffer, dim_send, MPI_BYTE, partner, 1, work_buffer + dim_keep, dim_recv, MPI_BYTE, partner, 1,
                           comm, MPI_STATUS_IGNORE);
        if (err != MPI_SUCCESS)
            goto err_hndl;

        memcpy(work_buffer, keep_buffer, (size_t)dim_keep);

        dim_work = (size_t)dim_keep + (size_t)dim_recv;
    }

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

err_hndl:
    free(keep_buffer);
    free(send_buffer);
    free(work_buffer);
    return err;
}