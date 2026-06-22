#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TABLE_PY="${REPO_ROOT}/plot/table.py"

# Remove the --bvb if you want to compare to all the algorithms
for extra_params in "--bvb" #"--y_no"
do
    for base in all #binomial
    do
        system="lumi"
        for metric in mean #median percentile_90
        do
            for collective in allreduce
            do
                python3 "$TABLE_PY" --system "${system}" --collective "${collective}" --tasks_per_node 1 --metric "${metric}" --base "${base}" ${extra_params} --nnodes 16,32,64,128,256,512,1024 --exclude "block_by_block|segmented" > "${REPO_ROOT}/plot/${system}/bine_vs_binomial.txt"
            done

            for collective in allgather
            do
                python3 "$TABLE_PY" --system "${system}" --collective "${collective}" --tasks_per_node 1 --metric "${metric}" --base "${base}" ${extra_params} --nnodes 16,32,64,128,256,512,1024 >> "${REPO_ROOT}/plot/${system}/bine_vs_binomial.txt"
            done

            for collective in reduce_scatter
            do
                python3 "$TABLE_PY" --system "${system}" --collective "${collective}" --tasks_per_node 1 --metric "${metric}" --base "${base}" ${extra_params} --nnodes 16,32,64,128,256,512,1024 --exclude "block_by_block|segmented" >> "${REPO_ROOT}/plot/${system}/bine_vs_binomial.txt"
            done

            for collective in alltoall bcast reduce gather scatter
            do
                python3 "$TABLE_PY" --system "${system}" --collective "${collective}" --tasks_per_node 1 --metric "${metric}" --base "${base}" ${extra_params} --nnodes 16,32,64,128,256,512,1024 >> "${REPO_ROOT}/plot/${system}/bine_vs_binomial.txt"
            done
        done

        system="leonardo"
        for metric in mean #median percentile_90
        do
            for collective in allreduce
            do
                python3 "$TABLE_PY" --system "${system}" --collective "${collective}" --tasks_per_node 1 --metric "${metric}" --base "${base}" ${extra_params} --nnodes 128,256,512,1024,2048 --exclude "block_by_block|segmented" > "${REPO_ROOT}/plot/${system}/bine_vs_binomial.txt"
            done

            for collective in allgather
            do
                python3 "$TABLE_PY" --system "${system}" --collective "${collective}" --tasks_per_node 1 --metric "${metric}" --base "${base}" ${extra_params} --nnodes 128,256,512,1024,2048 --exclude "block_by_block|sparbit" >> "${REPO_ROOT}/plot/${system}/bine_vs_binomial.txt"
            done

            for collective in reduce_scatter
            do
                python3 "$TABLE_PY" --system "${system}" --collective "${collective}" --tasks_per_node 1 --metric "${metric}" --base "${base}" ${extra_params} --nnodes 128,256 --exclude "block_by_block|segmented" >> "${REPO_ROOT}/plot/${system}/bine_vs_binomial.txt"
            done

            for collective in alltoall bcast reduce gather scatter
            do
                python3 "$TABLE_PY" --system "${system}" --collective "${collective}" --tasks_per_node 1 --metric "${metric}" --base "${base}" ${extra_params} --nnodes 128,256 >> "${REPO_ROOT}/plot/${system}/bine_vs_binomial.txt"
            done
        done

        system="mare_nostrum"
        for metric in mean #median percentile_90
        do
            for collective in allreduce
            do
                python3 "$TABLE_PY" --system "${system}" --collective "${collective}" --tasks_per_node 1 --metric "${metric}" --base "${base}" ${extra_params} --notes "UCX_MAX_RNDV_RAILS=1" --nnodes 4,8,16,32,64 --exclude "block_by_block|segmented" > "${REPO_ROOT}/plot/${system}/bine_vs_binomial.txt"
            done

            for collective in allgather
            do
                python3 "$TABLE_PY" --system "${system}" --collective "${collective}" --tasks_per_node 1 --metric "${metric}" --base "${base}" ${extra_params} --notes "UCX_MAX_RNDV_RAILS=1" --nnodes 4,8,16,32,64 >> "${REPO_ROOT}/plot/${system}/bine_vs_binomial.txt"
            done

            for collective in reduce_scatter
            do
                python3 "$TABLE_PY" --system "${system}" --collective "${collective}" --tasks_per_node 1 --metric "${metric}" --base "${base}" ${extra_params} --notes "UCX_MAX_RNDV_RAILS=1" --nnodes 4,8,16,32,64 --exclude "block_by_block|segmented" >> "${REPO_ROOT}/plot/${system}/bine_vs_binomial.txt"
            done

            for collective in alltoall bcast reduce gather scatter
            do
                python3 "$TABLE_PY" --system "${system}" --collective "${collective}" --tasks_per_node 1 --metric "${metric}" --base "${base}" ${extra_params} --notes "UCX_MAX_RNDV_RAILS=1" --nnodes 4,8,16,32,64 >> "${REPO_ROOT}/plot/${system}/bine_vs_binomial.txt"
            done
        done

        system="fugaku"
        for collective in allreduce allgather reduce_scatter
        do
            python3 "$TABLE_PY" --system "${system}" --collective "${collective}" --nnodes 2x2x2,8x8x8,64x64,32x256 > "${REPO_ROOT}/plot/${system}/bine_vs_binomial.txt"
        done

        for collective in alltoall bcast reduce gather scatter
        do
            python3 "$TABLE_PY" --system "${system}" --collective "${collective}" --nnodes 2x2x2,8x8x8,64x64 >> "${REPO_ROOT}/plot/${system}/bine_vs_binomial.txt"
        done
    done
done
