#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "$REPO_ROOT"

for extra_flag in "" # "--hide-y-labels"
do
    for base_system in "lumi" "leonardo"
    do
        system="$base_system"
        rm -rf "plot/${system}_hm/"*
        for metric in mean # median percentile_90
        do
            for collective in allreduce
            do
                cmd=(python3 -m plot heatmap
                    --system "$system"
                    --collective "$collective"
                    --tasks-per-node 1
                    --metric "$metric"
                )
                if [[ "$system" == "lumi" ]]; then
                    cmd+=(--nnodes 16,32,64,128,256,512,1024)
                    second_cmd=("${cmd[@]}")
                    second_cmd+=(--exclude "block_by_block|segmented")
                    [[ -n "$extra_flag" ]] && cmd+=("$extra_flag")
                    [[ -n "$extra_flag" ]] && second_cmd+=("$extra_flag")
                    "${cmd[@]}"
                    "${second_cmd[@]}"
                else
                    cmd+=(--nnodes 16,32,64,128,256,512,1024,2048 --exclude "segmented")
                    second_cmd=(python3 -m plot heatmap
                        --system "$system"
                        --collective "$collective"
                        --tasks-per-node 1
                        --metric "$metric"
                        --nnodes 16,32,64,128,256,512,1024,2048
                    )
                    [[ -n "$extra_flag" ]] && cmd+=("$extra_flag")
                    [[ -n "$extra_flag" ]] && second_cmd+=("$extra_flag")
                    "${cmd[@]}"
                    "${second_cmd[@]}"
                fi
            done
        done
    done
done
