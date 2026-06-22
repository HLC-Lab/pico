#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "$REPO_ROOT"

python3 -m plot boxplot \
    --system leonardo \
    --tasks-per-node 1 \
    --metric mean \
    --nnodes 16,32,64,128,256,512,1024,2048 \
    --exclude "block_by_block|sparbit"

python3 -m plot boxplot \
    --system lumi \
    --tasks-per-node 1 \
    --metric mean \
    --nnodes 16,32,64,128,256,512,1024

python3 -m plot boxplot \
    --system mare_nostrum \
    --tasks-per-node 1 \
    --metric mean \
    --nnodes 4,8,16,32,64 \
    --notes "UCX_MAX_RNDV_RAILS=1"

python3 -m plot boxplot \
    --system fugaku \
    --tasks-per-node 1 \
    --metric mean \
    --nnodes 2x2x2,4x4x4,8x8x8,64x64,32x256
