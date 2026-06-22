#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "$REPO_ROOT"

rm -rf plot/leonardo/
rm -rf plot/lumi/
rm -rf plot/mare_nostrum/

run_comparison() {
    python3 -m plot comparison-heatmap "$@"
}

# Ring algorithm
run_comparison --system leonardo --collective allreduce --nnodes 32,64,128,256,512,1024,2048 --target-algo ring_ompi --exclude bine,over --metric median
run_comparison --system lumi --collective allreduce --nnodes 8,16,32,64,128,256,1024 --target-algo ring_over --exclude bine,over --metric median
run_comparison --system mare_nostrum --collective allreduce --nnodes 4,8,16,32,64 --target-algo ring_ompi --exclude bine,over --notes UCX_MAX_RNDV_RAILS=1 --metric median

# Rabenseifner algorithm
run_comparison --system leonardo --collective allreduce --nnodes 32,64,128,256,512,1024,2048 --target-algo rabenseifner_ompi --exclude bine,over --metric median
run_comparison --system lumi --collective allreduce --nnodes 8,16,32,64,128,256,1024 --target-algo rabenseifner_mpich --exclude bine,over --metric median
run_comparison --system mare_nostrum --collective allreduce --nnodes 4,8,16,32,64 --target-algo rabenseifner_ompi --exclude bine,over --notes UCX_MAX_RNDV_RAILS=1 --metric median

# Untuned default algorithm
run_comparison --system leonardo --collective allreduce --nnodes 32,64,128,256,512,1024,2048 --target-algo default_ompi --exclude bine,over --metric median
run_comparison --system lumi --collective allreduce --nnodes 8,16,32,64,128,256,1024 --target-algo default_mpich --exclude bine,over --metric median
run_comparison --system mare_nostrum --collective allreduce --nnodes 4,8,16,32,64 --target-algo default_ompi --exclude bine,over --notes UCX_MAX_RNDV_RAILS=1 --metric median

# NCCL Ring
run_comparison --system mare_nostrum --collective allreduce --nnodes 4,8,16,32,64 --target-algo allreduce_nccl_ring --exclude hier --metric median --notes vsnccl --tasks-per-node 4
