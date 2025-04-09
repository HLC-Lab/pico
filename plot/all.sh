#!/bin/bash
# For allreduce and reduce_scatter we also consider the cases where we do not segment
for collective in allreduce
do
    python3 ./plot/heatmap.py --system leonardo --collective ${collective} --tasks_per_node 1 --nnodes 128,256,512,1024,2048 --exclude "segmented"
    python3 ./plot/heatmap.py --system leonardo --collective ${collective} --tasks_per_node 1 --nnodes 128,256,512,1024,2048 --exclude "segmented|block_by_block"
    python3 ./plot/heatmap.py --system leonardo --collective ${collective} --tasks_per_node 1 --nnodes 128,256,512,1024,2048 
done

for collective in reduce_scatter
do
    python3 ./plot/heatmap.py --system leonardo --collective ${collective} --tasks_per_node 1 --nnodes 128,256 --exclude "block_by_block"
    python3 ./plot/heatmap.py --system leonardo --collective ${collective} --tasks_per_node 1 --nnodes 128,256 
done

for collective in allgather
do    
    python3 ./plot/heatmap.py --system leonardo --collective ${collective} --tasks_per_node 1 --nnodes 128,256,512,1024,2048 
done

for collective in alltoall bcast gather reduce reduce_scatter scatter
do
    python3 ./plot/heatmap.py --system leonardo --collective ${collective} --tasks_per_node 1 --nnodes 128,256
done