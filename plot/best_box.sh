#!/bin/bash

python3 ./plot/best_box.py --system leonardo --tasks_per_node 1 --metric mean --base all --nnodes 256,512,1024,2048 --exclude "block_by_block|sparbit"
python3 ./plot/best_box.py --system lumi --tasks_per_node 1 --metric mean --base all --nnodes 16,32,64,128,256,512,1024
python3 ./plot/best_box.py --system mare_nostrum --tasks_per_node 1 --metric mean --base all --nnodes 4,8,16,32,64 --notes "UCX_MAX_RNDV_RAILS=1"
python3 ./plot/best_box.py --system fugaku --tasks_per_node 1 --metric mean --base all --nnodes 2x2x2,4x4x4,8x8x8,64x64,32x256