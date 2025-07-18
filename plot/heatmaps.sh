#!/bin/bash
for extra_params in "" #"--y_no"
do
    for base in all #binomial
    do

        system="lumi"
        rm -rf plot/${system}_hm/*
        for metric in mean #median percentile_90
        do
            # For allreduce and reduce_scatter we also consider the cases where we do not segment
            for collective in allreduce
            do
                python3 ./plot/heatmap.py --system ${system} --collective ${collective} --tasks_per_node 1 --metric ${metric} --base ${base} ${extra_params} --nnodes 16,32,64,128,256,512,1024 
                python3 ./plot/heatmap.py --system ${system} --collective ${collective} --tasks_per_node 1 --metric ${metric} --base ${base} ${extra_params} --nnodes 16,32,64,128,256,512,1024 --exclude "block_by_block|segmented"
            done

#            for collective in reduce_scatter
#            do
#                python3 ./plot/heatmap.py --system ${system} --collective ${collective} --tasks_per_node 1 --metric ${metric} --base ${base} ${extra_params} --nnodes 16,32,64,128,256,512,1024 
#                python3 ./plot/heatmap.py --system ${system} --collective ${collective} --tasks_per_node 1 --metric ${metric} --base ${base} ${extra_params} --nnodes 16,32,64,128,256,512,1024 --exclude "block_by_block|sparbit"
#            done
#
#            for collective in allgather
#            do    
#                python3 ./plot/heatmap.py --system ${system} --collective ${collective} --tasks_per_node 1 --metric ${metric} --base ${base} ${extra_params} --nnodes 16,32,64,128,256,512,1024
#            done
#
#            for collective in alltoall bcast gather reduce scatter
#            do
#                python3 ./plot/heatmap.py --system ${system} --collective ${collective} --tasks_per_node 1 --metric ${metric} --base ${base} ${extra_params} --nnodes 16,32,64,128,256,512,1024
#            done
        done

#        system="fugaku"
#        rm -rf plot/${system}_hm/*
#        # For these we have up to 32x256
#        for collective in allreduce allgather reduce_scatter
#        do
#            python3 plot/heatmap.py --system fugaku --collective ${collective} --nnodes 2x2x2,8x8x8,64x64,32x256 --y_no
#            python3 plot/heatmap.py --system fugaku --collective ${collective} --nnodes 2x2x2,8x8x8,64x64,32x256 
#        done
#
#        for collective in alltoall scatter gather reduce bcast
#        do
#            python3 plot/heatmap.py --system fugaku --collective ${collective} --nnodes 2x2x2,8x8x8,64x64 --y_no
#            python3 plot/heatmap.py --system fugaku --collective ${collective} --nnodes 2x2x2,8x8x8,64x64
#        done


        system="leonardo"
        rm -rf plot/${system}_hm/*
        for metric in mean #median percentile_90
        do
            # For allreduce and reduce_scatter we also consider the cases where we do not segment
            for collective in allreduce
            do
                python3 ./plot/heatmap.py --system ${system} --collective ${collective} --tasks_per_node 1 --metric ${metric} --base ${base} ${extra_params} --nnodes 16,32,64,128,256,512,1024,2048 --exclude "segmented"
                python3 ./plot/heatmap.py --system ${system} --collective ${collective} --tasks_per_node 1 --metric ${metric} --base ${base} ${extra_params} --nnodes 16,32,64,128,256,512,1024,2048 
            done

#            for collective in reduce_scatter
#            do
#                python3 ./plot/heatmap.py --system ${system} --collective ${collective} --tasks_per_node 1 --metric ${metric} --base ${base} ${extra_params} --nnodes 4,8,16,32,64,128,256 --exclude "block_by_block|sparbit"
#                python3 ./plot/heatmap.py --system ${system} --collective ${collective} --tasks_per_node 1 --metric ${metric} --base ${base} ${extra_params} --nnodes 4,8,16,32,64,128,256 
#            done
#
#            for collective in allgather
#            do    
#                python3 ./plot/heatmap.py --system ${system} --collective ${collective} --tasks_per_node 1 --metric ${metric} --base ${base} ${extra_params} --nnodes 4,8,16,32,64,128,256,512,1024,2048 --exclude "block_by_block|sparbit"
#                python3 ./plot/heatmap.py --system ${system} --collective ${collective} --tasks_per_node 1 --metric ${metric} --base ${base} ${extra_params} --nnodes 4,8,16,32,64,128,256,512,1024,2048 
#            done
#
#            for collective in alltoall bcast gather reduce scatter
#            do
#                python3 ./plot/heatmap.py --system ${system} --collective ${collective} --tasks_per_node 1 --metric ${metric} --base ${base} ${extra_params} --nnodes 4,8,16,32,64,128,256
#            done
        done

#        system="mare_nostrum"
#        rm -rf plot/${system}_hm/*
#        for metric in mean #median percentile_90
#        do
#            # For allreduce and reduce_scatter we also consider the cases where we do not segment
#            for collective in allreduce
#            do
#                python3 ./plot/heatmap.py --system ${system} --collective ${collective} --tasks_per_node 1 --metric ${metric} --base ${base} ${extra_params} --notes "UCX_MAX_RNDV_RAILS=1" --nnodes 4,8,16,32,64
#            done
#
#            for collective in reduce_scatter
#            do
#                python3 ./plot/heatmap.py --system ${system} --collective ${collective} --tasks_per_node 1 --metric ${metric} --base ${base} ${extra_params} --notes "UCX_MAX_RNDV_RAILS=1" --nnodes 4,8,16,32,64
#            done
#
#            for collective in allgather
#            do    
#                python3 ./plot/heatmap.py --system ${system} --collective ${collective} --tasks_per_node 1 --metric ${metric} --base ${base} ${extra_params} --notes "UCX_MAX_RNDV_RAILS=1" --nnodes 4,8,16,32,64
#            done
#
#            for collective in alltoall bcast gather reduce scatter
#            do
#                python3 ./plot/heatmap.py --system ${system} --collective ${collective} --tasks_per_node 1 --metric ${metric} --base ${base} ${extra_params} --notes "UCX_MAX_RNDV_RAILS=1" --nnodes 4,8,16,32,64
#            done
#        done        
    done
done
