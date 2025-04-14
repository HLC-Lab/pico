#!/bin/bash
for extra_params in "--bvb" #"--y_no"
do
    for base in all #binomial
    do
        system="lumi" 
        echo "*****************"
        echo "System: ${system}"
        echo "*****************"
        for metric in mean #median percentile_90
        do
            # For allreduce and reduce_scatter we also consider the cases where we do not segment
            for collective in allreduce
            do
                python3 ./plot/table.py --system ${system} --collective ${collective} --tasks_per_node 1 --metric ${metric} --base ${base} ${extra_params} --nnodes 16,32,64,128,256,512,1024 --exclude "block_by_block|segmented"
            done

            for collective in allgather
            do    
                python3 ./plot/table.py --system ${system} --collective ${collective} --tasks_per_node 1 --metric ${metric} --base ${base} ${extra_params} --nnodes 16,32,64,128,256,512,1024 
            done

            for collective in reduce_scatter
            do
                python3 ./plot/table.py --system ${system} --collective ${collective} --tasks_per_node 1 --metric ${metric} --base ${base} ${extra_params} --nnodes 16,32,64,128,256,512,1024 --exclude "block_by_block|segmented"
            done

            for collective in alltoall bcast reduce gather scatter
            do
                python3 ./plot/table.py --system ${system} --collective ${collective} --tasks_per_node 1 --metric ${metric} --base ${base} ${extra_params} --nnodes 16,32,64,128,256,512,1024
            done
        done

        echo ""
        echo ""

        system="leonardo"
        echo "*****************"
        echo "System: ${system}"
        echo "*****************"
        for metric in mean #median percentile_90
        do
            # For allreduce and reduce_scatter we also consider the cases where we do not segment
            for collective in allreduce
            do
                python3 ./plot/table.py --system ${system} --collective ${collective} --tasks_per_node 1 --metric ${metric} --base ${base} ${extra_params} --nnodes 128,256,512,1024,2048 --exclude "block_by_block|segmented"
            done

            for collective in allgather
            do    
                python3 ./plot/table.py --system ${system} --collective ${collective} --tasks_per_node 1 --metric ${metric} --base ${base} ${extra_params} --nnodes 128,256,512,1024,2048 --exclude "block_by_block|sparbit"
            done

            for collective in reduce_scatter
            do
                python3 ./plot/table.py --system ${system} --collective ${collective} --tasks_per_node 1 --metric ${metric} --base ${base} ${extra_params} --nnodes 128,256 --exclude "block_by_block|segmented"
            done

            for collective in alltoall bcast reduce gather scatter
            do
                python3 ./plot/table.py --system ${system} --collective ${collective} --tasks_per_node 1 --metric ${metric} --base ${base} ${extra_params} --nnodes 128,256
            done
        done
        
        echo ""
        echo ""

        system="mare_nostrum"
        echo "*****************"
        echo "System: ${system}"
        echo "*****************"        
        for metric in mean #median percentile_90
        do
            # For allreduce and reduce_scatter we also consider the cases where we do not segment
            for collective in allreduce
            do
                python3 ./plot/table.py --system ${system} --collective ${collective} --tasks_per_node 1 --metric ${metric} --base ${base} ${extra_params} --notes "UCX_MAX_RNDV_RAILS=1" --nnodes 4,8,16,32,64 --exclude "block_by_block|segmented"
            done

            for collective in allgather
            do    
                python3 ./plot/table.py --system ${system} --collective ${collective} --tasks_per_node 1 --metric ${metric} --base ${base} ${extra_params} --notes "UCX_MAX_RNDV_RAILS=1" --nnodes 4,8,16,32,64
            done

            for collective in reduce_scatter
            do
                python3 ./plot/table.py --system ${system} --collective ${collective} --tasks_per_node 1 --metric ${metric} --base ${base} ${extra_params} --notes "UCX_MAX_RNDV_RAILS=1" --nnodes 4,8,16,32,64 --exclude "block_by_block|segmented"
            done

            for collective in alltoall bcast reduce gather scatter
            do
                python3 ./plot/table.py --system ${system} --collective ${collective} --tasks_per_node 1 --metric ${metric} --base ${base} ${extra_params} --notes "UCX_MAX_RNDV_RAILS=1" --nnodes 4,8,16,32,64 
            done
        done      

        echo ""
        echo ""

        system="fugaku"
        echo "*****************"
        echo "System: ${system}"
        echo "*****************"        
        # For these we have up to 32x256
        for collective in allreduce allgather reduce_scatter
        do
            python3 plot/table.py --system fugaku --collective ${collective} --nnodes 2x2x2,8x8x8,64x64,32x256 
        done

        for collective in alltoall bcast reduce gather scatter
        do
            python3 plot/table.py --system fugaku --collective ${collective} --nnodes 2x2x2,8x8x8,64x64
        done          
    done
done
