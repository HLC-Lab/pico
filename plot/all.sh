#!/bin/bash
for extra_params in "" #"--y_no"
do
    for base in all #binomial
    do



        system="mare_nostrum"
        rm -rf plot/${system}_hm/*
        for metric in mean #median percentile_90
        do
            # For allreduce and reduce_scatter we also consider the cases where we do not segment
            for collective in allreduce
            do
                python3 ./plot/heatmap_new.py --system ${system} --collective ${collective} --tasks_per_node 1 --metric ${metric} --base ${base} ${extra_params} --notes "UCX_MAX_RNDV_RAILS=1" --nnodes 4,8,16,32,64
            done

            for collective in reduce_scatter
            do
                python3 ./plot/heatmap_new.py --system ${system} --collective ${collective} --tasks_per_node 1 --metric ${metric} --base ${base} ${extra_params} --notes "UCX_MAX_RNDV_RAILS=1" --nnodes 4,8,16,32,64
            done

            for collective in allgather
            do    
                python3 ./plot/heatmap_new.py --system ${system} --collective ${collective} --tasks_per_node 1 --metric ${metric} --base ${base} ${extra_params} --notes "UCX_MAX_RNDV_RAILS=1" --nnodes 4,8,16,32,64
            done

            for collective in alltoall bcast gather reduce scatter
            do
                python3 ./plot/heatmap_new.py --system ${system} --collective ${collective} --tasks_per_node 1 --metric ${metric} --base ${base} ${extra_params} --notes "UCX_MAX_RNDV_RAILS=1" --nnodes 4,8,16,32,64
            done
        done        
    done
done


#PAPER_DIR="/mnt/c/Users/ddese/Dropbox/Apps/ShareLaTeX/SC25 - Bine Trees/"
#for system in leonardo lumi mare_nostrum fugaku
#do
#    mkdir -p "${PAPER_DIR}/plots/${system}_hm/"
#    rm -rf "${PAPER_DIR}/plots/${system}_hm/*"
#    cp -r plot/${system}_hm/* "${PAPER_DIR}/plots/${system}_hm/"
#done