#!/bin/bash
PLOT_TYPE="box"
for collective in alltoall allgather allreduce reduce scatter gather bcast reduce_scatter
do
    python3 ./plot/violin_compressed.py --collective ${collective} --tasks_per_node 1 --metric mean --systems leonardo,lumi,mare_nostrum,fugaku --notes null,null,UCX_MAX_RNDV_RAILS=1,null --plot_type ${PLOT_TYPE}
done

PAPER_DIR="/mnt/c/Users/ddese/Dropbox/Apps/ShareLaTeX/SC25 - Bine Trees/"
mkdir -p "${PAPER_DIR}/plots/violin_compr/"
rm -rf "${PAPER_DIR}/plots/violin_compr/*"
cp -r plot/violin_compr/* "${PAPER_DIR}/plots/violin_compr/"