#!/bin/bash

MIN_HOSTS=4

while true; do
    # Get list of active job IDs
    ACTIVE_JOBS=$(squeue -h -o "%A" 2>/dev/null)
    
    for JOBID in $ACTIVE_JOBS; do
        # Check if file for job ID already exists
        if [ ! -f "out/${JOBID}.txt" ]; then
            # Get the expanded list of nodes for this job
            NODES=$(scontrol show hostnames $(squeue -j "$JOBID" -h -o "%N" 2>/dev/null) 2>/dev/null)
            NUM_NODES=$(echo "$NODES" | wc -l) 
            # Save to file named [jobid].txt only if NODES is not empty
            if [ -n "$NODES" ] && [ "$NUM_NODES" -ge "$MIN_HOSTS" ]; then
                echo "$NODES" > "out/${JOBID}.txt"
            fi
	    sleep 0.1
        fi
    done
    
    # Wait before checking again (adjust sleep interval as needed)
    sleep 600
done
