#!/bin/bash
# Core benchmarking loop executed on the compute nodes (invoked by submit_wrapper.sh).
# Usage: scripts/orchestrator.sh (normally via sbatch/srun; expects environment exported by submit_wrapper.sh)

# Trap SIGINT (Ctrl+C) and call cleanup function
trap cleanup SIGINT

[[ -n "$SLURM_PARAMS" ]] && inform "Sbatch params:" "$SLURM_PARAMS"

####################################################################################
#                           MAIN BENCHMARKING LOOP                                 #
####################################################################################
# WARN: The division between tui loop and normal loop is temporary, it will be changed
if [[ ! -n "$TUI_FILE" ]]; then
    iter=0
    for config in ${TEST_CONFIG_FILES[@]//,/ }; do
        export TEST_CONFIG="${config}"
        export TEST_ENV="${TEST_CONFIG}_env.sh"

        for n_gpu in ${GPU_PER_NODE[@]//,/ }; do
            cli_set_awareness_and_tasks "$n_gpu"

            if [[ "$GPU_AWARENESS" == "no" ]]; then
                cli_run_cpu_set iter || exit 1
            else
                cli_run_gpu_once iter || exit 1
            fi
        done
    done

else
    iter=0
    lib_count="${LIB_COUNT:-0}"
    if ! [[ "$lib_count" =~ ^[0-9]+$ ]] || (( lib_count == 0 )); then
        error "LIB_COUNT missing or zero in TUI env."
        exit 1
    fi
    for (( i=0; i<lib_count; i++ )); do
        run_library_tui "$i" iter || exit 1
    done
fi

success "All tests completed successfully"

###################################################################################
#              COMPRESS THE RESULTS AND DELETE THE OUTPUT DIR IF REQUESTED        #
###################################################################################
if [[ "$DEBUG_MODE" == "no" && "$DRY_RUN" == "no" && "$COMPRESS" == "yes" ]]; then
    tarball_path="$(dirname "$OUTPUT_DIR")/$(basename "$OUTPUT_DIR").tar.gz"
    if tar -czf "$tarball_path" -C "$(dirname "$OUTPUT_DIR")" "$(basename "$OUTPUT_DIR")"; then
        if [[ "$DELETE" == "yes" ]]; then
            rm -rf "$OUTPUT_DIR"
        fi
    fi
fi

[[ "$LOCATION" != "local" ]] && squeue -j $SLURM_JOB_ID
